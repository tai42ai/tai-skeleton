"""Op-level oracles for the config operations.

``reload_config`` is a convergence op: it applies locally when this worker is a
target, then broadcasts, embedding the per-origin fleet report; a failed local
reload publishes anyway and re-raises with the report. The projection carries
``destructiveHint``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai_contract.app import tai_app
from tai_contract.manifest import ApiToolsConfig

from tai_skeleton.app import instance
from tai_skeleton.app.bus import OpOutcome
from tai_skeleton.operations import BadRequestError, OperationRegistry, operation_metadata_of
from tai_skeleton.operations import config as config_ops
from tai_skeleton.operations._broadcast import FleetBroadcastError
from tai_skeleton.operations.projection import project_operations
from tests._fakes.bus import FakeBus


class _Admin:
    def __init__(self, result: object, *, raise_reload: Exception | None = None) -> None:
        self._result = result
        self._raise_reload = raise_reload
        self.calls = 0

    def reload_config(self) -> object:
        self.calls += 1
        if self._raise_reload is not None:
            raise self._raise_reload
        return self._result


def _install(
    monkeypatch: pytest.MonkeyPatch, *, admin: _Admin, manager: object = None, bus: FakeBus | None = None
) -> FakeBus:
    impl = SimpleNamespace(
        config=SimpleNamespace(config_manager=manager),
        admin=admin,
        backends=SimpleNamespace(backend=None),
    )
    monkeypatch.setattr(tai_app, "_impl", impl)
    bus = bus or FakeBus()
    monkeypatch.setattr(instance.app, "_bus", bus)
    return bus


# -- reload_config (convergence op, class c) ---------------


async def test_reload_config_untargeted_applies_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin({"status": "ok", "env_keys": 3})
    bus = _install(monkeypatch, admin=admin)

    result = await config_ops.reload_config()

    assert admin.calls == 1
    assert bus.publish_calls[0][0] == {"op": "reload_config"}
    assert bus.publish_calls[0][1] is None
    assert result["results"][0]["payload"] == {"status": "ok", "env_keys": 3}


async def test_reload_config_targeted_to_remote_skips_local(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin({"status": "ok"})
    bus = _install(monkeypatch, admin=admin, bus=FakeBus(remotes=["serve-w1"]))

    result = await config_ops.reload_config(["serve-w1"])

    assert admin.calls == 0  # self not targeted → no local reload
    assert bus.publish_calls == [({"op": "reload_config"}, ["serve-w1"], None)]
    assert {r["origin"] for r in result["results"]} == {"serve-w1"}


async def test_reload_config_local_failure_still_broadcasts_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Convergence: a failed local reload must NOT abort the broadcast — the door
    # exists to heal stale siblings — so it publishes anyway (self entry = failed)
    # and re-raises with the fleet report attached.
    admin = _Admin({"status": "ok"}, raise_reload=RuntimeError("local reload boom"))
    bus = _install(monkeypatch, admin=admin)

    with pytest.raises(FleetBroadcastError) as exc:
        await config_ops.reload_config()

    assert len(bus.publish_calls) == 1
    self_result = bus.publish_calls[0][2]
    assert self_result is not None
    assert self_result.outcome == OpOutcome.failed
    assert exc.value.report.op == "reload_config"


# -- write_env ----------------------------------------------------------------


async def test_write_env_maps_manager_value_error_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Manager:
        def read_env(self) -> dict:
            return {}

        def read_manifest_preserved(self) -> dict:
            return {}

        def write_env(self, config: dict) -> None:
            raise ValueError("invalid env key 'BAD KEY'")

    _install(monkeypatch, admin=_Admin({"status": "ok"}), manager=_Manager())

    with pytest.raises(BadRequestError, match="invalid env key"):
        await config_ops.write_env({"BAD KEY": "v"})


async def test_write_env_local_only_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Manager:
        def __init__(self) -> None:
            self.written: list[dict] = []
            self.env: dict[str, str] = {}

        def read_env(self) -> dict:
            return dict(self.env)

        def read_manifest_preserved(self) -> dict:
            # No backend registered, so the env-change invariant has nothing to reject.
            return {}

        def write_env(self, config: dict) -> None:
            self.written.append(config)
            self.env = {**self.env, **config}

    manager = _Manager()
    admin = _Admin({"status": "ok", "env_keys": 1})
    _install(monkeypatch, admin=admin, manager=manager)

    result = await config_ops.write_env({"NEW": "v"})

    # The env was validated (effective config resolved), persisted, reloaded locally,
    # and the reload broadcast; a lone worker collapses the fan-out to the local note.
    assert manager.written == [{"NEW": "v"}]
    assert result == {
        "status": "ok",
        "env_keys": 1,
        "fanout": {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"},
    }


async def test_write_env_removing_bus_with_backend_maps_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    # A registered backend needs the bus; an env change that empties TAI_BUS_REDIS_URL
    # leaves it with none, so ConfigService raises the RuntimeError
    # ``BackendNeedsBusError`` at MUTATE time. The op must map it to a loud 400 naming
    # TAI_BUS_REDIS_URL, not let it escape as an unhandled 500.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)

    class _Manager:
        def __init__(self) -> None:
            self.written: list[dict] = []

        def read_env(self) -> dict:
            # The bus is configured only through the stored env; the change empties it.
            return {"TAI_BUS_REDIS_URL": "redis://localhost:6379/0"}

        def read_manifest_preserved(self) -> dict:
            return {"backend_module": "myapp.backend"}

        def write_env(self, config: dict) -> None:
            self.written.append(config)

    manager = _Manager()
    _install(monkeypatch, admin=_Admin({"status": "ok"}), manager=manager)

    with pytest.raises(BadRequestError, match="TAI_BUS_REDIS_URL"):
        await config_ops.write_env({"TAI_BUS_REDIS_URL": ""})

    assert manager.written == []  # rejected in validation, before any write


# -- projection ---------------------------------------------------------------


def test_reload_config_projects_with_destructive_hint() -> None:
    reg = OperationRegistry()
    reg.register(operation_metadata_of(config_ops.reload_config))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert "reload_config" in names  # default-in (not authority-changing), destructive
    assert app.tools.registered["reload_config"]["annotations"].destructiveHint is True
