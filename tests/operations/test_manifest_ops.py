"""Op-level oracles for the manifest / MCP-status operations.

Covers ``update_manifest``, ``reload_mcp``, ``reload_failed_mcps``,
``list_failed_mcps`` and ``deregister_mcp``. The runtime ops (class a) apply on this
worker when it is a target, then broadcast on the bus; the response is the per-origin
fleet report, and a local-apply raise aborts before anything is broadcast. An unknown
title is a loud ``NotFoundError`` before any broadcast. Tier/destructive/projection
metadata is pinned too (``update_manifest`` is tier-2, off the default surface).
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from tai42_contract.app import tai42_app
from tai42_contract.manifest import ApiToolsConfig
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.app import instance
from tai42_skeleton.app.bus import LocalApplyResult, OpOutcome
from tai42_skeleton.operations import BadRequestError, NotFoundError, OperationRegistry, operation_metadata_of
from tai42_skeleton.operations import manifest as manifest_ops
from tai42_skeleton.operations.projection import is_tier2, project_operations
from tests._fakes.bus import FakeBus


class _Admin:
    def __init__(
        self,
        *,
        live_manifest: dict | None = None,
        results: dict[str, object] | None = None,
        raise_for: str | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.live_manifest = live_manifest if live_manifest is not None else {"mcp": [{"title": "svc"}]}
        self._results = results or {}
        self._raise_for = raise_for

    def _dispatch(self, method: str, *args: object) -> object:
        self.calls.append((method, *args))
        if self._raise_for == method:
            raise RuntimeError(f"{method} failed")
        return self._results.get(method)

    def list_failed_mcps(self) -> object:
        return self._dispatch("list_failed_mcps")

    def reload_mcp(self, title: str) -> object:
        return self._dispatch("reload_mcp", title)

    def reload_failed_mcps(self) -> object:
        return self._dispatch("reload_failed_mcps")

    def deregister_mcp(self, title: str) -> object:
        return self._dispatch("deregister_mcp", title)


def _install(
    monkeypatch: pytest.MonkeyPatch, *, admin: _Admin, backend: object = None, bus: FakeBus | None = None
) -> FakeBus:
    impl = SimpleNamespace(admin=admin, backends=SimpleNamespace(backend=backend))
    monkeypatch.setattr(tai42_app, "_impl", impl)
    bus = bus or FakeBus()
    monkeypatch.setattr(instance.app, "_bus", bus)
    return bus


# -- update_manifest (persist-through via the ConfigService pipeline) ----------


class _ReplaceStore:
    """A config manager whose ``replace_manifest`` records and persists the whole
    posted document — the seam the update_manifest pipeline drives."""

    def __init__(self, *, manifest: dict | None = None, env: dict | None = None) -> None:
        self.manifest: dict = manifest if manifest is not None else {}
        self.env: dict = env if env is not None else {}
        self.replaced: list[dict] = []

    def replace_manifest(self, document: dict) -> dict:
        self.replaced.append(dict(document))
        self.manifest = dict(document)
        return dict(document)

    def read_manifest_preserved(self) -> dict:
        return dict(self.manifest)

    def read_env(self) -> dict:
        return dict(self.env)


class _MutateStore:
    """A config manager whose ``mutate_manifest`` runs the guarded mutator on a copy of
    the stored document and persists only if it returns without raising — the seam the
    set_mcp_config pipeline drives (a raise inside leaves the store untouched)."""

    def __init__(self, *, manifest: dict | None = None) -> None:
        self.manifest: dict = manifest if manifest is not None else {}
        self.persisted: list[dict] = []

    def mutate_manifest(self, mutator: Any) -> dict:
        document = deepcopy(self.manifest)
        mutator(document)  # a raise here propagates before any persist
        self.manifest = document
        self.persisted.append(deepcopy(document))
        return document

    def read_manifest_preserved(self) -> dict:
        return deepcopy(self.manifest)


class _ReloadAdmin:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result if result is not None else {"status": "ok", "env_keys": 0}
        self.calls = 0

    def reload_config(self) -> dict:
        self.calls += 1
        return self.result


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, store: _ReplaceStore, admin: _ReloadAdmin, backend: object = None
) -> FakeBus:
    impl = SimpleNamespace(
        config=SimpleNamespace(config_manager=store),
        admin=admin,
        backends=SimpleNamespace(backend=backend),
    )
    monkeypatch.setattr(tai42_app, "_impl", impl)
    bus = FakeBus()
    monkeypatch.setattr(instance.app, "_bus", bus)
    return bus


async def test_update_manifest_persists_through_and_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _ReplaceStore()
    admin = _ReloadAdmin({"status": "ok", "env_keys": 2})
    bus = _install_pipeline(monkeypatch, store=store, admin=admin)

    result = await manifest_ops.update_manifest("mcp: []\n")

    # The whole posted document is validated, persisted, reloaded locally, and the
    # reload broadcast to the WHOLE fleet (targets None); a lone worker collapses the
    # fan-out to the local note.
    assert store.replaced == [{"mcp": []}]
    assert admin.calls == 1
    assert bus.publish_calls[0][0] == {"op": "reload_config"}
    assert bus.publish_calls[0][1] is None
    assert result == {
        "status": "ok",
        "env_keys": 2,
        "fanout": {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"},
    }


async def test_update_manifest_persists_env_markers_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    # A marker-carrying document pushed through the replace surface persists with its
    # ``!ENV`` markers INTACT — the resolved value is used only for in-memory
    # validation, so no secret ever bakes to disk.
    monkeypatch.setenv("TAI_BUS_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TAI_BACKEND", "myapp.backend")
    reset_all_settings()
    try:
        store = _ReplaceStore()
        _install_pipeline(monkeypatch, store=store, admin=_ReloadAdmin())

        await manifest_ops.update_manifest("backend_module: !ENV ${TAI_BACKEND}\n")

        # The marker string survives verbatim — never the resolved ``myapp.backend``.
        assert store.replaced == [{"backend_module": "!ENV ${TAI_BACKEND}"}]
    finally:
        reset_all_settings()


async def test_update_manifest_backend_without_bus_maps_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    # A replacement that registers a task backend with no worker bus configured is
    # refused by ConfigService's backend-needs-bus invariant, which raises the
    # RuntimeError ``BackendNeedsBusError`` at MUTATE time. The op must map it to a loud
    # 400 naming TAI_BUS_REDIS_URL, not let it escape as an unhandled 500.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()
    try:
        store = _ReplaceStore()
        _install_pipeline(monkeypatch, store=store, admin=_ReloadAdmin())

        with pytest.raises(BadRequestError, match="TAI_BUS_REDIS_URL"):
            await manifest_ops.update_manifest("backend_module: myapp.backend\n")

        assert store.replaced == []  # rejected in validation, before any persist
    finally:
        reset_all_settings()


# -- set_mcp_config (persist-through via the ConfigService pipeline) -----------


def _install_mutate_pipeline(monkeypatch: pytest.MonkeyPatch, *, store: _MutateStore, admin: _ReloadAdmin) -> FakeBus:
    impl = SimpleNamespace(
        config=SimpleNamespace(config_manager=store),
        admin=admin,
        backends=SimpleNamespace(backend=None),
    )
    monkeypatch.setattr(tai42_app, "_impl", impl)
    bus = FakeBus()
    monkeypatch.setattr(instance.app, "_bus", bus)
    return bus


async def test_set_mcp_config_backend_without_bus_maps_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    # The already-registered backend plus the new mcp section resolves to a config that
    # needs the bus; with none configured, ConfigService raises the RuntimeError
    # ``BackendNeedsBusError`` at MUTATE time. The op must map it to a loud 400 naming
    # TAI_BUS_REDIS_URL, not let it escape as an unhandled 500.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()
    try:
        store = _MutateStore(manifest={"backend_module": "myapp.backend"})
        _install_mutate_pipeline(monkeypatch, store=store, admin=_ReloadAdmin())

        with pytest.raises(BadRequestError, match="TAI_BUS_REDIS_URL"):
            await manifest_ops.set_mcp_config([])

        assert store.persisted == []  # rejected in validation, before any persist
    finally:
        reset_all_settings()


# -- list_failed_mcps -------------------------------


async def test_list_failed_mcps_untargeted_reads_local_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(results={"list_failed_mcps": [{"title": "redis", "status": "unavailable"}]})
    bus = _install(monkeypatch, admin=admin)

    result = await manifest_ops.list_failed_mcps()

    # A query rides the same fan-out shape: this worker's list is its self-entry payload.
    assert bus.publish_calls == [
        (
            {"op": "list_failed_mcps"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload=[{"title": "redis", "status": "unavailable"}]),
        )
    ]
    assert result["results"][0]["payload"] == [{"title": "redis", "status": "unavailable"}]


async def test_list_failed_mcps_targeted_to_remote_skips_local(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(results={"list_failed_mcps": []})
    bus = _install(monkeypatch, admin=admin, bus=FakeBus(remotes=["serve-w1"]))

    result = await manifest_ops.list_failed_mcps(["serve-w1"])

    assert admin.calls == []  # self not targeted → no local read
    assert bus.publish_calls == [({"op": "list_failed_mcps"}, ["serve-w1"], None)]
    assert {r["origin"] for r in result["results"]} == {"serve-w1"}


# -- reload_mcp ---------


async def test_reload_mcp_untargeted_applies_locally_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(
        live_manifest={"mcp": [{"title": "svc"}]},
        results={"reload_mcp": {"title": "svc", "status": "ok", "tools": ["t1"]}},
    )
    bus = _install(monkeypatch, admin=admin)

    result = await manifest_ops.reload_mcp("svc")

    assert admin.calls == [("reload_mcp", "svc")]
    assert bus.publish_calls == [
        (
            {"op": "reload_mcp", "title": "svc"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload={"title": "svc", "status": "ok", "tools": ["t1"]}),
        )
    ]
    assert result["results"][0]["payload"] == {"title": "svc", "status": "ok", "tools": ["t1"]}


async def test_reload_mcp_targeted_to_remote_skips_local(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(
        live_manifest={"mcp": [{"title": "svc"}]},
        results={"reload_mcp": {"title": "svc", "status": "ok"}},
    )
    bus = _install(monkeypatch, admin=admin, bus=FakeBus(remotes=["serve-w1"]))

    result = await manifest_ops.reload_mcp("svc", ["serve-w1"])

    assert admin.calls == []  # self not targeted → no local re-probe
    assert bus.publish_calls == [({"op": "reload_mcp", "title": "svc"}, ["serve-w1"], None)]
    assert {r["origin"] for r in result["results"]} == {"serve-w1"}


async def test_reload_mcp_unknown_title_404(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(live_manifest={"mcp": [{"title": "svc"}]})
    bus = _install(monkeypatch, admin=admin)

    with pytest.raises(NotFoundError, match="unknown mcp title"):
        await manifest_ops.reload_mcp("nope")
    # 404 precedes any broadcast.
    assert bus.publish_calls == []


async def test_reload_mcp_unknown_target_raises_before_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(live_manifest={"mcp": [{"title": "svc"}]})
    bus = _install(monkeypatch, admin=admin)

    with pytest.raises(BadRequestError, match="unknown fleet targets"):
        await manifest_ops.reload_mcp("svc", ["ghost"])
    assert admin.calls == []
    assert bus.publish_calls == []


async def test_reload_mcp_local_apply_raise_aborts_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(live_manifest={"mcp": [{"title": "svc"}]}, raise_for="reload_mcp")
    bus = _install(monkeypatch, admin=admin)

    with pytest.raises(RuntimeError, match="reload_mcp failed"):
        await manifest_ops.reload_mcp("svc")
    assert bus.publish_calls == []


# -- reload_failed_mcps -----------------------------


async def test_reload_failed_mcps_untargeted_applies_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(results={"reload_failed_mcps": [{"title": "svc", "status": "ok"}]})
    bus = _install(monkeypatch, admin=admin)

    result = await manifest_ops.reload_failed_mcps()

    assert admin.calls == [("reload_failed_mcps",)]
    assert bus.publish_calls == [
        (
            {"op": "reload_failed_mcps"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload=[{"title": "svc", "status": "ok"}]),
        )
    ]
    assert result["results"][0]["payload"] == [{"title": "svc", "status": "ok"}]


# -- deregister_mcp ---------------------------------


async def test_deregister_mcp_untargeted_applies_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(results={"deregister_mcp": {"title": "svc", "status": "ok", "removed": ["svc_t"]}})
    bus = _install(monkeypatch, admin=admin)

    result = await manifest_ops.deregister_mcp("svc")

    assert admin.calls == [("deregister_mcp", "svc")]
    assert bus.publish_calls == [
        (
            {"op": "deregister_mcp", "title": "svc"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload={"title": "svc", "status": "ok", "removed": ["svc_t"]}),
        )
    ]
    assert result["results"][0]["payload"] == {"title": "svc", "status": "ok", "removed": ["svc_t"]}


# -- destructive / reload-gate / tier metadata -------------------------------


def test_mutating_ops_are_destructive_and_reload_gated() -> None:
    for op in (
        manifest_ops.set_mcp_config,
        manifest_ops.reload_mcp,
        manifest_ops.update_manifest,
        manifest_ops.reload_failed_mcps,
        manifest_ops.deregister_mcp,
    ):
        meta = operation_metadata_of(op)
        assert meta.destructive is True, meta.name
        assert meta.reload_gated is True, meta.name


def test_read_ops_are_not_destructive() -> None:
    for op in (
        manifest_ops.get_manifest,
        manifest_ops.get_mcp_config_schema,
        manifest_ops.get_mcp_status,
        manifest_ops.list_failed_mcps,
    ):
        meta = operation_metadata_of(op)
        assert meta.destructive is False, meta.name
        assert meta.reload_gated is False, meta.name


def test_update_manifest_is_tier2_and_off_the_default_surface() -> None:
    update_meta = operation_metadata_of(manifest_ops.update_manifest)
    assert update_meta.authority_changing is True
    assert is_tier2(update_meta) is True

    reg = OperationRegistry()
    for op in (manifest_ops.update_manifest, manifest_ops.reload_mcp):
        reg.register(operation_metadata_of(op))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, Any] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = annotations
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    # Default surface: update_manifest is off (tier-2); reload_mcp projects with a
    # destructiveHint.
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert "update_manifest" not in names
    assert "reload_mcp" in names
    assert app.tools.registered["reload_mcp"].destructiveHint is True

    # Includable via explicit api_tools.include.
    app2 = SimpleNamespace(tools=_Rec())
    names2 = project_operations(
        app2, ApiToolsConfig(include=["update_manifest"], expose_destructive=True), registry=reg
    )
    assert "update_manifest" in names2
    assert app2.tools.registered["update_manifest"].destructiveHint is True
