"""Unit oracles for :class:`~tai_skeleton.config.service.ConfigService` — the single
manifest-mutation pipeline.

Each test drives the service against a fake config store (the transactional seams),
a fake reload admin, and a fake worker bus, asserting the pipeline's ordering and
its failure discipline:

* validation runs on the RESOLVED projection and rejects before any persist;
* the mutator is pure / re-runnable;
* ``apply_replace`` validates BEFORE it persists;
* a local reload that fails after the persist landed still broadcasts, then re-raises
  with the fleet report attached;
* an unconfirmed origin is a loud ERROR log but a returned success;
* the bus-unreachable shape is a returned success too;
* the backend-needs-bus invariant rejects both directions.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

import pytest
from pyaml_env import parse_config
from pydantic import ValidationError
from tai_kit.settings import reset_all_settings
from tai_kit.utils.data import dump_manifest

from tai_skeleton.app import instance
from tai_skeleton.app.boot_rules import BackendNeedsBusError
from tai_skeleton.app.bus import FleetResult, LocalApplyResult, OpOutcome, OriginResult
from tai_skeleton.config.secret_seal import ResolvedSecretError
from tai_skeleton.config.service import ConfigService
from tai_skeleton.operations._broadcast import FleetBroadcastError, apply_response
from tests._fakes.bus import FakeBus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeConfigStore:
    """A config manager whose transactional seams persist into an in-memory document.

    ``mutate_manifest`` runs the mutator on a fresh copy of the stored PRESERVED
    document and persists it only if the mutator returns without raising — so an
    aborting mutator leaves the store untouched, exactly like the real transaction.
    """

    def __init__(self, *, manifest: dict[str, Any] | None = None, env: dict[str, str] | None = None) -> None:
        self.manifest: dict[str, Any] = manifest if manifest is not None else {}
        self.env: dict[str, str] = env if env is not None else {}
        self.persisted: list[dict[str, Any]] = []
        self.env_writes: list[dict[str, str]] = []

    def mutate_manifest(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        document = copy.deepcopy(self.manifest)
        mutator(document)  # a raise here propagates before any persist
        self.manifest = document
        self.persisted.append(copy.deepcopy(document))
        return document

    def replace_manifest(self, document: dict[str, Any]) -> dict[str, Any]:
        self.manifest = copy.deepcopy(document)
        self.persisted.append(copy.deepcopy(document))
        return self.manifest

    def write_env(self, config: dict[str, str]) -> None:
        self.env_writes.append(dict(config))
        self.env = {**self.env, **config}

    def read_env(self) -> dict[str, str]:
        return dict(self.env)

    def read_manifest_preserved(self) -> dict[str, Any]:
        return copy.deepcopy(self.manifest)


class RetryingConfigStore(FakeConfigStore):
    """A store that re-runs the mutator (as the k8s optimistic-concurrency retry does)
    before persisting, so a test can prove the guarded mutator is re-runnable."""

    def mutate_manifest(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        # First attempt is discarded (a simulated conflict); the second is persisted.
        mutator(copy.deepcopy(self.manifest))
        return super().mutate_manifest(mutator)


class FakeReloadAdmin:
    def __init__(self, *, result: dict[str, Any] | None = None, raise_reload: Exception | None = None) -> None:
        self._result = result if result is not None else {"status": "ok", "env_keys": 0}
        self._raise = raise_reload
        self.calls = 0

    def reload_config(self) -> dict[str, Any]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._result


class RecordingBus:
    """A worker bus that records each publish and returns a crafted fleet report.

    ``remote_outcome`` shapes the report of a configured remote origin (``applied`` for
    a converged fleet, ``missing`` for an unconfirmed one); ``reachable=False`` returns
    the bus-unreachable shape (no origin list, only an error). ``publish_error`` makes
    ``publish`` RAISE that exception after recording the call — a non-transport
    broadcast fault (e.g. a redis ``ResponseError``) that the bus does not fold into a
    returned bus-unreachable report."""

    def __init__(
        self,
        *,
        remotes: list[str] | None = None,
        remote_outcome: OpOutcome = OpOutcome.applied,
        reachable: bool = True,
        error: str | None = None,
        publish_error: Exception | None = None,
    ) -> None:
        self.origin = "serve-test"
        self._remotes = remotes or []
        self._remote_outcome = remote_outcome
        self._reachable = reachable
        self._error = error
        self._publish_error = publish_error
        self.publish_calls: list[tuple[dict[str, Any], list[str] | None, LocalApplyResult | None]] = []

    async def publish(
        self, op: dict[str, Any], targets: list[str] | None, local: LocalApplyResult | None
    ) -> FleetResult:
        self.publish_calls.append((op, targets, local))
        if self._publish_error is not None:
            raise self._publish_error
        if not self._reachable:
            return FleetResult(op=op["op"], reachable=False, error=self._error)
        results: list[OriginResult] = []
        if local is not None:
            results.append(
                OriginResult(origin=self.origin, outcome=local.outcome, payload=local.payload, error=local.error)
            )
        for remote in self._remotes:
            results.append(OriginResult(origin=remote, outcome=self._remote_outcome, detail="crafted"))
        return FleetResult(op=op["op"], results=results)


def _service(
    store: FakeConfigStore, *, admin: FakeReloadAdmin | None = None, bus: RecordingBus | None = None
) -> tuple[ConfigService, FakeReloadAdmin, RecordingBus]:
    admin = admin or FakeReloadAdmin()
    bus = bus or RecordingBus()
    service = ConfigService(config_manager=store, admin=admin, bus=cast("Any", bus))
    return service, admin, bus


@pytest.fixture(autouse=True)
def _reset_settings_after() -> Iterator[None]:
    yield
    reset_all_settings()


def _with_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_BUS_REDIS_URL", "redis://localhost:6379/0")
    reset_all_settings()


def _no_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()


# ---------------------------------------------------------------------------
# apply_change
# ---------------------------------------------------------------------------


async def test_apply_change_mutates_validates_persists_reloads_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    service, admin, bus = _service(store)

    def add_server(document: dict[str, Any]) -> None:
        document["mcp"] = [*document.get("mcp", []), {"title": "srv", "config": {"url": "http://x"}}]

    result = await service.apply_change(add_server)

    # Persisted the mutated document exactly once, then reloaded locally, then
    # broadcast the reload to the WHOLE fleet with the local result as `local`.
    assert store.persisted == [{"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}]
    assert admin.calls == 1
    assert len(bus.publish_calls) == 1
    op, targets, local = bus.publish_calls[0]
    assert op == {"op": "reload_config"}
    assert targets is None
    assert local is not None
    assert local.outcome == OpOutcome.applied
    # ApplyResult carries the persisted document, the local reload result, and the report.
    assert result.document == {"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}
    assert result.local == {"status": "ok", "env_keys": 0}
    assert result.fleet.op == "reload_config"
    assert result.fleet.ok is True


async def test_apply_result_fanout_is_the_apply_response_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    # fleet_fanout reads the process bus origin to decide local-only vs fleet, so drive
    # the pipeline through that same bus (installed as instance.app.bus) — the value the
    # connector writers thread must equal exactly what apply_response embeds.
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    bus = FakeBus(origin="serve-test", remotes=["serve-w1"])
    monkeypatch.setattr(instance.app, "_bus", bus)
    service, _admin, _bus = _service(store, bus=cast("Any", bus))

    result = await service.apply_replace({"mcp": []})

    assert result.fanout == apply_response(result)["fanout"]
    assert result.fanout["mode"] == "fleet"
    assert {r["origin"] for r in result.fanout["results"]} == {"serve-test", "serve-w1"}


async def test_apply_change_invalid_manifest_raises_with_nothing_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    service, admin, bus = _service(store)

    def break_manifest(document: dict[str, Any]) -> None:
        document["tools"] = "not-a-list"  # fails Manifest schema validation

    with pytest.raises(ValidationError):
        await service.apply_change(break_manifest)

    # Validation rejected inside the transaction: nothing persisted, no reload, no broadcast.
    assert store.persisted == []
    assert store.manifest == {"mcp": []}
    assert admin.calls == 0
    assert bus.publish_calls == []


async def test_apply_change_mutator_rerun_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    # The store re-runs the mutator (a simulated concurrency conflict) before it
    # persists; a pure mutator yields the same document and persists once.
    store = RetryingConfigStore(manifest={"mcp": []})
    service, admin, _bus = _service(store)

    def add_server(document: dict[str, Any]) -> None:
        document["mcp"] = [*document.get("mcp", []), {"title": "srv", "config": {"url": "http://x"}}]

    result = await service.apply_change(add_server)

    assert store.persisted == [{"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}]
    assert result.document == {"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}
    assert admin.calls == 1


# ---------------------------------------------------------------------------
# apply_replace
# ---------------------------------------------------------------------------


async def test_apply_replace_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": [{"title": "old", "config": {"url": "http://old"}}]})
    service, admin, bus = _service(store)

    document = {"mcp": [{"title": "new", "config": {"url": "http://new"}}]}
    result = await service.apply_replace(document)

    assert store.manifest == document
    assert admin.calls == 1
    assert bus.publish_calls[0][0] == {"op": "reload_config"}
    assert bus.publish_calls[0][1] is None
    assert result.document == document


async def test_apply_replace_validates_before_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    service, admin, bus = _service(store)

    with pytest.raises(ValidationError):
        await service.apply_replace({"tools": "not-a-list"})

    # A replace has no mutator to abort, so validation must precede the persist.
    assert store.persisted == []
    assert store.manifest == {"mcp": []}
    assert admin.calls == 0
    assert bus.publish_calls == []


# ---------------------------------------------------------------------------
# Secret seal — a resolved !ENV secret never bakes to disk (decision 19)
# ---------------------------------------------------------------------------

# A manifest's mcp section is only ever read through the RESOLVED view, so the natural
# round-trip (read the resolved view → edit → post it back) hands the pipeline resolved
# secret values. The seal retags them back to the operator's !ENV marker before persist,
# and refuses a stranded resolved secret with no marker origin.
_TOKEN = "super-secret-token-value"


def _resolved(document: dict[str, Any]) -> dict[str, Any]:
    """The RESOLVED view a client reads through ``GET /api/manifest`` — ``!ENV``
    markers materialized against the current env, exactly as the live manifest
    exposes them."""
    return cast("dict[str, Any]", parse_config(data=dump_manifest(cast("Any", document))) or {})


def _marker_manifest() -> dict[str, Any]:
    """A manifest whose one mcp server carries an ``!ENV`` auth header marker."""
    return {
        "mcp": [
            {"title": "srv", "config": {"url": "http://x", "headers": {"Authorization": "!ENV ${TOKEN}"}}},
        ]
    }


async def test_apply_change_resolved_round_trip_restores_env_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact set_mcp_config round-trip: the client reads the RESOLVED mcp section,
    # edits it, and posts it back as document["mcp"] = <resolved list>. The seal must
    # restore the operator's !ENV marker so no resolved token bakes to disk.
    _with_bus(monkeypatch)
    monkeypatch.setenv("TOKEN", _TOKEN)
    store = FakeConfigStore(manifest=_marker_manifest())
    service, _admin, _bus = _service(store)
    resolved = _resolved(store.manifest)
    # The client read the resolved view: its Authorization is the plaintext token.
    assert resolved["mcp"][0]["config"]["headers"]["Authorization"] == _TOKEN

    def post_resolved(document: dict[str, Any]) -> None:
        # set_mcp_config's mutator: wholesale-replace mcp with the client-supplied
        # (resolved) list.
        document["mcp"] = copy.deepcopy(resolved["mcp"])

    result = await service.apply_change(post_resolved)

    # The persisted document restored the !ENV marker; the resolved token never landed.
    assert result.document is not None
    assert result.document["mcp"][0]["config"]["headers"]["Authorization"] == "!ENV ${TOKEN}"
    assert _TOKEN not in str(store.persisted)


async def test_apply_replace_resolved_round_trip_restores_env_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    # A whole-document replace carrying a resolved secret where the current doc has a
    # marker is retagged, so the marker persists.
    _with_bus(monkeypatch)
    monkeypatch.setenv("TOKEN", _TOKEN)
    store = FakeConfigStore(manifest=_marker_manifest())
    service, _admin, _bus = _service(store)

    replacement = copy.deepcopy(_resolved(store.manifest))
    result = await service.apply_replace(replacement)

    assert result.document is not None
    assert result.document["mcp"][0]["config"]["headers"]["Authorization"] == "!ENV ${TOKEN}"
    assert _TOKEN not in str(store.persisted)


async def test_apply_replace_stranded_resolved_secret_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # A replacement carrying a resolved secret whose entry has NO marker origin (its
    # identity was renamed away) is a stranded plaintext secret — refused loudly.
    _with_bus(monkeypatch)
    monkeypatch.setenv("TOKEN", _TOKEN)
    store = FakeConfigStore(manifest=_marker_manifest())
    service, admin, bus = _service(store)

    stranded = copy.deepcopy(_resolved(store.manifest))
    stranded["mcp"][0]["title"] = "renamed"  # no marker origin now; the token is plaintext

    with pytest.raises(ResolvedSecretError) as exc:
        await service.apply_replace(stranded)

    # ValueError-mappable to a 400 by the operations layer, and it names the offending path.
    assert isinstance(exc.value, ValueError)
    assert "mcp[0]" in str(exc.value)
    # Refused before any persist.
    assert store.persisted == []
    assert admin.calls == 0
    assert bus.publish_calls == []


async def test_apply_change_pure_marker_mutator_persists_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pure in-place mutator whose leaves already carry !ENV markers (a connector-style
    # append of a preserved-marker entry) is a retag no-op and passes the leak net with
    # no false rejection — the markers persist verbatim.
    _with_bus(monkeypatch)
    monkeypatch.setenv("TOKEN", _TOKEN)
    monkeypatch.setenv("OTHER", "other-secret-value")
    store = FakeConfigStore(manifest={"mcp": [{"title": "srv", "config": {"env": {"KEY": "!ENV ${TOKEN}"}}}]})
    service, _admin, _bus = _service(store)

    def append_marker_entry(document: dict[str, Any]) -> None:
        document["mcp"].append({"title": "srv2", "config": {"env": {"KEY2": "!ENV ${OTHER}"}}})

    result = await service.apply_change(append_marker_entry)

    assert result.document is not None
    # Both the untouched and the appended entries keep their !ENV markers verbatim.
    assert result.document["mcp"][0]["config"]["env"]["KEY"] == "!ENV ${TOKEN}"
    assert result.document["mcp"][1]["config"]["env"]["KEY2"] == "!ENV ${OTHER}"
    assert _TOKEN not in str(store.persisted)


# ---------------------------------------------------------------------------
# apply_env_change
# ---------------------------------------------------------------------------


async def test_apply_env_change_writes_reloads_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []}, env={"EXISTING": "1"})
    service, admin, bus = _service(store)

    result = await service.apply_env_change({"NEW_KEY": "v"})

    assert store.env_writes == [{"NEW_KEY": "v"}]
    assert store.env == {"EXISTING": "1", "NEW_KEY": "v"}
    assert admin.calls == 1
    assert bus.publish_calls[0] == ({"op": "reload_config"}, None, bus.publish_calls[0][2])
    # An env change touches no manifest document.
    assert result.document is None
    assert result.local == {"status": "ok", "env_keys": 0}


# ---------------------------------------------------------------------------
# Failure discipline
# ---------------------------------------------------------------------------


async def test_local_reload_failure_after_persist_still_broadcasts_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    admin = FakeReloadAdmin(raise_reload=RuntimeError("reload boom"))
    service, _admin, bus = _service(store, admin=admin)

    def add_server(document: dict[str, Any]) -> None:
        document["mcp"] = [{"title": "srv", "config": {"url": "http://x"}}]

    with pytest.raises(FleetBroadcastError) as exc:
        await service.apply_change(add_server)

    # The persist landed; the failed local reload does NOT strand the fleet.
    assert store.persisted == [{"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}]
    assert len(bus.publish_calls) == 1
    _op, _targets, local = bus.publish_calls[0]
    assert local is not None
    assert local.outcome == OpOutcome.failed
    # The fleet report the broadcast produced rides the raised error.
    assert exc.value.report.op == "reload_config"


async def test_broadcast_raise_after_persist_becomes_fleet_broadcast_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    # A non-transport broadcast fault the bus does NOT fold into a returned
    # bus-unreachable report (e.g. a redis ResponseError, or a malformed presence key
    # the census cannot parse) — it raises RAW out of publish AFTER the persist landed.
    bus = RecordingBus(publish_error=RuntimeError("ResponseError: WRONGTYPE"))
    service, admin, _bus = _service(store, bus=bus)

    def add_server(document: dict[str, Any]) -> None:
        document["mcp"] = [{"title": "srv", "config": {"url": "http://x"}}]

    with pytest.raises(FleetBroadcastError) as exc:
        await service.apply_change(add_server)

    # The raw broadcast error was wrapped as FleetBroadcastError, never propagated raw,
    # and rides as the cause.
    assert isinstance(exc.value.__cause__, RuntimeError)
    # The persist DID land — the committed mutation is in the store — and the local
    # reload ran before the broadcast raised.
    assert store.persisted == [{"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}]
    assert admin.calls == 1
    # The error carries the honest bus-unreachable report (no origin list, only error).
    assert exc.value.report.op == "reload_config"
    assert exc.value.report.reachable is False
    assert exc.value.report.results == []
    assert "ResponseError" in (exc.value.report.error or "")


async def test_apply_replace_broadcast_raise_after_persist_becomes_fleet_broadcast_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": [{"title": "old", "config": {"url": "http://old"}}]})
    bus = RecordingBus(publish_error=RuntimeError("ResponseError: WRONGTYPE"))
    service, admin, _bus = _service(store, bus=bus)

    document = {"mcp": [{"title": "new", "config": {"url": "http://new"}}]}
    with pytest.raises(FleetBroadcastError) as exc:
        await service.apply_replace(document)

    # apply_replace honors the same post-persist contract: the replace committed, then
    # the raw broadcast error surfaced as FleetBroadcastError with the unreachable report.
    assert store.manifest == document
    assert admin.calls == 1
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert exc.value.report.reachable is False


async def test_broadcast_raise_with_local_reload_failure_is_single_fleet_broadcast_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    admin = FakeReloadAdmin(raise_reload=RuntimeError("reload boom"))
    bus = RecordingBus(publish_error=RuntimeError("ResponseError: WRONGTYPE"))
    service, _admin, _bus = _service(store, admin=admin, bus=bus)

    def add_server(document: dict[str, Any]) -> None:
        document["mcp"] = [{"title": "srv", "config": {"url": "http://x"}}]

    with pytest.raises(FleetBroadcastError) as exc:
        await service.apply_change(add_server)

    # Both the local reload AND the broadcast failed after the persist landed — a SINGLE
    # FleetBroadcastError surfaces, carrying the broadcast error as cause and an
    # unreachable report whose error notes the local reload failure too.
    assert store.persisted == [{"mcp": [{"title": "srv", "config": {"url": "http://x"}}]}]
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert exc.value.report.reachable is False
    assert "ResponseError" in (exc.value.report.error or "")
    assert "local reload also failed" in (exc.value.report.error or "")


async def test_unconfirmed_origin_logs_error_but_returns_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    bus = RecordingBus(remotes=["serve-w1"], remote_outcome=OpOutcome.missing)
    service, _admin, _bus = _service(store, bus=bus)

    with caplog.at_level(logging.ERROR, logger="tai_skeleton.operations._broadcast"):
        result = await service.apply_replace({"mcp": []})

    # Persist + local reload landed, so the call SUCCEEDS; the unconfirmed origin is a
    # loud ERROR log and an explicit non-applied entry in the report.
    assert result.fleet.ok is False
    assert {r.origin: r.outcome for r in result.fleet.results}["serve-w1"] == OpOutcome.missing
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert any("did not fully converge" in record.message for record in caplog.records)


async def test_bus_unreachable_returns_success_with_unreachable_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={"mcp": []})
    bus = RecordingBus(reachable=False, error="ConnectionError: bus down")
    service, admin, _bus = _service(store, bus=bus)

    result = await service.apply_replace({"mcp": []})

    # Persist + local reload landed, so the call SUCCEEDS even though the transport was
    # down: the honest bus-unreachable shape (no origin list, only an error) rides through.
    assert admin.calls == 1
    assert result.fleet.reachable is False
    assert result.fleet.error == "ConnectionError: bus down"
    assert result.fleet.results == []


# ---------------------------------------------------------------------------
# backend-needs-bus invariant — both directions
# ---------------------------------------------------------------------------


async def test_manifest_change_adding_backend_without_bus_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_bus(monkeypatch)
    store = FakeConfigStore(manifest={})
    service, admin, bus = _service(store)

    def add_backend(document: dict[str, Any]) -> None:
        document["backend_module"] = "myapp.backend"

    with pytest.raises(BackendNeedsBusError, match="TAI_BUS_REDIS_URL"):
        await service.apply_change(add_backend)

    assert store.persisted == []
    assert admin.calls == 0
    assert bus.publish_calls == []


async def test_manifest_change_adding_backend_with_bus_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    store = FakeConfigStore(manifest={})
    service, admin, _bus = _service(store)

    def add_backend(document: dict[str, Any]) -> None:
        document["backend_module"] = "myapp.backend"

    result = await service.apply_change(add_backend)

    assert store.manifest == {"backend_module": "myapp.backend"}
    assert admin.calls == 1
    assert result.document == {"backend_module": "myapp.backend"}


async def test_env_change_materializing_backend_without_bus_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_bus(monkeypatch)
    # The manifest's backend module is an !ENV marker; the env change supplies the
    # value that materializes it — with no bus, the invariant rejects it.
    monkeypatch.delenv("TAI_BACKEND", raising=False)
    store = FakeConfigStore(manifest={"backend_module": "!ENV ${TAI_BACKEND}"})
    service, _admin, bus = _service(store)

    with pytest.raises(BackendNeedsBusError, match="TAI_BUS_REDIS_URL"):
        await service.apply_env_change({"TAI_BACKEND": "myapp.backend"})

    assert store.env_writes == []
    assert bus.publish_calls == []


async def test_env_change_removing_bus_while_backend_present_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_bus(monkeypatch)
    # A static backend is registered and the bus is configured only through the stored
    # env; the change empties the bus var — after it, a backend would run with no bus.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    store = FakeConfigStore(
        manifest={"backend_module": "myapp.backend"},
        env={"TAI_BUS_REDIS_URL": "redis://localhost:6379/0"},
    )
    service, _admin, bus = _service(store)

    with pytest.raises(BackendNeedsBusError, match="TAI_BUS_REDIS_URL"):
        await service.apply_env_change({"TAI_BUS_REDIS_URL": ""})

    assert store.env_writes == []
    assert bus.publish_calls == []


async def test_env_change_keeping_bus_with_backend_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_bus(monkeypatch)
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    store = FakeConfigStore(
        manifest={"backend_module": "myapp.backend"},
        env={"TAI_BUS_REDIS_URL": "redis://localhost:6379/0"},
    )
    service, admin, _bus = _service(store)

    result = await service.apply_env_change({"SOME_KEY": "v"})

    assert store.env_writes == [{"SOME_KEY": "v"}]
    assert admin.calls == 1
    assert result.document is None
