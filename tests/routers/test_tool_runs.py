"""Background tool-runs router: submit (202), single-run GET, per-tool list, the
supervisor's terminal writes, and the one-way ``lost`` reconciliation.

Handlers are driven directly (the router-test pattern). Redis is a focused
in-memory fake yielded at the router's ``client_ctx`` seam (submit, GET, list,
and the spawned supervisor all share the one store); the ``tai42_app.tools`` facet
is a stand-in whose ``run_tool`` records its call (incl. ``offload_sync``) and
returns/raises/gates on demand.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai42_skeleton.app import server as server_module
from tai42_skeleton.operations import tool_runs as ops
from tai42_skeleton.operations.tool_runs import ToolRunStore
from tai42_skeleton.routers import tool_runs as router
from tai42_skeleton.routers import tools as tools_router
from tai42_skeleton.routers.tool_runs_settings import ToolRunsSettings
from tai42_skeleton.tools import binding as binding_module
from tai42_skeleton.tools.binding import ToolBinding
from tests._fakes.tool_runs_redis import FakeRedis

# -- request builders --------------------------------------------------------


def _post(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/tool-runs", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _get(run_id: str) -> Request:
    return cast(Request, SimpleNamespace(path_params={"run_id": run_id}))


def _list(tool_name: str | None = None) -> Request:
    params = {} if tool_name is None else {"tool_name": tool_name}
    return cast(Request, SimpleNamespace(query_params=params))


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


@contextmanager
def _identity(*, user_id: str | None = None, owner: str | None = None) -> Iterator[None]:
    """Bind a caller identity for the duration of a door call: ``owner`` set makes it
    a RESTRICTED owned key (isolated to its OWN id ``user_id``, NOT its owner —
    each key is its own island), ``owner=None`` an unrestricted caller. Binds the same
    request-scoped seam the access-control guard binds. Tests pass an ``owner``
    DIFFERENT from ``user_id`` so the key-own-vs-owner distinction is exercised."""
    claims: dict[str, str] = {} if owner is None else {OWNER_USER_ID_CLAIM: owner}
    uid_token = set_request_user_id(user_id) if user_id is not None else None
    claims_token = set_request_identity_claims(claims)
    try:
        yield
    finally:
        reset_request_identity_claims(claims_token)
        if uid_token is not None:
            reset_request_user_id(uid_token)


# -- tools facet stand-in ----------------------------------------------------


class _FakeTools:
    def __init__(self, registered: set[str] | None = None) -> None:
        self.result: object = None
        self.exc: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.calls: list[tuple] = []
        # The submit door resolves the name against the registry up front; the
        # default set covers the tool names the happy-path tests submit.
        self._registered = registered if registered is not None else {"alpha", "slow"}

    async def get_tools(self):
        return {name: SimpleNamespace(name=name) for name in self._registered}

    async def run_tool(self, key, arguments, *, offload_sync=False):
        self.calls.append((key, arguments, offload_sync))
        if self.gate is not None:
            await self.gate.wait()
        if self.exc is not None:
            raise self.exc
        return self.result


# -- wiring ------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    fake = FakeRedis()
    settings = ToolRunsSettings()

    @asynccontextmanager
    async def ctx(client_cls, s=None, *, fresh=False, **kwargs):
        yield fake

    clock = {"t": datetime(2026, 1, 1, tzinfo=UTC)}

    monkeypatch.setattr(ops, "client_ctx", ctx)
    monkeypatch.setattr(ops, "tool_runs_settings", lambda: settings)
    monkeypatch.setattr(ops, "_now", lambda: clock["t"])
    # Isolate the per-worker concurrency counter per test (auto-restored).
    monkeypatch.setattr(ops, "_ACTIVE_RUNS", 0)

    def install_tools(registered: set[str] | None = None) -> _FakeTools:
        tools = _FakeTools(registered)
        monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=tools))
        return tools

    yield SimpleNamespace(
        fake=fake,
        settings=settings,
        clock=clock,
        monkeypatch=monkeypatch,
        store=ToolRunStore(settings.key_prefix),
        install_tools=install_tools,
    )

    # No supervisor may outlive the test — cancel any that gated open.
    for task in list(ops._SUPERVISORS):
        task.cancel()


async def _drain(timeout: float = 2.0) -> None:
    """Run every spawned supervisor to completion (they self-remove on done)."""
    tasks = list(ops._SUPERVISORS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)


# -- submit → poll → succeeded ----------------------------------------------


async def test_submit_returns_202_and_run_id(wired):
    wired.install_tools()
    resp = await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {"x": 2}}'))
    assert resp.status_code == 202
    assert isinstance(_json(resp)["data"]["run_id"], str)
    await _drain()


async def test_submit_poll_succeeded_round_trip(wired):
    tools = wired.install_tools()
    tools.result = {"ok": 1}
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {"x": 2}}')))["data"]["run_id"]

    # Before the run finishes it reads as running (created record + liveness).
    assert _json(await router.get_run(_get(run_id)))["data"]["status"] == "running"

    await _drain()
    data = _json(await router.get_run(_get(run_id)))["data"]
    assert data["status"] == "succeeded"
    assert data["result"] == {"ok": 1}
    assert data["tool_name"] == "alpha"
    assert data["started_at"]
    assert data["finished_at"]
    # The background path runs through the same seam with the sync offload gate on.
    assert tools.calls == [("alpha", {"x": 2}, True)]


async def test_submit_poll_running_then_succeeded_with_gate(wired):
    tools = wired.install_tools()
    tools.result = "done"
    tools.gate = asyncio.Event()
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}')))["data"]["run_id"]
    await asyncio.sleep(0)  # let the supervisor reach the gate

    assert _json(await router.get_run(_get(run_id)))["data"]["status"] == "running"
    tools.gate.set()
    await _drain()
    assert _json(await router.get_run(_get(run_id)))["data"]["status"] == "succeeded"


# -- failure surfaces as a record -------------------------------------------


async def test_failed_tool_records_failed_and_error(wired):
    tools = wired.install_tools()
    tools.exc = RuntimeError("boom")
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}')))["data"]["run_id"]
    await _drain()
    data = _json(await router.get_run(_get(run_id)))["data"]
    assert data["status"] == "failed"
    assert data["error"] == "boom"
    assert "result" not in data
    assert data["finished_at"]


async def test_unknown_tool_is_rejected_404_before_a_record_is_created(wired):
    # An unknown name is resolved against the registry up front and rejected with
    # a loud 404 — no ``running`` record is created and no supervisor is spawned,
    # so a typo can never pollute the store or masquerade as a runtime failure.
    wired.install_tools(registered={"alpha"})
    resp = await router.submit_run(_post(b'{"tool_name": "nope", "arguments": {}}'))
    assert resp.status_code == 404
    assert "unknown tool" in _json(resp)["error"]
    assert list(ops._SUPERVISORS) == []


# -- supervisor task failure surfaces at completion time ---------------------


async def test_supervisor_setup_failure_is_logged_loudly(wired, caplog):
    # A failure BEFORE the inner try (here the ``client_ctx`` enter raising because
    # Redis died after submit) escapes the record-writing guard; the done-callback
    # must log it with run_id/tool_name at completion time instead of leaving it to
    # asyncio's nondeterministic GC-time "never retrieved" message.
    wired.install_tools(registered={"alpha"})

    @asynccontextmanager
    async def boom_ctx(client_cls, s=None, *, fresh=False, **kwargs):
        raise RuntimeError("redis gone")
        yield  # pragma: no cover — unreachable

    resp = await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}'))
    assert resp.status_code == 202  # the record was created before the supervisor ran
    wired.monkeypatch.setattr(ops, "client_ctx", boom_ctx)
    with caplog.at_level("ERROR"):
        await _drain()
    assert "supervisor task failed" in caplog.text
    assert "redis gone" in caplog.text


# -- concurrency cap ---------------------------------------------------------


async def test_concurrency_cap_returns_503_then_frees_slot(wired):
    # Cap forced to 1: the first (blocked) run holds the only slot, the second is
    # refused with a 503 naming the env var, and once the first finishes and frees
    # its slot a third submit is accepted again.
    tools = wired.install_tools(registered={"slow"})
    tools.result = "ok"
    tools.gate = asyncio.Event()
    wired.monkeypatch.setattr(ops, "tool_runs_settings", lambda: ToolRunsSettings(max_concurrent_runs=1))

    first = await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}'))
    assert first.status_code == 202
    await asyncio.sleep(0)  # let the supervisor reach the gate and hold the slot

    second = await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}'))
    assert second.status_code == 503
    assert "TAI_TOOL_RUNS_MAX_CONCURRENT_RUNS" in _json(second)["error"]

    # Release the first run and drain it; its done-callback returns the slot.
    tools.gate.set()
    await _drain()
    assert ops._ACTIVE_RUNS == 0

    third = await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}'))
    assert third.status_code == 202
    await _drain()


async def test_create_run_failure_returns_the_concurrency_slot(wired):
    # If ``create_run`` raises after the slot is reserved, the slot must be
    # returned (and the error re-raised loudly) so a later submit is not falsely
    # refused.
    wired.install_tools(registered={"alpha"})

    @asynccontextmanager
    async def boom_ctx(client_cls, s=None, *, fresh=False, **kwargs):
        raise RuntimeError("redis down")
        yield  # pragma: no cover — unreachable

    wired.monkeypatch.setattr(ops, "client_ctx", boom_ctx)
    with pytest.raises(RuntimeError, match="redis down"):
        await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}'))
    assert ops._ACTIVE_RUNS == 0

    # A follow-up submit against a working store is accepted, proving the slot came back.
    @asynccontextmanager
    async def ok_ctx(client_cls, s=None, *, fresh=False, **kwargs):
        yield wired.fake

    wired.monkeypatch.setattr(ops, "client_ctx", ok_ctx)
    resp = await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}'))
    assert resp.status_code == 202
    await _drain()


# -- shutdown drain ----------------------------------------------------------


async def test_drain_supervisors_records_failed_shutdown_and_drains(wired):
    # A run in flight at shutdown is cancelled by the drain handler; its
    # cancellation branch CAS-writes a ``failed`` record carrying the shutdown
    # string, and the supervisor task set drains.
    tools = wired.install_tools(registered={"slow"})
    tools.gate = asyncio.Event()  # never set — the run is in flight at drain
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}')))["data"]["run_id"]
    await asyncio.sleep(0)  # let the supervisor reach the gate
    assert [t for t in ops._SUPERVISORS if not t.done()]

    await ops._drain_supervisors()
    await asyncio.sleep(0)  # let the done-callbacks settle

    record = await wired.store.get_run(wired.fake, run_id)
    assert record["status"] == "failed"
    assert record["error"] == "server shutdown before the tool-run completed"
    assert record["finished_at"]
    assert set() == ops._SUPERVISORS


async def test_drain_supervisors_does_not_overwrite_a_pre_lost_record(wired, caplog):
    # A run already reconciled to ``lost`` (dead-process reader) must not be flipped
    # by the cancelled supervisor's terminal write — the one-way CAS gates it out and
    # the supervisor logs the skip.
    tools = wired.install_tools(registered={"slow"})
    tools.gate = asyncio.Event()
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {}}')))["data"]["run_id"]
    await asyncio.sleep(0)

    # Drop liveness while still ``running`` and read it: the reader persists ``lost``.
    await wired.fake.delete(wired.store.liveness_key(run_id))
    assert _json(await router.get_run(_get(run_id)))["data"]["status"] == "lost"

    with caplog.at_level("WARNING"):
        await ops._drain_supervisors()
        await asyncio.sleep(0)

    record = await wired.store.get_run(wired.fake, run_id)
    assert record["status"] == "lost"  # not overwritten
    assert "one-way lost" in caplog.text
    assert set() == ops._SUPERVISORS


# -- request-validation parity with the sync route --------------------------


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        (b"not json", 400, "invalid JSON"),
        (b"[]", 400, "JSON object"),
        (b'{"arguments": {}}', 400, "tool_name"),
    ],
)
async def test_submit_request_errors_match_sync_shape(wired, body, status, fragment):
    wired.install_tools()
    resp = await router.submit_run(_post(body))
    assert resp.status_code == status
    payload = _json(resp)
    assert set(payload) == {"error"}
    assert fragment in payload["error"]


async def test_sync_route_body_validation_matches(wired):
    # Both doors share the one parser + field shape, so the sync door rejects a bad
    # body identically to the background door.
    wired.install_tools()
    assert _json(await tools_router.run_tool(_post(b"not json")))["error"] == "invalid JSON body"
    missing = _json(await tools_router.run_tool(_post(b'{"arguments": {}}')))
    assert missing["error"] == "body must contain a non-empty 'tool_name'"


# -- 404 ---------------------------------------------------------------------


async def test_get_unknown_run_404(wired):
    resp = await router.get_run(_get("does-not-exist"))
    assert resp.status_code == 404
    assert "not found" in _json(resp)["error"]


# -- lost reconciliation -----------------------------------------------------


async def _seed_running(wired, run_id: str, tool_name: str = "alpha") -> None:
    started = wired.clock["t"].isoformat()
    await wired.store.create_run(wired.fake, run_id, tool_name, started, wired.clock["t"].timestamp(), wired.settings)


async def test_running_with_liveness_is_not_lost(wired):
    await _seed_running(wired, "r1")
    assert _json(await router.get_run(_get("r1")))["data"]["status"] == "running"


async def test_process_death_persists_lost_and_stays(wired):
    await _seed_running(wired, "r1")
    # Simulate process death: the liveness key expired and no supervisor
    # ``finally`` ran to write a terminal record.
    await wired.fake.delete(wired.store.liveness_key("r1"))

    data = _json(await router.get_run(_get("r1")))["data"]
    assert data["status"] == "lost"
    assert data["finished_at"]

    # It is persisted (stored status is now lost) and one-way: even a resurrected
    # liveness key cannot flip it back — reconciliation only acts on running.
    assert (await wired.store.get_run(wired.fake, "r1"))["status"] == "lost"
    await wired.store.refresh_liveness(wired.fake, "r1", wired.settings.liveness_ttl_seconds)
    assert _json(await router.get_run(_get("r1")))["data"]["status"] == "lost"


async def test_terminal_write_cannot_overwrite_a_lost(wired):
    # A run whose liveness dropped while still ``running`` is reconciled to ``lost``
    # by a reader; the supervisor's later terminal write (the tool actually
    # finished) must be gated out by the compare-and-set — ``lost`` is one-way, so
    # it never flips to ``succeeded``.
    await _seed_running(wired, "r1")
    await wired.fake.delete(wired.store.liveness_key("r1"))
    assert _json(await router.get_run(_get("r1")))["data"]["status"] == "lost"

    persisted = await wired.store.mark_terminal_if_running(
        wired.fake,
        "r1",
        {"status": "succeeded", "finished_at": wired.clock["t"].isoformat(), "result": json.dumps("done")},
        wired.settings.result_ttl_seconds,
    )
    assert persisted is False
    data = _json(await router.get_run(_get("r1")))["data"]
    assert data["status"] == "lost"
    assert "result" not in data


async def test_terminal_cas_writes_only_from_running(wired):
    # The CAS transitions a still-``running`` record (returns True) and refuses any
    # record that already reached a terminal state (returns False, no clobber).
    await _seed_running(wired, "r1")
    ttl = wired.settings.result_ttl_seconds
    ok = await wired.store.mark_terminal_if_running(
        wired.fake, "r1", {"status": "succeeded", "finished_at": "t", "result": json.dumps(1)}, ttl
    )
    assert ok is True
    again = await wired.store.mark_terminal_if_running(
        wired.fake, "r1", {"status": "failed", "finished_at": "t", "error": "no"}, ttl
    )
    assert again is False
    assert (await wired.store.get_run(wired.fake, "r1"))["status"] == "succeeded"


async def test_reconcile_reflects_real_terminal_when_cas_loses_the_race(wired):
    # A reader read a stale ``running`` snapshot with liveness absent, but the
    # supervisor's ``succeeded`` write landed before the reader's CAS: the CAS is
    # rejected and the reader reflects the REAL terminal record, never a spurious
    # ``lost``.
    await _seed_running(wired, "r1")
    await wired.store.mark_terminal_if_running(
        wired.fake,
        "r1",
        {"status": "succeeded", "finished_at": wired.clock["t"].isoformat(), "result": json.dumps(1)},
        wired.settings.result_ttl_seconds,
    )
    stale = {"tool_name": "alpha", "status": "running", "started_at": wired.clock["t"].isoformat()}
    reconciled = await ops._reconcile_lost_with_liveness(
        wired.fake, wired.store, "r1", stale, False, wired.settings.result_ttl_seconds
    )
    assert reconciled["status"] == "succeeded"


async def test_liveness_refresher_survives_a_transient_refresh_error():
    # A single failing ``SET`` must not permanently kill the refresher: it logs and
    # keeps refreshing on the next cadence. A refresher that died on the error would
    # never set the liveness key, so a live run could be wrongly marked ``lost``.
    class _FlakyOnceRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self._blips = 1

        async def set(self, key: str, value: str, ex: int | None = None) -> bool:
            if self._blips > 0:
                self._blips -= 1
                raise RuntimeError("redis blip")
            return await super().set(key, value, ex=ex)

    store = ToolRunStore("trtest:")
    redis = _FlakyOnceRedis()
    settings = ToolRunsSettings(liveness_ttl_seconds=1)  # cadence ~0.33s
    task = asyncio.create_task(ops._refresh_liveness_loop(redis, store, "r1", settings))
    try:
        # The first refresh raises; a surviving loop reaches a later successful one
        # that finally sets the liveness key.
        for _ in range(400):
            if await redis.get(store.liveness_key("r1")) is not None:
                break
            await asyncio.sleep(0.01)
        assert await redis.get(store.liveness_key("r1")) is not None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# -- list route --------------------------------------------------------------


async def test_list_missing_tool_name_400(wired):
    resp = await router.list_tool_runs(_list(None))
    assert resp.status_code == 400
    assert "tool_name" in _json(resp)["error"]


async def test_list_order_and_trim_to_limit(wired):
    wired.monkeypatch.setattr(ops, "tool_runs_settings", lambda: ToolRunsSettings(recent_runs_limit=3))
    settings = ops.tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)
    base = wired.clock["t"]
    for i in range(5):
        ts = (base + timedelta(seconds=i)).isoformat()
        await store.create_run(wired.fake, f"r{i}", "alpha", ts, float(i), settings)

    entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    # Newest first, trimmed to the limit.
    assert [e["run_id"] for e in entries] == ["r4", "r3", "r2"]
    for entry in entries:
        assert entry["tool_name"] == "alpha"
        assert entry["status"] == "running"
        assert entry["started_at"]


async def test_list_entries_carry_no_result_or_error(wired):
    await _seed_running(wired, "r1")
    await wired.store.mark_terminal_if_running(
        wired.fake,
        "r1",
        {"status": "succeeded", "finished_at": wired.clock["t"].isoformat(), "result": json.dumps({"secret": 1})},
        wired.settings.result_ttl_seconds,
    )
    await router.get_run(_get("r1"))  # no effect — already terminal
    entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert len(entries) == 1
    assert "result" not in entries[0]
    assert "error" not in entries[0]
    assert entries[0]["status"] == "succeeded"


async def test_list_reconciles_lost_for_a_dead_running_entry(wired):
    await _seed_running(wired, "r1")
    await wired.fake.delete(wired.store.liveness_key("r1"))
    entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert entries[0]["status"] == "lost"


async def test_list_prunes_a_phantom_index_member(wired):
    # A ZSET member whose record hash expired is pruned and skipped, never listed.
    await wired.fake.zadd(wired.store.recent_key("alpha"), {"ghost": 1.0})
    entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert entries == []
    assert await wired.fake.zrevrange(wired.store.recent_key("alpha"), 0, -1) == []


# -- TTL ---------------------------------------------------------------------


async def test_result_ttl_applied_on_create_and_terminal(wired):
    tools = wired.install_tools()
    tools.result = {"ok": True}
    run_id = _json(await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}')))["data"]["run_id"]
    run_key = wired.store.run_key(run_id)
    assert wired.fake.ttl_of(run_key) == wired.settings.result_ttl_seconds
    await _drain()
    # Terminal write refreshes the record TTL so the result outlives the run.
    assert wired.fake.ttl_of(run_key) == wired.settings.result_ttl_seconds


# -- end-to-end: a slow SYNC tool outlasting the liveness TTL is NOT lost -----


def _function_tool(fn):
    tool = MagicMock(spec=binding_module.FunctionTool)
    tool.fn = fn
    return tool


def _real_binding_for(fn) -> ToolBinding:
    """A real ``ToolBinding`` whose ``get_tool`` yields ``fn`` — the same
    validate/call/offload path the sync door and the supervisor drive, with no
    mocked adapter."""
    binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))

    async def _get_tool(_key: str):
        return _function_tool(fn)

    async def _get_tools():
        # The submit door resolves the name against the registry before creating a
        # record; register the submitted name so the real run path is reached.
        return {"slow": _function_tool(fn)}

    binding.get_tool = _get_tool  # type: ignore[method-assign]
    binding.get_tools = _get_tools  # type: ignore[method-assign]
    return binding


async def _seed(
    wired, run_id: str, *, tool_name: str = "alpha", user_id: str | None = None, score: float = 1.0
) -> None:
    await wired.store.create_run(
        wired.fake, run_id, tool_name, wired.clock["t"].isoformat(), score, wired.settings, user_id=user_id
    )


async def test_submit_stamps_owning_identity_and_writes_per_identity_index(wired):
    # A restricted owned key (own id "keyA", owner "alice") submits: the run is stamped
    # with the caller's OWN id "keyA" and pushed onto BOTH the shared index and keyA's
    # per-identity index — the owner claim ("alice") is NOT the isolation identity.
    wired.install_tools()
    with _identity(user_id="keyA", owner="alice"):
        run_id = _json(await router.submit_run(_post(b'{"tool_name": "alpha", "arguments": {}}')))["data"]["run_id"]
    await _drain()
    record = await wired.store.get_run(wired.fake, run_id)
    assert record["user_id"] == "keyA"
    assert run_id in await wired.fake.zrevrange(wired.store.recent_key("alpha", "keyA"), 0, -1)
    assert run_id in await wired.fake.zrevrange(wired.store.recent_key("alpha"), 0, -1)
    # The owner claim is not a slice key — no index was written under it.
    assert await wired.fake.zrevrange(wired.store.recent_key("alpha", "alice"), 0, -1) == []


async def test_restricted_get_own_run_200_others_403(wired):
    # keyA owns "ra"; "rb" belongs to a foreign key; "sib" belongs to keyB — a SIBLING
    # owned key of the SAME owner "alice". keyA reads only its own run; both the foreign
    # run AND the same-owner sibling's run are 403 (each key is its own island).
    await _seed(wired, "ra", user_id="keyA")
    await _seed(wired, "rb", user_id="bob")
    await _seed(wired, "sib", user_id="keyB")
    with _identity(user_id="keyA", owner="alice"):
        own = await router.get_run(_get("ra"))
        assert _json(own)["data"]["run_id"] == "ra"
        other = await router.get_run(_get("rb"))
        sibling = await router.get_run(_get("sib"))
    assert other.status_code == 403
    assert _json(other)["error"] == "run belongs to another identity"
    # A same-owner sibling key's run is NOT shared — key-keyed isolation, not owner-keyed.
    assert sibling.status_code == 403


async def test_restricted_list_reads_its_own_index_excluding_other_identities(wired):
    # keyB is a SIBLING owned key of the SAME owner "alice"; its run must still be
    # excluded from keyA's list (key-keyed, not owner-keyed isolation).
    await _seed(wired, "ra", user_id="keyA", score=1.0)
    await _seed(wired, "rb", user_id="bob", score=2.0)
    await _seed(wired, "sib", user_id="keyB", score=3.0)
    with _identity(user_id="keyA", owner="alice"):
        entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert [e["run_id"] for e in entries] == ["ra"]


async def test_restricted_list_stays_complete_when_shared_window_saturated(wired):
    # The truncation regression: keyA's run is the OLDEST, so a shared bounded window
    # would evict it once other identities flood past the cap. Its per-identity index
    # keeps it, so its list stays complete — proving a per-identity index, not a
    # post-filter over the shared window.
    wired.monkeypatch.setattr(ops, "tool_runs_settings", lambda: ToolRunsSettings(recent_runs_limit=3))
    settings = ops.tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)

    async def seed(run_id: str, own_id: str, score: float) -> None:
        await store.create_run(
            wired.fake, run_id, "alpha", wired.clock["t"].isoformat(), score, settings, user_id=own_id
        )

    await seed("keyA-run", "keyA", 0.0)
    for i in range(1, 6):
        await seed(f"bob-{i}", "bob", float(i))

    # The shared window trimmed to the newest 3 evicted keyA's older run…
    assert "keyA-run" not in await wired.fake.zrevrange(store.recent_key("alpha"), 0, -1)
    # …but its own per-identity index still lists it complete.
    with _identity(user_id="keyA", owner="alice"):
        entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert [e["run_id"] for e in entries] == ["keyA-run"]


async def test_unrestricted_caller_keeps_full_shared_view(wired):
    await _seed(wired, "ra", user_id="alice", score=1.0)
    await _seed(wired, "rb", user_id="bob", score=2.0)
    # An authenticated but unrestricted caller (no owner claim) reads the shared
    # window and may GET any run — today's view, regression-pinned.
    with _identity(user_id="op1", owner=None):
        entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
        assert {e["run_id"] for e in entries} == {"ra", "rb"}
        assert _json(await router.get_run(_get("rb")))["data"]["run_id"] == "rb"


async def test_record_without_user_id_readable_by_unrestricted_403_to_restricted(wired):
    # A record with no ``user_id`` field is owned by no identity. An unrestricted caller
    # reads it; a restricted caller sees absent ≠ theirs and is denied.
    await _seed(wired, "old", user_id=None)
    assert "user_id" not in await wired.store.get_run(wired.fake, "old")
    assert _json(await router.get_run(_get("old")))["data"]["run_id"] == "old"
    with _identity(user_id="keyA", owner="alice"):
        resp = await router.get_run(_get("old"))
    assert resp.status_code == 403


async def test_restricted_list_prunes_the_per_identity_index_not_the_shared_one(wired):
    # A phantom (record expired, index member lingers) in a restricted caller's
    # per-identity index is pruned from THAT index — never the shared one.
    await wired.fake.zadd(wired.store.recent_key("alpha", "keyA"), {"ghost": 1.0})
    await wired.fake.zadd(wired.store.recent_key("alpha"), {"ghost": 1.0})
    with _identity(user_id="keyA", owner="alice"):
        entries = _json(await router.list_tool_runs(_list("alpha")))["data"]
    assert entries == []
    assert await wired.fake.zrevrange(wired.store.recent_key("alpha", "keyA"), 0, -1) == []
    # The shared index was NOT touched by the restricted list's prune.
    assert await wired.fake.zrevrange(wired.store.recent_key("alpha"), 0, -1) == ["ghost"]


async def test_slow_sync_tool_over_liveness_ttl_is_not_marked_lost(monkeypatch):
    """A genuinely blocking SYNC tool running LONGER than ``liveness_ttl_seconds``
    must never be reconciled to ``lost`` — the exact regression the thread-offload
    prevents. Driven through the real supervisor (``offload_sync=True``): the sync
    body blocks a worker thread, leaving the loop free for the liveness refresher.

    Timing-dependent by construction. The tool blocks a real thread for ~2.0s while
    the liveness TTL is 1s of REAL wall-clock (``FakeRedis`` driven by
    ``time.monotonic``, matching the refresher's real ``asyncio.sleep`` cadence).
    The 2:1 margin is deliberately generous. Were the offload to regress (the sync
    ran inline), the loop would block, the refresher would stall, liveness would
    lapse at ~1s, the mid-run poll would persist ``lost``, and the one-way CAS would
    keep it ``lost`` — failing this test."""
    fake = FakeRedis(clock=time.monotonic)
    settings = ToolRunsSettings(liveness_ttl_seconds=1)

    @asynccontextmanager
    async def ctx(client_cls, s=None, *, fresh=False, **kwargs):
        yield fake

    monkeypatch.setattr(ops, "client_ctx", ctx)
    monkeypatch.setattr(ops, "tool_runs_settings", lambda: settings)

    def slow_sync(x: int) -> dict:
        # Block the worker thread well past the 1s liveness TTL, then echo so the
        # body is proven to have run.
        time.sleep(2.0)
        return {"echo": x}

    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_real_binding_for(slow_sync)))

    try:
        run_id = _json(await router.submit_run(_post(b'{"tool_name": "slow", "arguments": {"x": 9}}')))["data"][
            "run_id"
        ]

        # Poll PAST the liveness TTL while the tool still blocks on the thread: the
        # refresher (loop free) kept liveness alive, so the run reads running — not
        # lost.
        await asyncio.sleep(1.4)
        assert _json(await router.get_run(_get(run_id)))["data"]["status"] == "running"

        await _drain(timeout=6.0)
        data = _json(await router.get_run(_get(run_id)))["data"]
        assert data["status"] == "succeeded"
        assert data["result"] == {"echo": 9}
    finally:
        # No supervisor may outlive the test.
        for task in list(ops._SUPERVISORS):
            task.cancel()
