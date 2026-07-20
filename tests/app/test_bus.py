"""Worker bus unit/integration tests.

The bus is exercised over a real Redis pub/sub model — a shared ``fakeredis``
``FakeServer`` behind the bus's ``client_ctx`` seam. ``fakeredis`` genuinely models
pub/sub delivery, key TTL expiry (so missing-vs-departed is decided by a real
presence-key expiry, not a stub), and connection drops (``server.connected =
False`` raises the same ``ConnectionError`` the real client does), so the reconnect
and departed paths are driven, not mocked away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import fakeredis
import pytest
from fakeredis import aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from tai42_contract.errors import ClientDisconnectedError

import tai42_skeleton.app.bus as bus_module
from tai42_skeleton.app.bus import (
    FleetOrigin,
    FleetResult,
    LocalApplyResult,
    OpOutcome,
    OriginKind,
    OriginResult,
    UnknownFleetTargetsError,
    WorkerBus,
    _merge_terminal,
    make_origin,
)
from tai42_skeleton.app.bus_settings import BusRedisSettings, BusSettings


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def wire_bus_client(monkeypatch: pytest.MonkeyPatch, server: fakeredis.FakeServer) -> None:
    """Route the bus's ``client_ctx`` to a fresh fake handle on the shared server."""

    @asynccontextmanager
    async def fake_ctx(client_cls, settings=None, *, fresh=False, **kwargs) -> AsyncIterator[aioredis.FakeRedis]:
        client = aioredis.FakeRedis(server=server, decode_responses=True)
        try:
            yield client
        finally:
            await client.aclose()

    monkeypatch.setattr(bus_module, "client_ctx", fake_ctx)


@pytest.fixture
def wire_pooled_bus_client(monkeypatch: pytest.MonkeyPatch, server: fakeredis.FakeServer) -> None:
    """Route the bus's ``client_ctx`` through a handle that models the REAL pooled
    wrapper, not the raw driver: a severed connection inside the body surfaces as
    ``ClientDisconnectedError``, because ``tai42_kit``'s ``client_ctx`` evicts the dead
    client and re-raises the disconnection wrapped in that type. This is the exact shape
    a live bus-Redis outage takes; the raw-``ConnectionError`` fixture bypasses the
    wrapper and so never exercises the wrapped transport path.
    """

    @asynccontextmanager
    async def fake_ctx(client_cls, settings=None, *, fresh=False, **kwargs) -> AsyncIterator[aioredis.FakeRedis]:
        client = aioredis.FakeRedis(server=server, decode_responses=True)
        try:
            yield client
        except RedisConnectionError as exc:
            raise ClientDisconnectedError(
                f"{type(client).__name__} disconnected and was removed from the cache. "
                f"Retry the operation to create a new client. (Original error: {exc})"
            ) from exc
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    monkeypatch.setattr(bus_module, "client_ctx", fake_ctx)


def make_settings(namespace: str = "tai", **over: float) -> BusSettings:
    return BusSettings(
        redis=BusRedisSettings(redis_url="redis://fake"),
        namespace=namespace,
        ack_timeout=over.get("ack_timeout", 0.05),
        apply_timeout=over.get("apply_timeout", 0.3),
        heartbeat_ttl=over.get("heartbeat_ttl", 0.5),
    )


def make_bus(namespace: str = "tai", kind: OriginKind = OriginKind.serve, **over: float) -> WorkerBus:
    return WorkerBus(
        make_settings(namespace, **over),
        make_origin(kind),
        reconnect_backoff_initial=0.02,
        reconnect_backoff_max=0.05,
    )


async def _spawn_subscriber(
    bus: WorkerBus,
    callback: Callable[[dict], Awaitable[object]],
) -> tuple[asyncio.Task[None], FleetOrigin]:
    """Start a subscriber and wait until it is counted in the census (channel
    subscribed, self-resync fired, presence registered), so a subsequent publish is
    guaranteed to reach it."""
    ready = asyncio.Event()

    async def on_ready() -> None:
        ready.set()

    task = asyncio.create_task(bus.subscribe(bus.origin, callback, on_ready))
    await asyncio.wait_for(ready.wait(), timeout=2.0)

    # Presence is advertised after the on-ready resync, so wait until this origin is
    # actually counted before returning.
    async def _counted() -> bool:
        return any(o.origin == bus.origin.origin for o in await bus.census())

    async with asyncio.timeout(2.0):
        while not await _counted():
            await asyncio.sleep(0.01)
    return task, bus.origin


async def _stop(*tasks: asyncio.Task[None]) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task


# -- census -------------------------------------------------------------------


async def test_census_lists_registered_origins(wire_bus_client: None) -> None:
    serve_bus = make_bus(kind=OriginKind.serve)
    backend_bus = make_bus(kind=OriginKind.backend)

    async def noop(_op: dict) -> None:
        return None

    t1, serve_origin = await _spawn_subscriber(serve_bus, noop)
    t2, backend_origin = await _spawn_subscriber(backend_bus, noop)
    try:
        census = await serve_bus.census()
        by_id = {o.origin: o for o in census}
        assert set(by_id) == {serve_origin.origin, backend_origin.origin}
        assert by_id[serve_origin.origin].kind == OriginKind.serve
        assert by_id[backend_origin.origin].kind == OriginKind.backend
        # The presence-key value carries pid so consumers can tell kinds/procs apart.
        assert by_id[backend_origin.origin].pid == backend_origin.pid
    finally:
        await _stop(t1, t2)


# -- two-phase ack then apply -------------------------------------------------


async def test_two_phase_ack_then_slow_apply_reports_applied(wire_bus_client: None) -> None:
    publisher = make_bus()
    worker = make_bus()
    seen: list[dict] = []

    async def slow_apply(op: dict) -> dict:
        seen.append(op)
        # Longer than the ack timeout, shorter than the apply timeout: the fast
        # received-ack must keep this origin from being judged missing.
        await asyncio.sleep(0.12)
        return {"reloaded": True}

    task, worker_origin = await _spawn_subscriber(worker, slow_apply)
    try:
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        assert result.ok
        by_id = {r.origin: r for r in result.results}
        # The slow worker is applied (not missing), and the payload rides its entry.
        assert by_id[worker_origin.origin].outcome == OpOutcome.applied
        assert by_id[worker_origin.origin].payload == {"reloaded": True}
        # Self entry synthesized from local.
        assert by_id[publisher.origin.origin].outcome == OpOutcome.applied
        assert seen == [{"op": "reload_config"}]
    finally:
        await _stop(task)


async def test_all_terminal_early_exit_returns_before_apply_deadline(wire_bus_client: None) -> None:
    publisher = make_bus(apply_timeout=5.0)
    worker = make_bus(apply_timeout=5.0)

    async def fast(_op: dict) -> None:
        return None

    task, _ = await _spawn_subscriber(worker, fast)
    try:
        loop = asyncio.get_running_loop()
        start = loop.time()
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        elapsed = loop.time() - start
        assert result.ok
        # Every expected origin held a terminal reply, so the cut fired at once.
        assert elapsed < 2.0
    finally:
        await _stop(task)


# -- missing vs departed (real TTL expiry) ------------------------------------


async def _register_bare_presence(
    server: fakeredis.FakeServer,
    settings: BusSettings,
    origin: FleetOrigin,
    ttl_ms: int | None,
) -> None:
    """Write a presence key with no live subscriber behind it (models a worker that
    is counted by the census but will not reply)."""
    client = aioredis.FakeRedis(server=server, decode_responses=True)
    key = settings.presence_key(origin.origin)
    value = f'{{"kind": "{origin.kind.value}", "pid": {origin.pid}}}'
    if ttl_ms is None:
        await client.set(key, value)
    else:
        await client.set(key, value, px=ttl_ms)
    await client.aclose()


async def test_silent_but_present_origin_is_missing(wire_bus_client: None, server: fakeredis.FakeServer) -> None:
    publisher = make_bus()
    ghost = make_origin(OriginKind.backend)
    # Presence stays live throughout — no TTL — but nobody replies.
    await _register_bare_presence(server, publisher._settings, ghost, ttl_ms=None)

    result = await publisher.publish({"op": "reload_config"}, targets=[ghost.origin], local=None)
    by_id = {r.origin: r for r in result.results}
    assert by_id[ghost.origin].outcome == OpOutcome.missing
    assert by_id[ghost.origin].detail is not None


async def test_expired_presence_origin_is_departed(wire_bus_client: None, server: fakeredis.FakeServer) -> None:
    # Ack window short, apply window long enough for the presence key to lapse
    # between the publish-time census and the report cut.
    publisher = make_bus(ack_timeout=0.03, apply_timeout=0.25)
    ghost = make_origin(OriginKind.backend)
    # Alive when the census runs, expired by the cut — a genuine TTL expiry.
    await _register_bare_presence(server, publisher._settings, ghost, ttl_ms=80)

    result = await publisher.publish({"op": "reload_config"}, targets=[ghost.origin], local=None)
    by_id = {r.origin: r for r in result.results}
    assert by_id[ghost.origin].outcome == OpOutcome.departed
    assert by_id[ghost.origin].detail is not None


# -- fork safety: a forked child re-mints its bus origin ----------------------


async def test_remint_origin_changes_identity_same_kind_and_pid() -> None:
    # The re-mint the at_fork hook runs in a child: a fresh same-kind origin carrying
    # this process's pid, so a child's published op is a foreign origin to the parent.
    bus = make_bus(kind=OriginKind.backend)
    parent_origin = bus.origin
    bus._remint_origin_after_fork()
    child_origin = bus.origin
    assert child_origin.origin != parent_origin.origin
    assert child_origin.kind == parent_origin.kind == OriginKind.backend
    assert child_origin.pid == os.getpid()


async def test_forked_child_op_not_echo_skipped_by_parent(wire_bus_client: None) -> None:
    # A forked child inherits the parent's bus origin P. Before the re-mint, a frame
    # carrying P is echo-skipped by the parent's subscription (P applied it locally and
    # broadcast it); after the re-mint the child's op carries a fresh origin C, so the
    # parent applies it instead of skipping — the exact staleness the re-mint prevents.
    bus = make_bus()
    parent_origin = bus.origin
    applied: list[dict] = []

    async def record(op: dict) -> None:
        applied.append(op)

    self_frame = {"data": json.dumps({"op": "reload_config", "origin": parent_origin.origin})}
    await bus._handle_op(None, parent_origin, record, self_frame)
    assert applied == []  # a frame carrying the parent's own origin is echo-skipped

    bus._remint_origin_after_fork()
    child_origin = bus.origin
    assert child_origin.origin != parent_origin.origin
    child_frame = {"data": json.dumps({"op": "reload_config", "origin": child_origin.origin})}
    # The parent's subscription still runs as P; the child's re-minted origin is foreign.
    await bus._handle_op(None, parent_origin, record, child_frame)
    assert applied == [{"op": "reload_config"}]


def test_fork_child_remints_bus_origin_via_at_fork_hook() -> None:
    # A real os.fork(): the ``os.register_at_fork`` after-in-child hook the bus
    # registers at construction fires automatically in the child, re-minting its copy
    # of the origin. The child writes its origin back over a pipe; it must differ from
    # the parent's (which is left untouched) and keep the same kind.
    bus = make_bus(kind=OriginKind.backend)
    parent_origin = bus.origin.origin
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            os.write(write_fd, bus.origin.origin.encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        child_origin = os.read(read_fd, 4096).decode()
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
    assert child_origin
    assert child_origin != parent_origin
    assert child_origin.startswith(f"{OriginKind.backend.value}-")
    # The parent's own origin is unchanged — only the child's inherited copy re-minted.
    assert bus.origin.origin == parent_origin


# -- echo-skip + self-confirmation --------------------------------------------


async def test_echo_skip_synthesizes_self_and_never_reapplies(wire_bus_client: None) -> None:
    # The publisher is ALSO subscribed (its own presence key in the census). Without
    # echo-skip it would wait for its own reply and report itself timed_out.
    bus = make_bus()
    applied: list[dict] = []

    async def record(op: dict) -> None:
        applied.append(op)

    task, self_origin = await _spawn_subscriber(bus, record)
    try:
        local = LocalApplyResult(outcome=OpOutcome.applied, payload={"n": 1})
        result = await bus.publish({"op": "reload_config"}, targets=None, local=local)
        assert result.ok
        assert [r.origin for r in result.results] == [self_origin.origin]
        assert result.results[0].outcome == OpOutcome.applied
        assert result.results[0].payload == {"n": 1}
        # Give any (wrongly) delivered echo a chance to land — it must not.
        await asyncio.sleep(0.05)
        assert applied == []  # publisher never re-applies its own broadcast
    finally:
        await _stop(task)


async def test_self_failure_reported_when_local_failed(wire_bus_client: None) -> None:
    publisher = make_bus()
    local = LocalApplyResult(outcome=OpOutcome.failed, error="ValueError: boom")
    result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
    assert not result.ok
    self_entry = result.results[0]
    assert self_entry.outcome == OpOutcome.failed
    assert self_entry.error == "ValueError: boom"


# -- targets-vs-self bidirectional validation ---------------------------------


async def test_publish_raises_when_targeted_self_without_local(wire_bus_client: None) -> None:
    bus = make_bus()
    # targets=None => whole fleet includes self, but no local => cannot self-report.
    with pytest.raises(ValueError, match="can never reply"):
        await bus.publish({"op": "reload_config"}, targets=None, local=None)
    # explicit self-including targets, still no local.
    with pytest.raises(ValueError, match="can never reply"):
        await bus.publish({"op": "reload_config"}, targets=[bus.origin.origin], local=None)


async def test_publish_raises_when_local_given_but_self_excluded(wire_bus_client: None) -> None:
    bus = make_bus()
    other = make_origin(OriginKind.backend)
    local = LocalApplyResult(outcome=OpOutcome.applied)
    with pytest.raises(ValueError, match="exclude the publisher"):
        await bus.publish({"op": "reload_config"}, targets=[other.origin], local=local)


# -- namespace isolation ------------------------------------------------------


async def test_namespace_isolation_no_cross_talk(wire_bus_client: None, server: fakeredis.FakeServer) -> None:
    bus_a = make_bus(namespace="stack-a")
    bus_b = make_bus(namespace="stack-b")
    a_calls: list[dict] = []

    async def record(op: dict) -> None:
        a_calls.append(op)

    task_a, origin_a = await _spawn_subscriber(bus_a, record)
    try:
        # Presence keys are namespaced: A sees its origin, B sees an empty fleet.
        assert {o.origin for o in await bus_a.census()} == {origin_a.origin}
        assert await bus_b.census() == []

        # B publishes on its own channel; A's subscriber (different channel) must not fire.
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result_b = await bus_b.publish({"op": "reload_config"}, targets=None, local=local)
        assert [r.origin for r in result_b.results] == [bus_b.origin.origin]
        await asyncio.sleep(0.05)
        assert a_calls == []  # no cross-channel delivery
    finally:
        await _stop(task_a)


# -- reconnect / resubscribe after transport failure --------------------------


async def test_reconnect_after_transport_drop(wire_bus_client: None, server: fakeredis.FakeServer) -> None:
    publisher = make_bus()
    worker = make_bus(heartbeat_ttl=5.0)
    reconnects: list[str] = []

    async def apply(op: dict) -> None:
        return None

    async def on_ready() -> None:
        reconnects.append("ready")

    task = asyncio.create_task(worker.subscribe(worker.origin, apply, on_ready))
    try:
        await asyncio.wait_for(_until(lambda: len(reconnects) >= 1), timeout=2.0)
        # Drop the transport: the next poll raises ConnectionError inside the loop.
        server.connected = False
        await asyncio.sleep(0.15)
        # Restore and let the reconnect loop re-subscribe + re-register + re-fire ready.
        server.connected = True
        await asyncio.wait_for(_until(lambda: len(reconnects) >= 2), timeout=3.0)

        # Prove the resubscribed worker is live: it is back in the census and applies.
        census_ids = {o.origin for o in await publisher.census()}
        assert worker.origin.origin in census_ids
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker.origin.origin].outcome == OpOutcome.applied
    finally:
        await _stop(task)


async def test_reconnect_after_wrapped_disconnect(wire_pooled_bus_client: None, server: fakeredis.FakeServer) -> None:
    # The subscription's transport error arrives WRAPPED too: the pooled ``client_ctx``
    # folds the severed poll into ``ClientDisconnectedError``. The reconnect loop must
    # treat that wrapped type as transient exactly like the raw ``ConnectionError`` —
    # re-subscribe, re-register presence, re-fire ready — never let it kill the task.
    publisher = make_bus()
    worker = make_bus(heartbeat_ttl=5.0)
    reconnects: list[str] = []

    async def apply(op: dict) -> None:
        return None

    async def on_ready() -> None:
        reconnects.append("ready")

    task = asyncio.create_task(worker.subscribe(worker.origin, apply, on_ready))
    try:
        await asyncio.wait_for(_until(lambda: len(reconnects) >= 1), timeout=2.0)
        # Drop the transport: the next poll raises ConnectionError inside the loop, which
        # the pooled wrapper re-raises as ClientDisconnectedError out of the subscription.
        server.connected = False
        await asyncio.sleep(0.15)
        # Restore and let the reconnect loop re-subscribe + re-register + re-fire ready.
        server.connected = True
        await asyncio.wait_for(_until(lambda: len(reconnects) >= 2), timeout=3.0)

        # Prove the resubscribed worker is live: it re-registered presence (back in the
        # census) and applies again.
        census_ids = {o.origin for o in await publisher.census()}
        assert worker.origin.origin in census_ids
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker.origin.origin].outcome == OpOutcome.applied
    finally:
        await _stop(task)


async def _until(pred: Callable[[], bool]) -> None:
    while not pred():
        await asyncio.sleep(0.01)


# -- heartbeat supervision: an asymmetric refresh failure forces reconnect ----


async def test_heartbeat_death_forces_reconnect(
    wire_bus_client: None, server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Model an ASYMMETRIC transport failure: the presence-refresh heartbeat raises
    # (its pooled command connection drops) while the held pub/sub message loop stays
    # healthy. An unsupervised heartbeat would die in isolation and let this worker's
    # presence key expire — silently dropping a live worker from the census. Supervised,
    # its death tears the subscription down so the outer reconnect loop re-subscribes,
    # re-registers presence, and RESTARTS the heartbeat.
    publisher = make_bus()
    worker = make_bus(heartbeat_ttl=5.0)
    readies: list[str] = []
    heartbeats = {"n": 0}
    real_heartbeat = worker._heartbeat_loop

    async def flaky_heartbeat(r: object, origin: FleetOrigin, presence_key: str) -> None:
        heartbeats["n"] += 1
        if heartbeats["n"] == 1:
            # First subscription's heartbeat dies like a lost pooled connection.
            await asyncio.sleep(0.05)
            raise RedisConnectionError("presence refresh lost its pooled connection")
        # Every heartbeat after the forced reconnect behaves normally.
        await real_heartbeat(r, origin, presence_key)

    monkeypatch.setattr(worker, "_heartbeat_loop", flaky_heartbeat)

    async def apply(_op: dict) -> None:
        return None

    async def on_ready() -> None:
        readies.append("ready")

    task = asyncio.create_task(worker.subscribe(worker.origin, apply, on_ready))
    try:
        # Generous ceilings: the supervised reconnect completes in well under a
        # second, so these bounds only guard against a hang, not normal scheduling.
        await asyncio.wait_for(_until(lambda: len(readies) >= 1), timeout=10.0)
        # The heartbeat dies; supervision must force a full reconnect (a second
        # on_ready) and a fresh heartbeat task.
        await asyncio.wait_for(_until(lambda: len(readies) >= 2), timeout=10.0)
        # The reconnect restarted the heartbeat (a fresh task off the reconnected
        # subscription).
        await asyncio.wait_for(_until(lambda: heartbeats["n"] >= 2), timeout=10.0)

        # The reconnected worker is live: back in the census and applying ops.
        census_ids = {o.origin for o in await publisher.census()}
        assert worker.origin.origin in census_ids
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker.origin.origin].outcome == OpOutcome.applied
    finally:
        await _stop(task)


# -- per-origin payload channel -----------------------------------------------


async def test_query_op_payload_rides_per_origin_and_self(wire_bus_client: None) -> None:
    publisher = make_bus()
    worker = make_bus()

    async def list_failed(_op: dict) -> dict:
        return {"failed": ["mcp-a", "mcp-b"]}

    task, worker_origin = await _spawn_subscriber(worker, list_failed)
    try:
        # The serving worker's own query data rides its self entry via local.payload.
        local = LocalApplyResult(outcome=OpOutcome.applied, payload={"failed": ["mcp-local"]})
        result = await publisher.publish({"op": "list_failed_mcps"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker_origin.origin].payload == {"failed": ["mcp-a", "mcp-b"]}
        assert by_id[publisher.origin.origin].payload == {"failed": ["mcp-local"]}
    finally:
        await _stop(task)


# -- worst-outcome-wins terminal merge ----------------------------------------


def test_merge_terminal_failure_is_never_overridden() -> None:
    applied = OriginResult(origin="w1", outcome=OpOutcome.applied)
    failed = OriginResult(origin="w1", outcome=OpOutcome.failed, error="boom")

    # applied first, then failed => failed wins (worst outcome).
    terminal: dict[str, OriginResult] = {}
    _merge_terminal(terminal, "w1", applied)
    _merge_terminal(terminal, "w1", failed)
    assert terminal["w1"].outcome == OpOutcome.failed

    # failed first, then applied => failure is never masked.
    terminal = {}
    _merge_terminal(terminal, "w1", failed)
    _merge_terminal(terminal, "w1", applied)
    assert terminal["w1"].outcome == OpOutcome.failed


async def test_failing_callback_reports_failed(wire_bus_client: None) -> None:
    publisher = make_bus()
    worker = make_bus()

    async def boom(_op: dict) -> None:
        raise RuntimeError("nope")

    task, worker_origin = await _spawn_subscriber(worker, boom)
    try:
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker_origin.origin].outcome == OpOutcome.failed
        assert "nope" in (by_id[worker_origin.origin].error or "")
        assert not result.ok
    finally:
        await _stop(task)


# -- bus-unreachable shape ----------------------------------------------------


async def test_bus_unreachable_shape(wire_bus_client: None, server: fakeredis.FakeServer) -> None:
    publisher = make_bus()
    server.connected = False
    local = LocalApplyResult(outcome=OpOutcome.applied)
    result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
    assert result.reachable is False
    assert result.results == []  # no origin list on the unreachable shape
    assert result.error is not None


async def test_bus_unreachable_shape_on_wrapped_disconnect(
    wire_pooled_bus_client: None, server: fakeredis.FakeServer
) -> None:
    # A real outage arrives WRAPPED: the pooled ``client_ctx`` folds the severed
    # connection into ``ClientDisconnectedError``, not the raw ``ConnectionError`` the
    # sibling test injects. ``publish`` must fold that wrapped type the same way — the
    # bus-unreachable report — never let it escape and re-raise (the bare-500 root cause).
    publisher = make_bus()
    server.connected = False
    local = LocalApplyResult(outcome=OpOutcome.applied)
    result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
    assert result.reachable is False
    assert result.results == []  # no origin list on the unreachable shape
    assert result.error is not None
    assert "ClientDisconnectedError" in result.error  # the wrapped type, not the raw driver error


# -- validate_targets ---------------------------------------------------------


async def test_validate_targets_raises_on_unknown(wire_bus_client: None) -> None:
    bus = make_bus()
    worker = make_bus()

    async def noop(_op: dict) -> None:
        return None

    task, worker_origin = await _spawn_subscriber(worker, noop)
    try:
        # A known origin passes; self is always valid.
        await bus.validate_targets([worker_origin.origin])
        await bus.validate_targets([bus.origin.origin])
        with pytest.raises(UnknownFleetTargetsError, match="unknown fleet targets"):
            await bus.validate_targets(["backend-does-not-exist"])
    finally:
        await _stop(task)


# -- no-op local() variant ----------------------------------------------------


async def test_local_variant() -> None:
    bus = WorkerBus.local()

    # publish: local-only result with the synthesized self entry.
    local = LocalApplyResult(outcome=OpOutcome.applied, payload={"ok": True})
    result = await bus.publish({"op": "reload_config"}, targets=None, local=local)
    assert isinstance(result, FleetResult)
    assert result.local_only
    assert result.ok
    assert [r.origin for r in result.results] == [bus.origin.origin]
    assert result.results[0].payload == {"ok": True}

    # census: just this process.
    assert await bus.census() == [bus.origin]

    # validate_targets: self ok, any other target raises.
    await bus.validate_targets([bus.origin.origin])
    with pytest.raises(ValueError, match="cannot reach"):
        await bus.publish({"op": "x"}, targets=["backend-other"], local=None)

    # bidirectional validation still applies on the local variant.
    with pytest.raises(ValueError, match="can never reply"):
        await bus.publish({"op": "x"}, targets=None, local=None)

    # subscribe parks until cancelled.
    async def noop(_op: dict) -> None:
        return None

    task = asyncio.create_task(bus.subscribe(bus.origin, noop))
    await asyncio.sleep(0.05)
    assert not task.done()
    await _stop(task)


# -- timed_out: live worker, apply outlasts the report cut --------------------


async def test_live_but_slow_apply_is_timed_out(wire_bus_client: None) -> None:
    # heartbeat_ttl < apply_timeout, and the callback sleeps PAST the apply timeout.
    # The presence key would expire mid-apply if liveness rode the (now blocked)
    # message loop — the census would drop this live worker and the report cut would
    # misclassify it as departed. With the heartbeat on its OWN task, presence stays
    # live throughout the long apply, so the acked-but-unterminated origin is correctly
    # timed_out. This passes only with the decoupled heartbeat; the single-loop design
    # would report departed here.
    publisher = make_bus(ack_timeout=0.05, apply_timeout=0.5, heartbeat_ttl=0.2)
    worker = make_bus(ack_timeout=0.05, apply_timeout=0.5, heartbeat_ttl=0.2)

    async def slow(_op: dict) -> None:
        # Longer than the apply timeout: no terminal reply by the report cut. The fast
        # received-ack still lands, and the sleep yields so the heartbeat task refreshes
        # presence throughout.
        await asyncio.sleep(0.7)

    task, worker_origin = await _spawn_subscriber(worker, slow)
    try:
        local = LocalApplyResult(outcome=OpOutcome.applied)
        result = await publisher.publish({"op": "reload_config"}, targets=None, local=local)
        by_id = {r.origin: r for r in result.results}
        assert by_id[worker_origin.origin].outcome == OpOutcome.timed_out
        assert by_id[worker_origin.origin].detail is not None
    finally:
        await _stop(task)


# -- subscriber-side targets filter -------------------------------------------


async def test_subscriber_skips_op_outside_targets(wire_bus_client: None) -> None:
    # Two live subscribers on the same channel; the op targets only worker_a. worker_b
    # receives the broadcast but must SKIP it (it is not in targets), so its callback
    # side effect never fires — proving the subscriber-side targets filter.
    publisher = make_bus()
    worker_a = make_bus()
    worker_b = make_bus()
    a_seen: list[dict] = []
    b_seen: list[dict] = []

    async def record_a(op: dict) -> None:
        a_seen.append(op)

    async def record_b(op: dict) -> None:
        b_seen.append(op)

    task_a, origin_a = await _spawn_subscriber(worker_a, record_a)
    task_b, origin_b = await _spawn_subscriber(worker_b, record_b)
    try:
        # Target only worker_a; the publisher excludes itself, so local=None.
        result = await publisher.publish({"op": "reload_config"}, targets=[origin_a.origin], local=None)
        by_id = {r.origin: r for r in result.results}
        # worker_a is the sole expected origin and applied; worker_b is not expected.
        assert set(by_id) == {origin_a.origin}
        assert origin_b.origin not in by_id
        assert by_id[origin_a.origin].outcome == OpOutcome.applied
        assert a_seen == [{"op": "reload_config"}]
        # worker_b saw the broadcast on the shared channel but skipped it — no side effect.
        await asyncio.sleep(0.05)
        assert b_seen == []
    finally:
        await _stop(task_a, task_b)


async def test_empty_targets_list_reaches_nobody(wire_bus_client: None) -> None:
    # An empty targets list reaches NOBODY: it self-excludes the publisher (local=None)
    # and every live subscriber must skip it, matching the publisher's empty expected
    # set. Under the old truthiness filter (``if targets and ...``) an empty list is
    # falsy, so both subscribers would NOT skip and would apply the op — a silent
    # sibling mutation invisible in the report. This test would fail there.
    publisher = make_bus()
    worker_a = make_bus()
    worker_b = make_bus()
    a_seen: list[dict] = []
    b_seen: list[dict] = []

    async def record_a(op: dict) -> None:
        a_seen.append(op)

    async def record_b(op: dict) -> None:
        b_seen.append(op)

    task_a, _origin_a = await _spawn_subscriber(worker_a, record_a)
    task_b, _origin_b = await _spawn_subscriber(worker_b, record_b)
    try:
        # Empty targets self-excludes the publisher, so local=None is the valid pairing.
        result = await publisher.publish({"op": "reload_config"}, targets=[], local=None)
        # No expected origins — the report honestly shows nobody, no hidden sibling apply.
        assert result.results == []
        # Both live subscribers saw the broadcast on the shared channel but skipped it.
        await asyncio.sleep(0.05)
        assert a_seen == []
        assert b_seen == []
    finally:
        await _stop(task_a, task_b)
