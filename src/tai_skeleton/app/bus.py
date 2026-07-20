"""The worker bus — the app's one INTERNAL fleet fan-out primitive.

Every process (HTTP server or backend runtime) that shares a manifest joins one
Redis pub/sub control channel through :meth:`WorkerBus.subscribe` and keeps a TTL
presence key alive; a mutation reaches the whole fleet through the single awaited
:meth:`WorkerBus.publish`, which collects a per-origin outcome from every live
worker.

This is app-owned internal infrastructure, like the reload gate — NOT a plugin.
Nothing here is registrable, swappable, or user-selectable; there is exactly one
bus, and no manifest field chooses an implementation. A deployment with no Redis
configured runs on :meth:`WorkerBus.local`, the no-op variant.

Namespacing
-----------
``TAI_BUS_NAMESPACE`` (default ``tai``) prefixes the control channel, every reply
channel, and the presence keys. Redis pub/sub is server-global (NOT scoped by db
index), so two deployments/stacks sharing one Redis MUST diverge by namespace or
they would cross-talk; a shared-Redis deployment sets a unique namespace per stack.

Transport shape
---------------
One control channel carries every op; each op names an ephemeral reply channel
for its confirmations. The census is a scan of the per-origin presence keys — the
registered origins ARE the fleet listing. Two wire messages come back from a
subscriber per op: a ``received`` ack the instant the op is delivered (the fast
liveness signal) and exactly one terminal ``applied``/``failed`` once the op has
fully applied (the slow correctness signal). ``timed_out`` and ``departed`` are
never wire replies — the publisher computes them from the presence census at the
report cut.

Fork safety
-----------
A forked child (an rq work-horse per job, a celery prefork pool child per task)
inherits the parent's :class:`WorkerBus` object across ``os.fork()`` — its origin
included. The bus registers an ``os.register_at_fork`` after-in-child hook at
construction that re-mints the child's copy of the origin to a fresh same-kind
value, so a fleet op the child publishes is applied by the parent instead of being
echo-skipped as the parent's own broadcast. The child remains a non-member: its
inherited subscription task is dormant and it registers no presence, so the re-mint
neither resumes a subscription nor advertises a second census identity.

Missing vs departed
-------------------
``census()`` and the expected-origin set cover REGISTERED origins only; a
mid-reconnect worker is intentionally absent and self-heals (its startup re-reads
persisted state) before rejoining. An expected origin that never sends its
``received`` ack is re-checked against presence: the key still live ⇒ ``missing``
(alive but silent, only reachable inside the short ack window); the key expired ⇒
``departed`` (the worker went away). An origin that acked but never sent a terminal
by the apply deadline is re-checked the same way: expired ⇒ ``departed`` (ack then
die — SIGKILL / rolling restart mid-reload), alive ⇒ ``timed_out`` (the op WAS
delivered and usually still applies when its reload gate frees; verify with the
fleet census or a fleet reload).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from tai_contract.errors import ClientDisconnectedError
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.app.bus_settings import BusSettings

logger = logging.getLogger(__name__)

# Wire fields that address/route an op rather than describe it; stripped before the
# op payload is handed to the subscriber callback.
_TRANSPORT_KEYS = frozenset({"origin", "reply_to", "targets"})

# How long one idle pub/sub poll blocks in the subscribe loop; bounds the latency
# of noticing a cancellation and of the next presence refresh.
_POLL_TIMEOUT = 0.1

# Transport errors that a reconnect loop recovers from and a publish reports as the
# bus-unreachable shape. ``ClientDisconnectedError`` is the pooled ``client_ctx``
# wrapper for a severed connection, so it belongs here alongside the raw redis
# transport errors: a real bus outage arrives wrapped, and must fold into the
# bus-unreachable report (publish) and drive a reconnect + re-register (subscription),
# never escape as a bare error.
_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    RedisConnectionError,
    RedisTimeoutError,
    ClientDisconnectedError,
)


class OpOutcome(StrEnum):
    """Per-origin outcome of a fleet op.

    ``applied`` / ``failed`` are terminal WIRE replies from a subscriber;
    ``missing`` / ``departed`` / ``timed_out`` are publisher-COMPUTED from the
    presence census (a silent origin never sends them)."""

    applied = "applied"
    failed = "failed"
    missing = "missing"
    departed = "departed"
    timed_out = "timed_out"


class OriginKind(StrEnum):
    """The two worker kinds that join the bus."""

    serve = "serve"
    backend = "backend"


class FleetOrigin(BaseModel):
    """One live worker on the bus. ``origin`` is ``{kind}-{uuid}``; the presence-key
    value carries ``kind`` and ``pid`` so a consumer can tell worker kinds apart."""

    origin: str
    kind: OriginKind
    pid: int


class LocalApplyResult(BaseModel):
    """The publisher's own already-completed self-apply outcome, handed to
    :meth:`WorkerBus.publish` so the bus can synthesize a truthful self entry.

    ``outcome`` is terminal — ``applied`` on success, ``failed`` (with ``error``
    attached) on the publish-anyway path where the broadcast still goes out after a
    failed local apply. ``payload`` rides the same optional shape as a wire reply so
    the serving worker's own query data appears in a query op's fleet result."""

    outcome: OpOutcome
    payload: Any | None = None
    error: str | None = None

    @field_validator("outcome")
    @classmethod
    def _terminal_only(cls, value: OpOutcome) -> OpOutcome:
        if value not in (OpOutcome.applied, OpOutcome.failed):
            raise ValueError(
                "LocalApplyResult.outcome must be 'applied' or 'failed' — the caller's own terminal result"
            )
        return value


class OriginResult(BaseModel):
    """One origin's outcome within a :class:`FleetResult`.

    ``payload`` carries query-op data (a read rides the same fan-out shape as a
    mutation); ``error`` carries a failed apply's message; ``detail`` carries the
    publisher's report text for a computed ``missing``/``departed``/``timed_out``."""

    origin: str
    outcome: OpOutcome
    payload: Any | None = None
    error: str | None = None
    detail: str | None = None


class FleetResult(BaseModel):
    """The awaited result of one :meth:`WorkerBus.publish`.

    Two honest shapes. Reachable (``reachable=True``): ``results`` holds one
    :class:`OriginResult` per expected origin (the synthesized self entry included).
    Bus-unreachable (``reachable=False``): the transport failed before any origin
    could reply, so there is NO origin list — only ``error``. ``local_only`` marks
    the result of the no-op :meth:`WorkerBus.local` variant."""

    op: str
    reachable: bool = True
    local_only: bool = False
    results: list[OriginResult] = Field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the bus was reachable and every origin applied."""
        return self.reachable and all(r.outcome == OpOutcome.applied for r in self.results)


def make_origin(kind: OriginKind, pid: int | None = None) -> FleetOrigin:
    """Mint a fresh origin identity for this process: ``{kind}-{uuid}`` + pid."""
    ident = f"{kind.value}-{uuid.uuid4().hex}"
    return FleetOrigin(origin=ident, kind=kind, pid=pid if pid is not None else os.getpid())


def _decode(raw: Any) -> dict[str, Any] | None:
    """Decode a wire frame to a dict, or ``None`` (logged) when it is not an object."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("worker bus: discarding non-JSON wire frame", exc_info=True)
        return None
    if not isinstance(data, dict):
        logger.warning("worker bus: discarding non-object wire frame: %r", data)
        return None
    return data


def _merge_terminal(terminal: dict[str, OriginResult], origin: str, result: OriginResult) -> None:
    """Fold one terminal reply into the per-origin map, worst-outcome-wins.

    A terminal reply supersedes nothing but another terminal; among terminals a
    failure is never overridden by a later same-origin success, so a genuinely
    failed apply cannot be masked."""
    existing = terminal.get(origin)
    if existing is not None and existing.outcome == OpOutcome.failed:
        return
    terminal[origin] = result


class UnknownFleetTargetsError(ValueError):
    """A fleet op named target workers absent from the presence census.

    A caller-side bad-request error (a typo'd or departed worker name), raised
    BEFORE any side effect; publishers surface it as a 400.
    """


class WorkerBus:
    """The one app-owned internal fan-out primitive over Redis pub/sub.

    Bound to this process's :class:`FleetOrigin` at construction: :meth:`publish`
    uses it to echo-skip the publisher's own broadcast and synthesize the self
    entry from the caller's ``local`` result. Constructed once per process by the
    lifecycle; the same origin is passed to :meth:`subscribe`."""

    def __init__(
        self,
        settings: BusSettings,
        origin: FleetOrigin,
        *,
        local: bool = False,
        reconnect_backoff_initial: float = 0.5,
        reconnect_backoff_max: float = 30.0,
        reconnect_backoff_factor: float = 2.0,
    ) -> None:
        self._settings = settings
        self._origin = origin
        self._local = local
        self._backoff_initial = reconnect_backoff_initial
        self._backoff_max = reconnect_backoff_max
        self._backoff_factor = reconnect_backoff_factor
        if not local:
            # Fork safety: a forked child inherits this bus object — origin included —
            # so it re-mints its own origin in the child (see _remint_origin_after_fork).
            # One registration per real bus instance (one bus per process); the bound
            # hook is inherited across further forks, so a re-minted child that forks
            # again re-mints again. The no-op ``local`` variant never publishes to a
            # real bus, so it needs no re-mint.
            os.register_at_fork(after_in_child=self._remint_origin_after_fork)

    def _remint_origin_after_fork(self) -> None:
        """Re-mint this bus's origin in a forked child (an ``os.register_at_fork``
        after-in-child hook registered at construction of the real bus).

        A child inherits the parent's ``WorkerBus`` — origin included — across
        ``os.fork()`` (an rq work-horse per job, a celery prefork pool child per
        task). Left shared, a fleet op the child publishes would carry the PARENT's
        origin and be echo-skipped by the parent's own subscription (:meth:`_handle_op`),
        leaving the publishing worker stale on the change it just made. A fresh
        same-kind origin (carrying the child's pid) makes the child's published op a
        foreign origin the parent applies instead of skipping.

        In-memory only — no Redis I/O runs in a fork hook. The child stays a
        non-member: its inherited subscription task is dormant and it registers no
        presence, so re-minting neither resumes a subscription nor advertises a second
        census identity."""
        self._origin = make_origin(self._origin.kind)

    @classmethod
    def local(cls, origin: FleetOrigin | None = None) -> WorkerBus:
        """The no-op variant for a single-worker / file-mode / no-backend / no-bus
        process: :meth:`publish` returns a local-only result, :meth:`subscribe`
        parks, :meth:`census` returns just this process. Legal only under the boot
        rules that permit a busless deployment."""
        return cls(settings=BusSettings(), origin=origin or make_origin(OriginKind.serve), local=True)

    @property
    def origin(self) -> FleetOrigin:
        """This process's bus identity. Re-minted in a forked child (see
        :meth:`_remint_origin_after_fork`) so an op the child publishes is applied by
        the parent, not echo-skipped."""
        return self._origin

    # -- Publisher side --------------------------------------------------------

    async def publish(
        self,
        op: dict[str, Any],
        targets: list[str] | None,
        local: LocalApplyResult | None,
    ) -> FleetResult:
        """Broadcast one op to the fleet and collect a per-origin outcome; awaited.

        ``targets=None`` is the whole fleet. ``local`` is the caller's own
        already-completed self-apply outcome — the bus cannot truthfully report the
        publisher's own origin otherwise. The two are validated against ``targets``
        BOTH ways: ``local=None`` with the publisher targeted (``targets=None`` or a
        self-including list) raises (the publisher would be an expected origin that
        can never reply), and ``local`` supplied with self-EXCLUDING targets raises
        (a self entry for an unexpected origin would be a false report).

        Absent targets are NOT re-raised here — a targeted publisher validates first
        via :meth:`validate_targets`, so a target that vanished between validation
        and here is churn, reported honestly as ``departed`` rather than raised."""
        op_name = self._op_name(op)
        self._validate_local_targets(local, targets)

        if self._local:
            return self._local_publish(op_name, targets, local)

        try:
            fleet = await self._broadcast(op_name, op, targets)
        except _TRANSPORT_ERRORS as exc:
            logger.error("worker bus: publish of %r failed — bus unreachable", op_name, exc_info=True)
            return FleetResult(op=op_name, reachable=False, error=f"{type(exc).__name__}: {exc}")

        if local is not None:
            fleet.append(self._self_result(local))
        fleet.sort(key=lambda r: r.origin)
        return FleetResult(op=op_name, results=fleet)

    async def _broadcast(self, op_name: str, op: dict[str, Any], targets: list[str] | None) -> list[OriginResult]:
        reply_channel = f"{self._settings.reply_prefix}{uuid.uuid4().hex}"
        async with client_ctx(RedisClient, self._settings.redis) as conn:
            r: Any = conn
            expected = await self._expected_origins(r, targets)
            pubsub = r.pubsub()
            await pubsub.subscribe(reply_channel)
            try:
                wire = {**op, "origin": self._origin.origin, "reply_to": reply_channel, "targets": targets}
                await r.publish(self._settings.channel, json.dumps(wire))
                collected = await self._collect(r, pubsub, expected, op_name)
            finally:
                await pubsub.unsubscribe(reply_channel)
                await pubsub.aclose()
        return list(collected.values())

    async def _expected_origins(self, r: Any, targets: list[str] | None) -> set[str]:
        live = {o.origin for o in await self._scan_origins(r)}
        if targets is None:
            return live - {self._origin.origin}
        # Targets are NOT re-validated here (the caller validated); an absent target
        # is reported as departed at the cut, never raised.
        return set(targets) - {self._origin.origin}

    async def _collect(self, r: Any, pubsub: Any, expected: set[str], op_name: str) -> dict[str, OriginResult]:
        terminal: dict[str, OriginResult] = {}
        acked: set[str] = set()
        loop = asyncio.get_running_loop()
        start = loop.time()
        ack_deadline = start + self._settings.ack_timeout
        apply_deadline = start + self._settings.apply_timeout
        ack_checked = False
        transport_error: str | None = None

        # Early exit counts TERMINAL wire replies only; a provisional missing/departed
        # never enables it, so an in-flight ``applied`` can never be cut off early.
        while expected - terminal.keys():
            now = loop.time()
            if now >= apply_deadline:
                break
            if not ack_checked and now >= ack_deadline:
                # Provisional verdicts at the ack deadline are diagnostic only; the
                # finalize pass below re-checks presence, so they never block a
                # later terminal reply. Marking the pass done is what matters here.
                ack_checked = True
            next_deadline = apply_deadline if ack_checked else ack_deadline
            timeout = max(0.01, next_deadline - now)
            try:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
            except _TRANSPORT_ERRORS as exc:
                # A blip mid-collection is reported loudly on the affected origins
                # (below), never silently dropped.
                transport_error = f"{type(exc).__name__}: {exc}"
                logger.error("worker bus: reply collection for %r hit a transport error", op_name, exc_info=True)
                break
            if msg is None:
                continue
            frame = _decode(msg["data"])
            if frame is None:
                continue
            origin = frame.get("origin")
            if origin not in expected:
                continue
            phase = frame.get("phase")
            if phase == "received":
                acked.add(origin)
            elif phase == "terminal":
                _merge_terminal(terminal, origin, self._terminal_result(origin, frame))

        results: dict[str, OriginResult] = {}
        for origin in expected:
            verdict = terminal.get(origin)
            if verdict is None:
                verdict = await self._computed_verdict(r, origin, origin in acked, transport_error)
            results[origin] = verdict
        return results

    def _terminal_result(self, origin: str, frame: dict[str, Any]) -> OriginResult:
        raw = frame.get("outcome")
        outcome = OpOutcome.failed if raw == OpOutcome.failed.value else OpOutcome.applied
        return OriginResult(origin=origin, outcome=outcome, payload=frame.get("payload"), error=frame.get("error"))

    async def _computed_verdict(self, r: Any, origin: str, acked: bool, transport_error: str | None) -> OriginResult:
        """Classify an origin that holds no terminal reply, re-checking presence."""
        alive = await self._presence_alive(r, origin)
        if acked:
            if alive is False:
                return OriginResult(
                    origin=origin,
                    outcome=OpOutcome.departed,
                    detail="acked then its presence key expired before applying — worker departed mid-apply",
                )
            detail = "acked but did not apply within the apply timeout — op delivered, verify via fleet census/reload"
            if transport_error is not None:
                detail = f"{detail} (reply collection transport error: {transport_error})"
            return OriginResult(origin=origin, outcome=OpOutcome.timed_out, detail=detail)
        if alive is False:
            return OriginResult(
                origin=origin,
                outcome=OpOutcome.departed,
                detail="no reply and presence key expired — worker departed",
            )
        detail = "no received-ack within the ack timeout while presence stayed live — worker missing (alive but silent)"
        if transport_error is not None:
            detail = f"{detail} (reply collection transport error: {transport_error})"
        return OriginResult(origin=origin, outcome=OpOutcome.missing, detail=detail)

    async def _presence_alive(self, r: Any, origin: str) -> bool | None:
        """True/False for a live/expired presence key; ``None`` when the check itself
        could not reach Redis (the caller degrades to a loud missing/timed_out)."""
        try:
            return bool(await r.exists(self._settings.presence_key(origin)))
        except _TRANSPORT_ERRORS:
            logger.error("worker bus: presence re-check for %s failed", origin, exc_info=True)
            return None

    def _self_result(self, local: LocalApplyResult) -> OriginResult:
        return OriginResult(
            origin=self._origin.origin,
            outcome=local.outcome,
            payload=local.payload,
            error=local.error,
        )

    def _local_publish(self, op_name: str, targets: list[str] | None, local: LocalApplyResult | None) -> FleetResult:
        if targets is not None:
            unknown = sorted(set(targets) - {self._origin.origin})
            if unknown:
                raise UnknownFleetTargetsError(
                    f"worker bus (local): cannot reach targets not on this process: {unknown}"
                )
        results = [self._self_result(local)] if local is not None else []
        return FleetResult(op=op_name, local_only=True, results=results)

    def _validate_local_targets(self, local: LocalApplyResult | None, targets: list[str] | None) -> None:
        self_targeted = targets is None or self._origin.origin in targets
        if local is None and self_targeted:
            raise ValueError(
                "publish: local=None but the publisher is a targeted origin (targets=None or self-including); "
                "the publisher would be an expected origin that can never reply"
            )
        if local is not None and not self_targeted:
            raise ValueError(
                "publish: a local result was supplied but targets exclude the publisher; "
                "a self entry for an unexpected origin would be a false report"
            )

    @staticmethod
    def _op_name(op: dict[str, Any]) -> str:
        name = op.get("op")
        if not isinstance(name, str) or not name:
            raise ValueError("publish: op dict must carry a non-empty 'op' name")
        return name

    # -- Census + target validation --------------------------------------------

    async def census(self) -> list[FleetOrigin]:
        """The live fleet: one :class:`FleetOrigin` per registered presence key.

        This is the fleet worker listing (it backs ``GET /api/fleet/workers``)."""
        if self._local:
            return [self._origin]
        async with client_ctx(RedisClient, self._settings.redis) as conn:
            return await self._scan_origins(conn)

    async def validate_targets(self, targets: list[str] | None) -> None:
        """Raise naming any target absent from the census — a caller-side seam run
        BEFORE the caller's local apply, so validation precedes side effects.

        ``targets=None`` (whole fleet) is always valid. A typo'd worker name is an
        error here, never a silent narrowing at publish time."""
        if targets is None:
            return
        live = {self._origin.origin}
        if not self._local:
            live |= {o.origin for o in await self.census()}
        unknown = sorted(set(targets) - live)
        if unknown:
            raise UnknownFleetTargetsError(f"worker bus: unknown fleet targets (not on the census): {unknown}")

    async def _scan_origins(self, r: Any) -> list[FleetOrigin]:
        origins: list[FleetOrigin] = []
        prefix = self._settings.presence_prefix
        async for key in r.scan_iter(match=self._settings.presence_pattern):
            key_str = key.decode() if isinstance(key, bytes) else key
            ident = key_str[len(prefix) :]
            raw = await r.get(key_str)
            if raw is None:
                # Expired between the scan and the read — no longer live, skipped.
                continue
            origins.append(self._origin_from_presence(ident, raw))
        return origins

    @staticmethod
    def _origin_from_presence(ident: str, raw: Any) -> FleetOrigin:
        text = raw.decode() if isinstance(raw, bytes) else raw
        meta = json.loads(text)
        return FleetOrigin(origin=ident, kind=OriginKind(meta["kind"]), pid=int(meta["pid"]))

    # -- Subscriber side -------------------------------------------------------

    async def subscribe(
        self,
        origin: FleetOrigin,
        callback: Callable[[dict[str, Any]], Awaitable[Any]],
        on_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Consume the control channel until cancelled, reconnecting with backoff.

        Registers a TTL presence key for ``origin`` (refreshed at ttl/3). Each op is
        applied by awaiting ``callback`` on this task, with a two-phase confirmation:
        a ``received`` ack the instant the op is delivered, then one terminal
        ``applied``/``failed`` when the callback returns (a raising callback ⇒
        ``failed``). The publisher's own broadcast is echo-skipped — its self entry
        is synthesized from the ``local`` result it hands :meth:`publish`.

        Boot-join and every reconnect follow the same order: subscribe the channel
        FIRST, fire ``on_ready`` (the self-resync), THEN register presence. A worker
        joins the census only once its resync has finished, so a reader that sees it
        counted finds it already converged and past its reload gate — never mid-resync.
        The channel is subscribed before the resync, so an op broadcast during it is
        buffered and applied by the message loop; on a reconnect within the heartbeat
        TTL the prior presence key is still live, so the worker is never dropped from the
        census mid-resync — only a first boot, which has no prior key, appears once, after
        its resync completes. Each reconnect attempt is ERROR-logged."""
        if self._local:
            await asyncio.Event().wait()
            return

        backoff = self._backoff_initial
        while True:
            established = False

            def _mark_established() -> None:
                nonlocal established
                established = True

            try:
                await self._run_subscription(origin, callback, on_ready, _mark_established)
                return
            except asyncio.CancelledError:
                raise
            except _TRANSPORT_ERRORS:
                logger.error(
                    "worker bus: subscription transport error for %s; reconnecting in %.2fs",
                    origin.origin,
                    backoff,
                    exc_info=True,
                )
            if established:
                backoff = self._backoff_initial
            await asyncio.sleep(backoff)
            backoff = min(backoff * self._backoff_factor, self._backoff_max)

    async def _run_subscription(
        self,
        origin: FleetOrigin,
        callback: Callable[[dict[str, Any]], Awaitable[Any]],
        on_ready: Callable[[], Awaitable[None]] | None,
        on_established: Callable[[], None],
    ) -> None:
        presence_key = self._settings.presence_key(origin.origin)
        async with client_ctx(RedisClient, self._settings.redis) as conn:
            r: Any = conn
            pubsub = r.pubsub()
            await pubsub.subscribe(self._settings.channel)
            heartbeat: asyncio.Task[None] | None = None
            try:
                on_established()
                # The self-resync runs BEFORE presence is advertised, so this worker is
                # counted only once it has converged and left its reload gate — never
                # mid-resync (a reader that sees it counted can act at once). The channel
                # is already subscribed, so an op broadcast during the resync is buffered
                # and applied by the message loop below.
                if on_ready is not None:
                    await on_ready()
                await self._register_presence(r, origin, presence_key)
                # Liveness runs on its OWN task, decoupled from op apply: refreshing
                # presence here — never inside the message loop — keeps this worker's
                # presence key alive across an apply that outlasts the TTL (the census
                # must not drop a live worker mid-reload).
                heartbeat = asyncio.create_task(
                    self._heartbeat_loop(r, origin, presence_key),
                    name=f"tai-worker-bus-heartbeat-{origin.origin}",
                )
                logger.info("worker bus: subscription live as %s", origin.origin)
                while True:
                    if heartbeat.done():
                        # The heartbeat ended on its own — a presence refresh raised on
                        # a pooled command connection while this held pub/sub connection
                        # stayed healthy (an asymmetric transport failure). Left alone it
                        # would leave this worker live but about to fall out of the census
                        # once its presence key expires — the exact census-drop the
                        # heartbeat exists to prevent. Surface it and force the whole
                        # subscription down so the outer reconnect loop re-subscribes and
                        # re-registers presence (which restarts the heartbeat).
                        failure = self._heartbeat_failure(heartbeat, origin)
                        # Its exception is already retrieved and logged; drop the
                        # reference so teardown does not re-await and re-log it.
                        heartbeat = None
                        raise failure
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT)
                    if msg is not None:
                        await self._handle_op(r, origin, callback, msg)
            finally:
                await self._stop_heartbeat(heartbeat)
                await self._teardown(r, pubsub, presence_key)

    async def _heartbeat_loop(self, r: Any, origin: FleetOrigin, presence_key: str) -> None:
        """Refresh this worker's presence key at ``ttl/3`` on a task of its own.

        Liveness is decoupled from op apply on purpose: were the refresh folded into
        the single message loop, a callback that reloads for longer than the presence
        TTL would park the loop and let this worker's own presence key EXPIRE while it
        is alive and applying — the census would drop the live worker and the report
        cut would misclassify its ``timed_out`` as ``departed``. The callback's own
        awaits yield control, so this task refreshes presence throughout a long apply."""
        interval = self._settings.heartbeat_ttl / 3
        while True:
            await asyncio.sleep(interval)
            await self._register_presence(r, origin, presence_key)

    def _heartbeat_failure(self, heartbeat: asyncio.Task[None], origin: FleetOrigin) -> BaseException:
        """Turn a self-terminated heartbeat task into the failure that forces a
        reconnect, logging it at ERROR immediately.

        A transport error (the usual presence-refresh failure) flows back into the
        outer reconnect-with-backoff loop, which re-subscribes and re-registers
        presence. A heartbeat that returned without raising — it never should, its
        loop is unbounded — becomes a loud ``RuntimeError`` on the same path."""
        exc = heartbeat.exception()
        if exc is None:
            exc = RuntimeError("worker bus: presence heartbeat returned without an error")
        logger.error("worker bus: presence heartbeat for %s stopped; forcing reconnect", origin.origin, exc_info=exc)
        return exc

    async def _stop_heartbeat(self, heartbeat: asyncio.Task[None] | None) -> None:
        """Cancel and await the presence-heartbeat task during teardown, so it never
        leaks past the subscription. A heartbeat that already died of a transport error
        (the dropped connection the subscription is reconnecting from) is logged loudly
        and swallowed so teardown still completes."""
        if heartbeat is None:
            return
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("worker bus: presence heartbeat task terminated abnormally", exc_info=True)

    async def _register_presence(self, r: Any, origin: FleetOrigin, presence_key: str) -> None:
        value = json.dumps({"kind": origin.kind.value, "pid": origin.pid})
        await r.set(presence_key, value, px=int(self._settings.heartbeat_ttl * 1000))

    async def _handle_op(
        self,
        r: Any,
        origin: FleetOrigin,
        callback: Callable[[dict[str, Any]], Awaitable[Any]],
        msg: dict[str, Any],
    ) -> None:
        frame = _decode(msg["data"])
        if frame is None:
            return
        if frame.get("origin") == origin.origin:
            # Echo-skip: the publisher applied this op locally before broadcasting and
            # synthesizes its own self entry — delivering it back would double-apply.
            return
        targets = frame.get("targets")
        # Test presence, not truthiness: ``targets is None`` reaches every worker, a
        # list reaches exactly its members — so an empty list reaches nobody, matching
        # the publisher's empty expected-origin set (no silent sibling apply).
        if targets is not None and origin.origin not in targets:
            return
        reply_to = frame.get("reply_to")

        if reply_to:
            await self._reply(r, reply_to, {"origin": origin.origin, "phase": "received"})

        op_payload = {k: v for k, v in frame.items() if k not in _TRANSPORT_KEYS}
        try:
            result = await callback(op_payload)
            terminal: dict[str, Any] = {
                "origin": origin.origin,
                "phase": "terminal",
                "outcome": OpOutcome.applied.value,
            }
            if result is not None:
                terminal["payload"] = result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("worker bus: op %r failed on %s", op_payload.get("op"), origin.origin, exc_info=True)
            terminal = {
                "origin": origin.origin,
                "phase": "terminal",
                "outcome": OpOutcome.failed.value,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if reply_to:
            await self._reply(r, reply_to, terminal)

    async def _reply(self, r: Any, channel: str, frame: dict[str, Any]) -> None:
        await r.publish(channel, json.dumps(frame))

    async def _teardown(self, r: Any, pubsub: Any, presence_key: str) -> None:
        """Leave the census and the channel; best-effort, never silent."""
        try:
            await r.delete(presence_key)
        except Exception:
            logger.warning(
                "worker bus: presence delete for %s failed (TTL will expire it)", presence_key, exc_info=True
            )
        try:
            await pubsub.unsubscribe(self._settings.channel)
            await pubsub.aclose()
        except Exception:
            logger.warning("worker bus: pub/sub close failed during teardown", exc_info=True)
