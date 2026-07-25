"""Redis store for the interactions capability.

Holds every key shape and the read/write operations behind one class so the
producer (the ``ask_user`` helper in this package) and the consumer (the API SSE
+ answer endpoint) share the exact key contract. Operations take the redis
client as an argument: each caller opens it from the interactions settings via
``client_ctx(RedisClient, settings.redis)``.

Requires Redis server >= 6.2: ``add`` extends the group's pending-deadline index
with ``ZADD ... GT`` (extend-only), an option that exists only from 6.2. Against
older servers this command errors loudly (a visible break, never a silent
degrade).

Assumes a single-node Redis (not Redis Cluster): the phantom-purge Lua drops
per-group index members read at runtime rather than from ``KEYS`` (the number of
expired groups is variable), which a Cluster would reject as an undeclared-key
access. The interactions keys share one prefix and are not hash-tag co-located,
so single-node is the operating assumption.

Loud by contract — no swallowed errors, no silent fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Literal, cast, overload

from redis.asyncio import Redis
from redis.exceptions import WatchError
from tai42_contract.interactions import (
    InteractionRequest,
    InteractionResponse,
    InteractionState,
)

ADD_EVENT = "interaction.add"
ANSWERED_EVENT = "interaction.answered"
REMOVED_EVENT = "interaction.removed"
_EVENTS_MAXLEN = 10000

# Atomic phantom self-heal for the pending-deadline index. A waiter killed
# mid-flight (SIGKILL/OOM) never runs cleanup, so its group lingers in
# ``pending_key``. This script — run on every ``add`` — reads the groups whose
# furthest question deadline has passed and, per expired group, drops it from BOTH
# the pending index and the parallel deadline index. It leaves ``count_key``
# untouched: the count shares the group's state ``idle_ttl`` (see ``add``), so a
# surviving state always keeps a live count and a genuinely-dead group's count
# expires on that same basis — death and revival stay symmetric, and a group that
# revives after a purge cannot re-seed a torn count. Reading the expired set
# INSIDE the script makes the correlated multi-index delete atomic: a concurrent
# ``add`` that revives a group (later deadline via ``ZADD GT``, re-added to
# ``pending_key``) between a would-be read and delete cannot be wrongly purged,
# because the script re-reads the current deadline index rather than acting on a
# stale snapshot.
#
# The group of the ``add`` running this purge is SKIPPED: this call is about to
# make that group live (its future deadline is not recorded via ``ZADD GT`` until
# the pipeline that follows the purge), so scanning it as "expired" and purging it
# would drop a group that is gaining a live question — invariant (b). The
# phantom self-heal of a genuinely dead group still fires, driven by any UNRELATED
# ``add``.
#   KEYS[1] = pending_deadline_key,  KEYS[2] = pending_key
#   ARGV[1] = now_ms (purge cutoff),  ARGV[2] = group to skip (the current add's group)
_PENDING_PURGE_LUA = """
-- interactions:pending-deadline-purge
local current = ARGV[2]
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], 0, ARGV[1])
local purged = 0
for _, group in ipairs(expired) do
    if group ~= current then
        redis.call('ZREM', KEYS[2], group)
        redis.call('ZREM', KEYS[1], group)
        purged = purged + 1
    end
end
return purged
"""

# Atomic reserve-and-check for the ``max_concurrent`` cap. The open index carries
# no TTL, so a SIGKILLed waiter's member lingers; this script first purges every
# member whose deadline has passed, then admits the caller — adding its open-index
# member — ONLY while the live count is below ``limit``. ZCARD and the ZADD run in
# one server round trip, so a concurrent burst can never overshoot the cap the way
# a separate count-then-add pair can (the check-then-act gap between two commands).
#   KEYS[1] = open_key
#   ARGV[1] = now_ms (stale-member cutoff),  ARGV[2] = limit,
#   ARGV[3] = timeout_at_ms (member score),  ARGV[4] = interaction_id (member)
_OPEN_RESERVE_LUA = """
-- interactions:open-slot-reserve
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
if tonumber(redis.call('ZCARD', KEYS[1])) >= tonumber(ARGV[2]) then
    return 0
end
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
return 1
"""


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _created_ms(request: InteractionRequest) -> int:
    return int(request.created_at.timestamp() * 1000)


def _timeout_ms(request: InteractionRequest) -> int:
    return int(request.timeout_at.timestamp() * 1000)


@overload
def as_str(value: None) -> None: ...
@overload
def as_str(value: str | bytes | bytearray) -> str: ...
def as_str(value: str | bytes | bytearray | None) -> str | None:
    """Normalize a redis value to ``str`` whether the client decodes or not."""
    if value is None:
        return None
    return value.decode() if isinstance(value, (bytes, bytearray)) else value


class InteractionStore:
    def __init__(self, key_prefix: str) -> None:
        self._p = key_prefix

    # -- key shapes ----------------------------------------------------------

    def group_key(self, group_id: str) -> str:
        return f"{self._p}group:{group_id}"

    def state_key(self, interaction_id: str) -> str:
        return f"{self._p}state:{interaction_id}"

    def reply_key(self, interaction_id: str) -> str:
        return f"{self._p}reply:{interaction_id}"

    def ticket_key(self, ticket: str) -> str:
        return f"{self._p}ticket:{ticket}"

    @property
    def pending_key(self) -> str:
        return f"{self._p}pending"

    @property
    def pending_deadline_key(self) -> str:
        """Parallel index to ``pending_key``, scored by each group's FURTHEST
        question deadline (extend-only via ``ZADD GT``). ``pending_key`` is scored
        by creation TIME, not deadline — ``add`` sets a group's score to its most
        recent question's ``created_at`` — so the reconnect backlog reads in
        creation-timestamp order rather than deadline order; only this parallel
        index carries the deadline the atomic phantom purge keys on, and the purge
        never rescores ``pending_key``."""
        return f"{self._p}pending:deadline"

    @property
    def open_key(self) -> str:
        return f"{self._p}open"

    @property
    def _count_prefix(self) -> str:
        return f"{self._p}pending:count:"

    def count_key(self, group_id: str) -> str:
        return f"{self._count_prefix}{group_id}"

    @property
    def events_key(self) -> str:
        return f"{self._p}events"

    # -- writes --------------------------------------------------------------

    async def add(
        self,
        r: Redis,
        request: InteractionRequest,
        idle_ttl: int,
        ticket: str | None = None,
        ticket_ttl: int | None = None,
        open_member_reserved: bool = False,
    ) -> None:
        """Persist a new question: stream entry, state, pending index + deadline
        index + count, the open-index ZSET member, add-event, and refreshed TTLs.
        The TTL refresh covers the group stream, the group's ``count_key``, and the
        state hash of every interaction in the group (not just the new one) so a
        still-open question can't expire out from under a group that is otherwise
        active.

        When ``open_member_reserved`` is True the open-index member has already
        been added by ``reserve_open_slot`` (the atomic concurrency guard), so this
        call does NOT re-add it — avoiding a double ZADD. The unbounded (no
        ``max_concurrent``) path leaves it False and adds the member here.

        Additional writes:

        * ``ZREMRANGEBYSCORE(open_key, 0, now_ms)`` runs UNCONDITIONALLY on every
          call, purging open-index members whose deadline has passed — a
          SIGKILLed waiter's member would otherwise linger forever (the ZSET has
          no TTL).
        * an ATOMIC phantom self-heal (``_PENDING_PURGE_LUA``) runs first: it
          drops every group whose furthest question deadline has passed from
          ``pending_key`` + ``pending_deadline_key`` (leaving ``count_key`` to its
          ``idle_ttl``), SKIPPING this call's own group (which is about to become
          live). Reading the expired set inside the script keeps the correlated
          multi-index delete safe against a concurrent revive (invariant: a group
          with a live question is never purged).
        * ``ZADD pending_deadline_key {group: timeout_at_ms} GT`` — the group's
          entry in the parallel deadline index, extend-only so a later question
          with a SHORTER deadline never shortens it.
        * ``ZADD open_key {interaction_id: timeout_at_ms}`` for ALL formats,
          UNLESS ``open_member_reserved`` (the atomic guard already added it) — the
          member is removed on answer (``record_answer``) or cleanup
          (``prune_pending``).
        * ``count_key`` TTL is refreshed to ``idle_ttl`` on the same basis as the
          group stream and every sibling state hash, so a surviving state always
          has a live count. The phantom-purge Lua reconciles genuinely-dead groups
          on the deadline index, so the count needs no shorter deadline of its own;
          tying it to ``idle_ttl`` makes ``current is None`` at decrement a true
          torn-index invariant violation rather than a silently-read zero.
        * when ``ticket`` is given (external format): ``SET ticket_key
          interaction_id EX ticket_ttl``, mapping the callback capability to this
          interaction. The ticket is never deleted; it expires on its TTL."""
        group_key = self.group_key(request.group_id)
        state_key = self.state_key(request.interaction_id)
        count_key = self.count_key(request.group_id)
        state = InteractionState(status="pending", group_id=request.group_id, request=request)
        siblings = await r.xrange(group_key)
        # Atomic phantom self-heal over the parallel deadline index, BEFORE the new
        # question is written (its deadline is in the future, so it is never a
        # purge target). redis-py's async ``eval`` stub types a non-awaitable
        # return; it is awaitable at runtime.
        await cast(
            "Awaitable[int]",
            r.eval(
                _PENDING_PURGE_LUA,
                2,
                self.pending_deadline_key,
                self.pending_key,
                _now_ms(),
                request.group_id,
            ),
        )
        # The sensitive flag rides the state hash as a denormalized ``"1"`` (absent
        # when false), so ``record_answer`` can gate the response-body write on a
        # single ``hget`` inside its WATCH loop without deserializing the request.
        state_mapping: dict[str, str] = {
            "status": state.status,
            "group_id": state.group_id,
            "request": request.model_dump_json(),
        }
        if request.sensitive:
            state_mapping["sensitive"] = "1"
        pipe = r.pipeline()
        pipe.zremrangebyscore(self.open_key, 0, _now_ms())
        pipe.xadd(
            group_key,
            {
                "interaction_id": request.interaction_id,
                "request": request.model_dump_json(),
            },
        )
        pipe.hset(state_key, mapping=state_mapping)
        pipe.incr(count_key)
        pipe.zadd(self.pending_key, {request.group_id: _created_ms(request)})
        pipe.zadd(self.pending_deadline_key, {request.group_id: _timeout_ms(request)}, gt=True)
        if not open_member_reserved:
            # The atomic guard (``reserve_open_slot``) already added this member;
            # re-adding here would double-count the open index.
            pipe.zadd(self.open_key, {request.interaction_id: _timeout_ms(request)})
        if ticket is not None:
            if ticket_ttl is None:
                raise ValueError("add(): ticket given without ticket_ttl")
            pipe.set(self.ticket_key(ticket), request.interaction_id, ex=ticket_ttl)
        pipe.xadd(
            self.events_key,
            {
                "type": ADD_EVENT,
                "interaction_id": request.interaction_id,
                "group_id": request.group_id,
            },
            maxlen=_EVENTS_MAXLEN,
            approximate=True,
        )
        pipe.expire(group_key, idle_ttl)
        pipe.expire(state_key, idle_ttl)
        for _entry_id, fields in siblings:
            sibling_id = as_str(fields.get("interaction_id") or fields.get(b"interaction_id"))
            if sibling_id:
                pipe.expire(self.state_key(sibling_id), idle_ttl)
        # count_key's EXPIRE is issued LAST, after every state EXPIRE: an absolute
        # expiry is (server time at command execution) + idle_ttl, and pipeline
        # commands run in order, so issuing it last makes count_key outlive every
        # state hash. A surviving pending state therefore always has a live count,
        # so ``current is None`` at a decrement is a genuine torn-index invariant
        # violation (raise-worthy), never a spurious miss at the expiry boundary.
        pipe.expire(count_key, idle_ttl)
        await pipe.execute()

    async def reserve_open_slot(self, r: Redis, request: InteractionRequest, limit: int) -> bool:
        """Atomically reserve an open-index slot under the ``max_concurrent`` cap.

        Purges stale open members (deadline passed), then — in the SAME server
        round trip — admits this question by adding its open-index member ONLY
        while the live open count is below ``limit``. Returns ``True`` when the
        slot was reserved (the member is now in the open index), ``False`` when the
        cap is already full (nothing written). Because the check and the add are
        one atomic step, a concurrent burst admits exactly ``limit`` callers and
        refuses the rest — no unbounded overshoot.

        A ``True`` reservation adds the SAME member ``add`` would, so the caller
        must then invoke ``add(..., open_member_reserved=True)`` to avoid a double
        ZADD. redis-py's async ``eval`` stub types a non-awaitable return; it is
        awaitable at runtime."""
        reserved = await cast(
            "Awaitable[int]",
            r.eval(
                _OPEN_RESERVE_LUA,
                1,
                self.open_key,
                _now_ms(),
                limit,
                _timeout_ms(request),
                request.interaction_id,
            ),
        )
        return bool(reserved)

    async def record_answer(
        self,
        r: Redis,
        response: InteractionResponse,
        group_id: str,
        reply_ttl: int,
        ticket: str | None = None,
        ticket_ttl: int | None = None,
    ) -> bool:
        """Atomically claim and record an answer: mark answered, remove the
        open-index member, decrement the group's pending count (drop the group
        from the index at zero), wake the caller, append the answered-event. The
        reply key gets a short TTL so a late answer to a timed-out question
        expires instead of resurrecting it.

        When ``ticket`` is given (the callback doors pass the resolved ticket +
        ``idle_ttl_seconds``): ``EXPIRE ticket_key ticket_ttl`` inside the MULTI,
        refreshing the idempotency window to match the answered state's lifetime
        so late provider retries still resolve the ticket and reach the
        already-answered path. The ticket is never deleted (EXPIRE on an
        already-expired key is a harmless no-op). The ``/answer`` and prune paths
        pass no ticket (no refresh).

        When the question was marked ``sensitive`` at ``add`` time, the answered
        state records ONLY ``{"status": "answered"}`` — the response body is never
        written into the durable hash. The reply-key RPUSH is unchanged, so the
        blocked waiter still receives the full answer; only the persisted record
        drops the body. A late duplicate to a sensitive question therefore takes
        the already-answered path with no body available — by design.

        Returns ``True`` when this call claimed the answer, ``False`` when the
        interaction was missing or already answered (a lost duplicate race) —
        in which case nothing is written and no caller is woken."""
        if ticket is not None and ticket_ttl is None:
            raise ValueError("record_answer(): ticket given without ticket_ttl")
        interaction_id = response.interaction_id
        state_key = self.state_key(interaction_id)
        count_key = self.count_key(group_id)
        reply_key = self.reply_key(interaction_id)
        response_json = response.model_dump_json()

        async with r.pipeline() as pipe:
            while True:
                try:
                    # Watch the count key too: a concurrent add() to the same
                    # group INCRs it, which must invalidate this transaction so
                    # the at-zero cleanup can't drop a group that just gained a
                    # new open question.
                    await pipe.watch(state_key, count_key)
                    # redis-py's async stubs type pre-MULTI pipeline reads with the
                    # sync (non-awaitable) return; the value is awaitable at runtime.
                    status = as_str(await cast("Awaitable[str | None]", pipe.hget(state_key, "status")))
                    if status is None or status == "answered":
                        await pipe.reset()
                        return False
                    sensitive = as_str(await cast("Awaitable[str | None]", pipe.hget(state_key, "sensitive"))) == "1"
                    current = await pipe.get(count_key)
                    if current is None:
                        # count_key shares the pending state's idle_ttl, so a live
                        # (pending) state ALWAYS has a live count. A missing count
                        # here is a torn index, not a zero — raise, never guess.
                        raise RuntimeError(
                            f"pending count missing for group {group_id!r} with a live state {interaction_id!r}"
                        )
                    remaining = int(current) - 1
                    pipe.multi()
                    # A sensitive question persists only the answered status — the
                    # body is deliberately never written to the durable hash.
                    answered_mapping = {"status": "answered"}
                    if not sensitive:
                        answered_mapping["response"] = response_json
                    pipe.hset(state_key, mapping=answered_mapping)
                    pipe.decr(count_key)
                    pipe.zrem(self.open_key, interaction_id)
                    if ticket is not None:
                        # ``ticket_ttl`` is guaranteed non-None here (guarded at
                        # the top); pin it for the type checker.
                        assert ticket_ttl is not None
                        pipe.expire(self.ticket_key(ticket), ticket_ttl)
                        # Refresh the state key to the same window as the ticket:
                        # the idempotent already-answered path resolves the ticket
                        # AND reads the state, so a state expiring before the ticket
                        # would turn a late provider retry into a 404. Tying both to
                        # ticket_ttl keeps the answered state alive as long as the
                        # ticket can still resolve.
                        pipe.expire(state_key, ticket_ttl)
                    pipe.rpush(reply_key, response_json)
                    pipe.expire(reply_key, reply_ttl)
                    pipe.xadd(
                        self.events_key,
                        {
                            "type": ANSWERED_EVENT,
                            "interaction_id": interaction_id,
                            "group_id": group_id,
                        },
                        maxlen=_EVENTS_MAXLEN,
                        approximate=True,
                    )
                    if remaining <= 0:  # this was the group's last open question
                        pipe.zrem(self.pending_key, group_id)
                        pipe.zrem(self.pending_deadline_key, group_id)
                        pipe.delete(count_key)
                    await pipe.execute()
                    return True
                except WatchError:
                    continue

    async def prune_pending(self, r: Redis, interaction_id: str, group_id: str) -> bool:
        """Remove a still-open question that is being abandoned (cancel-cleanup or
        the timeout path). Status-gated exactly like ``record_answer``, INCLUDING
        its ``except WatchError: continue`` retry loop: an answer committing
        between the status read and EXEC fires WatchError, the retry then reads
        ``answered`` and returns False cleanly.

        WATCHes the state key AND the count key — the count WATCH for the same
        reason ``record_answer`` has it: a concurrent ``add()`` to the group INCRs
        the count and must invalidate this transaction, or the at-zero cleanup
        would delete the count key and drop the group from the pending index while
        a just-added sibling is still open.

        When ``status`` is ``None`` or ``"answered"`` this is a no-op returning
        ``False`` with NO writes (an answered interaction must never be pruned).
        When still ``pending``, in one MULTI: delete the state key, ``ZREM
        open_key``, ``DECR`` the group count, and at zero delete the count key +
        ``ZREM`` the group from BOTH the pending index and the parallel
        pending-deadline index; then append an ``interaction.removed`` event so a
        live SSE consumer can drop the pruned question. Returns ``True`` when it
        pruned, ``False`` otherwise."""
        state_key = self.state_key(interaction_id)
        count_key = self.count_key(group_id)

        async with r.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(state_key, count_key)
                    status = as_str(await cast("Awaitable[str | None]", pipe.hget(state_key, "status")))
                    if status is None or status == "answered":
                        await pipe.reset()
                        return False
                    current = await pipe.get(count_key)
                    if current is None:
                        # count_key shares the pending state's idle_ttl, so a live
                        # (pending) state ALWAYS has a live count. A missing count
                        # here is a torn index, not a zero — raise, never guess.
                        raise RuntimeError(
                            f"pending count missing for group {group_id!r} with a live state {interaction_id!r}"
                        )
                    remaining = int(current) - 1
                    pipe.multi()
                    pipe.delete(state_key)
                    pipe.zrem(self.open_key, interaction_id)
                    pipe.decr(count_key)
                    if remaining <= 0:  # this was the group's last open question
                        pipe.zrem(self.pending_key, group_id)
                        pipe.zrem(self.pending_deadline_key, group_id)
                        pipe.delete(count_key)
                    pipe.xadd(
                        self.events_key,
                        {
                            "type": REMOVED_EVENT,
                            "interaction_id": interaction_id,
                            "group_id": group_id,
                        },
                        maxlen=_EVENTS_MAXLEN,
                        approximate=True,
                    )
                    await pipe.execute()
                    return True
                except WatchError:
                    continue

    # -- reads ---------------------------------------------------------------

    async def resolve_ticket(self, r: Redis, ticket: str) -> str | None:
        """Return the interaction id a callback ticket maps to, or ``None`` when
        the ticket never existed or has expired (lookup-by-exact-key IS the
        comparison — no user-supplied string is compared in Python)."""
        return as_str(await cast("Awaitable[str | bytes | None]", r.get(self.ticket_key(ticket))))

    async def count_open(self, r: Redis) -> int:
        """The live open-question count: purge open-index members whose deadline
        has passed, then ``ZCARD``. All ``open_key`` access stays inside the store
        so no caller touches the key inline. The ``max_concurrent`` cap enforces
        itself atomically in ``reserve_open_slot``; this read serves callers that
        only need the current count."""
        await r.zremrangebyscore(self.open_key, 0, _now_ms())
        return await cast("Awaitable[int]", r.zcard(self.open_key))

    def _state_from_raw(self, raw: dict[str | bytes, str | bytes]) -> InteractionState | None:
        """Build an ``InteractionState`` from a raw state-hash mapping (as returned
        by ``HGETALL``), or ``None`` when the hash is empty (missing/expired)."""
        if not raw:
            return None
        fields = {as_str(k): as_str(v) for k, v in raw.items()}
        request = InteractionRequest.model_validate_json(fields["request"])
        response_json = fields.get("response")
        response = InteractionResponse.model_validate_json(response_json) if response_json else None
        return InteractionState(
            # Pydantic validates the stored status against the Literal at runtime.
            status=cast("Literal['pending', 'answered']", fields["status"]),
            group_id=fields["group_id"],
            request=request,
            response=response,
        )

    async def get_state(self, r: Redis, interaction_id: str) -> InteractionState | None:
        # redis-py's async stubs type ``hgetall`` with the sync (non-awaitable)
        # return; it is awaitable at runtime.
        raw = await cast("Awaitable[dict[str | bytes, str | bytes]]", r.hgetall(self.state_key(interaction_id)))
        return self._state_from_raw(raw)

    async def backlog(self, r: Redis) -> list[InteractionRequest]:
        """The pending-question backlog an SSE consumer replays on connect, in
        ``pending_key`` score order (each group's most-recent question
        ``created_at``) then stream order within a group. Performs the same
        reconciliation an inline read would, so the route
        holds zero store-key knowledge:

        * a phantom group (its stream expired but it lingers in the index) is
          pruned from BOTH ``pending_key`` and ``pending_deadline_key`` and skipped;
        * an answered or missing state is skipped;
        * an abandoned pending question (past its deadline — e.g. a SIGKILLed
          waiter whose cleanup never ran) is pruned via ``prune_pending`` and
          skipped, so the badge/list stay honest.

        Each group's per-entry state reads are batched into ONE pipeline (a single
        round trip for the group's open questions) rather than an N+1 of
        per-question ``HGETALL`` calls."""
        now = datetime.now(UTC)
        backlog: list[InteractionRequest] = []
        for raw_group in await r.zrange(self.pending_key, 0, -1):
            group_id = as_str(raw_group)
            entries = await r.xrange(self.group_key(group_id))
            if not entries:
                # The group's stream expired but lingered in the indexes — prune
                # the phantom from both so the badge/list don't count it.
                await r.zrem(self.pending_key, group_id)
                await r.zrem(self.pending_deadline_key, group_id)
                continue
            requests = [
                InteractionRequest.model_validate_json(as_str(fields.get("request") or fields.get(b"request")))
                for _entry_id, fields in entries
            ]
            # One pipeline for the whole group's state hashes — no N+1.
            pipe = r.pipeline()
            for req in requests:
                pipe.hgetall(self.state_key(req.interaction_id))
            raw_states = await pipe.execute()
            for req, raw in zip(requests, raw_states, strict=True):
                state = self._state_from_raw(raw)
                if state is None or state.status != "pending":
                    continue
                if now >= state.request.timeout_at:
                    await self.prune_pending(r, req.interaction_id, group_id)
                    continue
                backlog.append(req)
        return backlog

    async def wait_for_reply(
        self, r: Redis, reply_to: str, timeout_seconds: float, grace_seconds: float
    ) -> InteractionResponse | None:
        """Block on the reply channel up to ``timeout_seconds``. Returns the
        recorded response, or ``None`` when the budget elapses with no answer.

        The BLPOP blocks legitimately for the whole answer budget, so its
        connection carries no socket read timeout (the caller strips it). To keep
        a black-holed redis from wedging the loop task forever, the BLPOP is wrapped
        in an outer ``asyncio.wait_for(timeout_seconds + grace_seconds)``: ``grace``
        is the slack past the server-side block window, passed in by the caller (the
        store holds no settings). A timeout there means the connection is presumed
        stalled — a loud ``RuntimeError``, DISTINCT from the normal no-answer path
        (BLPOP nil -> ``None`` -> ``InteractionTimeoutError`` in the caller), which
        is unchanged."""
        # redis-py's async stubs type ``blpop`` with the sync (non-awaitable)
        # return and an ``int`` timeout; at runtime it is awaitable and accepts a
        # float (fractional-second) timeout. The value is passed through untouched
        # — casting it to ``int`` would truncate a sub-second budget to 0, which
        # BLPOP reads as "block forever". The ignore only silences the stub's
        # int-only ``timeout`` kwarg; redis supports a float timeout natively.
        try:
            result = await asyncio.wait_for(
                cast(
                    "Awaitable[tuple[Any, Any] | None]",
                    r.blpop([reply_to], timeout=timeout_seconds),  # type: ignore[arg-type]
                ),
                timeout=timeout_seconds + grace_seconds,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "interactions reply wait: redis BLPOP returned nothing within "
                f"budget+{grace_seconds}s grace — connection presumed stalled"
            ) from exc
        if result is None:
            return None
        _, value = result
        return InteractionResponse.model_validate_json(as_str(value))
