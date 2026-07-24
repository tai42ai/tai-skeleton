"""The answer/record store — keyspaces 1-3 and 6 of the conversation bridge, all transient
runtime state (NOT a backup section) and all Redis-backed:

1. Inbound dedupe: ``conversations:dedupe:{channel}:{provider_message_id}`` → the
   ``message_id`` that first claimed the pair. The claim has no release path, so it is
   taken only once a durable record already stands behind it.
2. Answer record: ``conversations:record:{message_id}`` — intake, produced answer and
   delivery state, split into a content blob plus the delivery-control fields the atomic
   transitions mutate. The retention TTL is applied ONLY on reaching a terminal state.
3. Outbound-id reverse index: ``conversations:outbound:{channel}:{outbound_id}`` →
   ``message_id``, so an out-of-band receipt resolves back to its record.
6. Per-status record index: ``conversations:status:{delivery_status}`` → the ``message_id``s
   in that state, moved by the same atomic step that moves the record. Every listing reads
   it, so a sweep costs the work outstanding and not the whole retained keyspace.

Every exactly-once transition is a single Lua step guarded on the record's current
``delivery_status``, so racing writers produce ONE outcome, never two.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis as AsyncRedis
from tai42_contract.conversations import DeliveryReceipt
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.conversations.models import TERMINAL_STATUSES, ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations.errors import NotSupportedError
from tai42_skeleton.utils.redis_typing import awaited, eval_script

logger = logging.getLogger(__name__)

_NO_BACKEND = "conversation answer records require the redis conversations backend"

# Hash field names on a record key: ``data`` is the content JSON, the rest are the
# delivery-control fields the atomic transitions mutate.
_F_DATA = "data"
_F_STATUS = "delivery_status"
_F_OUTBOUND = "outbound_ids"
_F_ATTEMPTS = "attempts"
_F_GRACE = "grace_deadline"
_F_UPDATED = "updated_at"

# Every record-mutating script takes the same key layout: KEYS[1]=record key, KEYS[2..7]=the
# per-status indexes, KEYS[8]=the index the transition moves the record INTO.
_INDEXED_STATUSES = tuple(DeliveryStatus)
_TARGET_INDEX_KEY = f"KEYS[{2 + len(_INDEXED_STATUSES)}]"

# A live record's index member outlives nothing; a terminal one expires with its row.
_NO_EXPIRY_SCORE = "+inf"


def _reindex(member_argv: str, score_argv: str) -> str:
    """Lua moving the record's id into the target status index and out of every other, so
    exactly one index names it and a listing never walks the record keyspace."""
    return f"""
for i = 2, {1 + len(_INDEXED_STATUSES)} do redis.call('ZREM', KEYS[i], {member_argv}) end
redis.call('ZADD', {_TARGET_INDEX_KEY}, {score_argv}, {member_argv})
"""


# Atomic get-or-set of the inbound-dedupe marker: returns the message_id owning the pair —
# the caller's on a fresh claim, the prior turn's on a redelivery.
# KEYS[1]=dedupe key; ARGV = message_id, ttl_seconds.
_CLAIM_INBOUND_LUA = """
-- conversations:dedupe:claim
local existing = redis.call('GET', KEYS[1])
if existing then return existing end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return ARGV[1]
"""

# Write a freshly minted record, applying the retention TTL in the SAME step when it is
# created already terminal so such a record can never be left without one. An ``accepted``
# record is created ALREADY holding its intake lease, so no window exists in which a live
# turn looks stranded. ARGV = content_json, delivery_status, outbound_json, attempts,
# updated_at, ttl_ms ('' for a record that must not expire yet), intake_claim ('' off the
# intake path), message_id, index_score.
_CREATE_LUA = f"""
-- conversations:record:create
redis.call('HSET', KEYS[1], 'data', ARGV[1], 'delivery_status', ARGV[2], 'outbound_ids', ARGV[3],
  'attempts', ARGV[4], 'claim', '', 'grace_deadline', '', 'updated_at', ARGV[5], 'intake_claim', ARGV[7])
if ARGV[6] ~= '' then redis.call('PEXPIRE', KEYS[1], ARGV[6]) end
{_reindex("ARGV[8]", "ARGV[9]")}
return 1
"""

# Move an intake record from ``accepted`` to ``pending_delivery`` carrying the turn's
# outcome, releasing the intake lease: 1 transitioned, 0 no longer at intake, -1 gone.
# Guarded on the current status, so a finishing turn and a re-drive produce ONE outcome.
# ARGV = content_json, now, message_id, index_score.
_COMPLETE_TURN_LUA = f"""
-- conversations:record:complete_turn
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status ~= 'accepted' then return 0 end
redis.call('HSET', KEYS[1], 'data', ARGV[1], 'delivery_status', 'pending_delivery', 'updated_at', ARGV[2],
  'intake_claim', '')
{_reindex("ARGV[3]", "ARGV[4]")}
return 1
"""

# Take (or refresh) the intake lease under a worker token — the liveness marker a running
# turn holds: 1 held, 0 a DIFFERENT worker's lease is still live, -1 gone, -2 the record
# has left intake. Claim value is ``token:expiry``. The holder refreshes; a re-drive may
# adopt only a LAPSED lease. KEYS[1]=record key; ARGV = now, lease_seconds, token.
_CLAIM_INTAKE_LUA = """
-- conversations:record:intake_claim
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status ~= 'accepted' then return -2 end
local claim = redis.call('HGET', KEYS[1], 'intake_claim')
local now = tonumber(ARGV[1])
if claim and claim ~= '' then
  local sep = string.find(claim, ':')
  local ctoken = string.sub(claim, 1, sep - 1)
  local cexp = tonumber(string.sub(claim, sep + 1))
  if ctoken ~= ARGV[3] and cexp > now then return 0 end
end
redis.call('HSET', KEYS[1], 'intake_claim', ARGV[3] .. ':' .. tostring(now + tonumber(ARGV[2])))
return 1
"""

# Take (or refresh) the exactly-once delivery lease under a worker token: 1 won, 0 the
# record is already sent (provisional) or terminal or a DIFFERENT worker holds a live lease,
# -1 gone, -2 still at intake and carrying no answer. Only a pending_delivery record is
# claimable for a send; a provisional one awaits a receipt, not a re-send. The holder may
# re-claim to extend; anyone else waits for expiry. Claim value is ``token:expiry``.
# KEYS[1]=record key; ARGV = now, lease_seconds, token.
_CLAIM_DELIVERY_LUA = """
-- conversations:record:claim
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status == 'accepted' then return -2 end
if status ~= 'pending_delivery' then return 0 end
local claim = redis.call('HGET', KEYS[1], 'claim')
local now = tonumber(ARGV[1])
if claim and claim ~= '' then
  local sep = string.find(claim, ':')
  local ctoken = string.sub(claim, 1, sep - 1)
  local cexp = tonumber(string.sub(claim, sep + 1))
  if ctoken ~= ARGV[3] and cexp > now then return 0 end
end
redis.call('HSET', KEYS[1], 'claim', ARGV[3] .. ':' .. tostring(now + tonumber(ARGV[2])))
return 1
"""


def _foreign_lease_guard(now_argv: str, token_argv: str) -> str:
    """Lua that returns -3 when a DIFFERENT worker holds a live delivery lease — the check
    every delivery-state write takes before it mutates, so a worker whose lease lapsed and
    was taken over cannot overwrite the holder's progress."""
    return f"""
local claim = redis.call('HGET', KEYS[1], 'claim')
if claim and claim ~= '' then
  local sep = string.find(claim, ':')
  if string.sub(claim, 1, sep - 1) ~= {token_argv}
     and tonumber(string.sub(claim, sep + 1)) > tonumber({now_argv}) then
    return -3
  end
end
"""


# Move a record to ``provisional``, recording the outbound ids and grace deadline and
# releasing the lease: 1 transitioned, 0 already terminal, -1 gone, -3 a foreign live lease.
# ARGV = outbound_json, attempts, grace_deadline, now, token, message_id, index_score.
_PROVISIONAL_LUA = f"""
-- conversations:record:provisional
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status == 'delivered' or status == 'failed' or status == 'shed' then return 0 end
{_foreign_lease_guard("ARGV[4]", "ARGV[5]")}
redis.call('HSET', KEYS[1], 'delivery_status', 'provisional', 'outbound_ids', ARGV[1],
  'attempts', ARGV[2], 'grace_deadline', ARGV[3], 'updated_at', ARGV[4], 'claim', '')
{_reindex("ARGV[6]", "ARGV[7]")}
return 1
"""

# Terminal delivered write from the send path: 1 transitioned, 0 already delivered
# (idempotent), -1 gone, -2 already failed (a conflict the caller logs), -3 a foreign live
# lease. Sets the retention TTL.
# ARGV = outbound_json, attempts, now, ttl_ms, token, message_id, index_score.
_DELIVERED_LUA = f"""
-- conversations:record:delivered
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status == 'failed' or status == 'shed' then return -2 end
if status == 'delivered' then return 0 end
{_foreign_lease_guard("ARGV[3]", "ARGV[5]")}
redis.call('HSET', KEYS[1], 'delivery_status', 'delivered', 'outbound_ids', ARGV[1],
  'attempts', ARGV[2], 'updated_at', ARGV[3], 'claim', '', 'grace_deadline', '')
redis.call('PEXPIRE', KEYS[1], ARGV[4])
{_reindex("ARGV[6]", "ARGV[7]")}
return 1
"""

# Terminal failed write from the send path (retries exhausted): 1 transitioned, 0 already
# failed, -1 gone, -2 the send already completed (delivered/shed/provisional), -3 a foreign
# live lease. Sets the retention TTL.
# ARGV = attempts, now, ttl_ms, token, message_id, index_score.
_FAILED_LUA = f"""
-- conversations:record:failed
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
if status == 'delivered' or status == 'shed' or status == 'provisional' then return -2 end
if status == 'failed' then return 0 end
{_foreign_lease_guard("ARGV[2]", "ARGV[4]")}
redis.call('HSET', KEYS[1], 'delivery_status', 'failed', 'attempts', ARGV[1],
  'updated_at', ARGV[2], 'claim', '', 'grace_deadline', '')
redis.call('PEXPIRE', KEYS[1], ARGV[3])
{_reindex("ARGV[5]", "ARGV[6]")}
return 1
"""

# Ingest an out-of-band receipt against a fully sent (``provisional``) record: 1
# transitioned, 0 already in the target terminal state, -1 gone, -2 conflicting terminal
# state, -3 the send has not finished. ARGV = target_status, now, ttl_ms, message_id,
# index_score.
_RECEIPT_LUA = f"""
-- conversations:record:receipt
local status = redis.call('HGET', KEYS[1], 'delivery_status')
if not status then return -1 end
local target = ARGV[1]
if status == target then return 0 end
if status == 'delivered' or status == 'failed' or status == 'shed' then return -2 end
if status ~= 'provisional' then return -3 end
redis.call('HSET', KEYS[1], 'delivery_status', target, 'updated_at', ARGV[2],
  'claim', '', 'grace_deadline', '')
redis.call('PEXPIRE', KEYS[1], ARGV[3])
{_reindex("ARGV[4]", "ARGV[5]")}
return 1
"""


# Delete a record and its index membership in ONE step, so no listing can name a row that
# is gone. Returns the number of keys removed. ARGV = message_id.
_DELETE_LUA = f"""
-- conversations:record:delete
local removed = redis.call('DEL', KEYS[1])
for i = 2, {1 + len(_INDEXED_STATUSES)} do redis.call('ZREM', KEYS[i], ARGV[1]) end
return removed
"""


@dataclass(frozen=True)
class PendingWork:
    """A non-terminal record a delivery pass found, with the control fields it needs to
    decide the next move."""

    message_id: str
    delivery_status: DeliveryStatus
    attempts: int
    grace_deadline: float | None


class ConversationRecordStore:
    """The Redis-backed answer/record store (keyspaces 1-3). Construction refuses with a
    loud 501 without the redis conversations backend — nothing here may be persisted to
    state that vanishes with the process."""

    def __init__(self, settings: ConversationsSettings) -> None:
        if settings.in_memory:
            raise NotSupportedError(_NO_BACKEND)
        self.settings = settings

    # -- the key layout every record-mutating script takes --------------------

    def _record_keys(self, message_id: str, target: DeliveryStatus | None = None) -> list[str]:
        """``[record key, every status index, the target status index]``. ``target`` is
        omitted only by a script that writes no status."""
        keys = [self.settings.record_key(message_id)]
        keys.extend(self.settings.status_index_key(status.value) for status in _INDEXED_STATUSES)
        if target is not None:
            keys.append(self.settings.status_index_key(target.value))
        return keys

    def _index_score(self, status: DeliveryStatus, now: float) -> str:
        """The index member's expiry: a terminal row's member is swept with the row it
        names, a live one's is never swept."""
        if status not in TERMINAL_STATUSES:
            return _NO_EXPIRY_SCORE
        return str(now + self.settings.answer_retention_ttl_seconds)

    # -- inbound dedupe (keyspace 1) -----------------------------------------

    async def get_inbound_owner(self, channel: str, provider_message_id: str) -> str | None:
        """The ``message_id`` owning ``(channel, provider_message_id)``, or ``None`` when
        unclaimed — a door's redelivery fast path. It claims nothing, so
        :meth:`claim_inbound` stays the authority concurrent accepts arbitrate through."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            owner = await awaited(r.get(self.settings.dedupe_key(channel, provider_message_id)))
        if owner is None:
            return None
        return owner.decode() if isinstance(owner, bytes) else owner

    async def claim_inbound(self, channel: str, provider_message_id: str, message_id: str) -> str:
        """Claim ``(channel, provider_message_id)`` for ``message_id``, returning the
        ``message_id`` that OWNS the pair — the passed one on a fresh claim, the prior
        turn's on a redelivery. The claim has no release path, so the caller must take it
        only once its record is durably persisted, and discard that record on a loss."""
        key = self.settings.dedupe_key(channel, provider_message_id)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            owner = await eval_script(
                r, _CLAIM_INBOUND_LUA, 1, key, message_id, self.settings.inbound_dedupe_ttl_seconds
            )
        return owner.decode() if isinstance(owner, bytes) else owner

    # -- answer record (keyspace 2) ------------------------------------------

    def _content_blob(self, record: ConversationRecord) -> str:
        """The record's content JSON — every field except the delivery-control ones, which
        live in their own hash fields."""
        content = record.model_dump(mode="json")
        for control in ("delivery_status", "outbound_message_ids", "attempts", "updated_at"):
            content.pop(control, None)
        return json.dumps(content)

    async def create_record(self, record: ConversationRecord, *, intake_token: str | None = None) -> None:
        """Persist a freshly minted record in the state it carries (always a create — the
        ``message_id`` is a fresh uuid4). A non-terminal record carries NO expiry until it
        reaches a terminal state; one created already terminal gets the retention TTL.

        An ``accepted`` record REQUIRES ``intake_token`` and is created already holding that
        worker's intake lease; any other state requires none."""
        if (record.delivery_status is DeliveryStatus.ACCEPTED) is not (intake_token is not None):
            raise ValueError(
                "an accepted record is created holding an intake lease and any other state without one; got "
                f"{record.delivery_status.value!r} with intake_token={intake_token!r}"
            )
        now = time.time()
        terminal = record.delivery_status in TERMINAL_STATUSES
        ttl_ms = self.settings.answer_retention_ttl_seconds * 1000 if terminal else ""
        lease_until = now + self.settings.intake_claim_lease_seconds
        keys = self._record_keys(record.message_id, record.delivery_status)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            await eval_script(
                r,
                _CREATE_LUA,
                len(keys),
                *keys,
                self._content_blob(record),
                record.delivery_status.value,
                json.dumps(record.outbound_message_ids),
                record.attempts,
                record.updated_at,
                ttl_ms,
                f"{intake_token}:{lease_until}" if intake_token is not None else "",
                record.message_id,
                self._index_score(record.delivery_status, now),
            )

    async def complete_turn(self, record: ConversationRecord) -> int:
        """Move an intake record from ``accepted`` to ``pending_delivery`` carrying its
        turn's outcome: 1 transitioned, 0 no longer at intake, -1 gone. Guarded on the
        current status, so a late turn and a re-drive cannot both write an outcome."""
        if record.delivery_status is not DeliveryStatus.PENDING_DELIVERY:
            raise ValueError(f"complete_turn writes a pending_delivery record, got {record.delivery_status.value!r}")
        keys = self._record_keys(record.message_id, DeliveryStatus.PENDING_DELIVERY)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _COMPLETE_TURN_LUA,
                    len(keys),
                    *keys,
                    self._content_blob(record),
                    record.updated_at,
                    record.message_id,
                    self._index_score(DeliveryStatus.PENDING_DELIVERY, record.updated_at),
                )
            )

    async def claim_intake(self, message_id: str, now: float, token: str, lease_seconds: float) -> int:
        """Take (or refresh) the intake lease on ``message_id`` under ``token``, leased for
        ``lease_seconds``: 1 when held, 0 when a DIFFERENT worker's lease is still live, -1
        when the record is gone, -2 when it has left intake. The worker running the turn
        refreshes its own lease; a re-drive may adopt only a LAPSED one, so a live turn is
        never reaped."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _CLAIM_INTAKE_LUA,
                    1,
                    self.settings.record_key(message_id),
                    now,
                    lease_seconds,
                    token,
                )
            )

    async def delete_record(self, message_id: str) -> bool:
        """Delete a record and its index membership outright, returning whether one was
        removed — the abort path for an accept that lost the inbound claim and so owns
        nothing."""
        keys = self._record_keys(message_id)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return bool(await eval_script(r, _DELETE_LUA, len(keys), *keys, message_id))

    async def get_record(self, message_id: str) -> ConversationRecord | None:
        """The record for ``message_id``, or ``None`` when no such record exists."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            hashed = await awaited(r.hgetall(self.settings.record_key(message_id)))
        if not hashed:
            return None
        return self._from_hash(hashed)

    def _from_hash(self, hashed: dict[str, str]) -> ConversationRecord:
        data = json.loads(hashed[_F_DATA])
        data[_F_STATUS] = hashed[_F_STATUS]
        data["outbound_message_ids"] = json.loads(hashed[_F_OUTBOUND])
        data[_F_ATTEMPTS] = int(hashed[_F_ATTEMPTS])
        data[_F_UPDATED] = float(hashed[_F_UPDATED])
        return ConversationRecord.model_validate(data)

    async def claim_delivery(self, message_id: str, now: float, token: str, lease_seconds: float) -> int:
        """Take (or refresh) the exactly-once delivery lease on ``message_id`` under
        ``token``, leased for ``lease_seconds``: 1 when won, 0 when the record is already
        sent (provisional) or terminal or a different worker holds a live lease, -1 when the
        record is gone, -2 when it is still at intake and carries no answer. Only a
        ``pending_delivery`` record is claimable for a send. The token holder re-claiming
        extends its own lease; a different token waits for expiry."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _CLAIM_DELIVERY_LUA,
                    1,
                    self.settings.record_key(message_id),
                    now,
                    lease_seconds,
                    token,
                )
            )

    async def bump_attempt(self, message_id: str) -> int:
        """Increment and return the record's attempt count — one per send attempt."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(await awaited(r.hincrby(self.settings.record_key(message_id), _F_ATTEMPTS, 1)))

    async def mark_provisional(
        self, message_id: str, outbound_ids: list[str], attempts: int, now: float, token: str
    ) -> int:
        """Move ``message_id`` to ``provisional`` awaiting an async delivery receipt or
        grace expiry: 1 transitioned, 0 already terminal, -1 gone, -3 a different worker's
        live lease. ``token`` is the delivery lease this caller holds."""
        grace_deadline = now + self.settings.delivery_grace_seconds
        keys = self._record_keys(message_id, DeliveryStatus.PROVISIONAL)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _PROVISIONAL_LUA,
                    len(keys),
                    *keys,
                    json.dumps(outbound_ids),
                    attempts,
                    grace_deadline,
                    now,
                    token,
                    message_id,
                    self._index_score(DeliveryStatus.PROVISIONAL, now),
                )
            )

    async def mark_delivered(
        self, message_id: str, outbound_ids: list[str], attempts: int, now: float, token: str
    ) -> int:
        """Terminal delivered write with the retention TTL: 1 transitioned, 0 already
        delivered, -1 gone, -2 already failed, -3 a different worker's live lease.
        ``token`` is the delivery lease this caller holds."""
        keys = self._record_keys(message_id, DeliveryStatus.DELIVERED)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _DELIVERED_LUA,
                    len(keys),
                    *keys,
                    json.dumps(outbound_ids),
                    attempts,
                    now,
                    self.settings.answer_retention_ttl_seconds * 1000,
                    token,
                    message_id,
                    self._index_score(DeliveryStatus.DELIVERED, now),
                )
            )

    async def mark_failed(self, message_id: str, attempts: int, now: float, token: str) -> int:
        """Terminal failed write with the retention TTL: 1 transitioned, 0 already failed,
        -1 gone, -2 the send already completed (delivered/shed/provisional), -3 a different
        worker's live lease. ``token`` is the delivery lease this caller holds."""
        keys = self._record_keys(message_id, DeliveryStatus.FAILED)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _FAILED_LUA,
                    len(keys),
                    *keys,
                    attempts,
                    now,
                    self.settings.answer_retention_ttl_seconds * 1000,
                    token,
                    message_id,
                    self._index_score(DeliveryStatus.FAILED, now),
                )
            )

    async def ingest_receipt(self, message_id: str, receipt: DeliveryReceipt, now: float) -> int:
        """Ingest an out-of-band receipt against a fully sent (``provisional``) record: 1
        transitioned, 0 already in the receipt's terminal state, -1 gone, -2 a conflicting
        terminal state already recorded, -3 the record's send has not finished."""
        target = DeliveryStatus.DELIVERED if receipt is DeliveryReceipt.DELIVERED else DeliveryStatus.FAILED
        keys = self._record_keys(message_id, target)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            return int(
                await eval_script(
                    r,
                    _RECEIPT_LUA,
                    len(keys),
                    *keys,
                    target.value,
                    now,
                    self.settings.answer_retention_ttl_seconds * 1000,
                    message_id,
                    self._index_score(target, now),
                )
            )

    async def _indexed_ids(self, r: AsyncRedis, statuses: frozenset[DeliveryStatus], now: float) -> list[str]:
        """The ``message_id``s indexed under ``statuses``, dropping the members whose row
        has expired out from under the index first."""
        ids: list[str] = []
        for status in statuses:
            key = self.settings.status_index_key(status.value)
            await awaited(r.zremrangebyscore(key, "-inf", now))
            for member in await awaited(r.zrange(key, 0, -1)):
                ids.append(member.decode() if isinstance(member, bytes) else member)
        return ids

    async def _drop_orphan(self, r: AsyncRedis, message_id: str) -> None:
        """Unindex a member whose row is gone — a row deleted from under the index rather
        than through :meth:`delete_record`."""
        logger.warning("conversations: record %r is indexed but has no row; unindexed", message_id)
        keys = self._record_keys(message_id)
        await eval_script(r, _DELETE_LUA, len(keys), *keys, message_id)

    async def pending_work(self) -> list[PendingWork]:
        """Every record the DELIVERY machine has unfinished work on — the listing behind
        the boot re-drive and the periodic sweep. Read from the status index, so it costs
        the work outstanding and not the whole retained keyspace. Terminal and intake
        records are not read (an intake record is the turn engine's to resolve); a corrupt
        row is logged and skipped rather than crashing the pass."""
        work: list[PendingWork] = []
        wanted = frozenset({DeliveryStatus.PENDING_DELIVERY, DeliveryStatus.PROVISIONAL})
        now = time.time()
        async with client_ctx(RedisClient, self.settings.redis) as r:
            for message_id in await self._indexed_ids(r, wanted, now):
                hashed = await awaited(r.hgetall(self.settings.record_key(message_id)))
                if not hashed:
                    await self._drop_orphan(r, message_id)
                    continue
                try:
                    # Parse the WHOLE row (content blob included), as list_by_status does: a
                    # row malformed anywhere is skipped here, not handed to a delivery that
                    # re-reads it unguarded and re-drives forever.
                    self._from_hash(hashed)
                    status = DeliveryStatus(hashed[_F_STATUS])
                    if status not in wanted:
                        # Moved on between the index read and this one; its new state is
                        # whoever wrote it to answer for.
                        continue
                    grace = hashed.get(_F_GRACE) or ""
                    found = PendingWork(
                        message_id=message_id,
                        delivery_status=status,
                        attempts=int(hashed[_F_ATTEMPTS]),
                        grace_deadline=float(grace) if grace else None,
                    )
                except (ValueError, KeyError):
                    # One unreadable row must not abort the pass, or every other record
                    # with unfinished work stays stranded behind it forever.
                    logger.warning(
                        "conversations: record %r is corrupt and was skipped in the delivery sweep", message_id
                    )
                    continue
                work.append(found)
        return work

    async def list_by_status(self, statuses: frozenset[DeliveryStatus]) -> list[ConversationRecord]:
        """Every record whose ``delivery_status`` is in ``statuses``, read from the status
        index. An unparseable row is logged and skipped rather than crashing the listing."""
        records: list[ConversationRecord] = []
        wanted = {status.value for status in statuses}
        now = time.time()
        async with client_ctx(RedisClient, self.settings.redis) as r:
            for message_id in await self._indexed_ids(r, statuses, now):
                hashed = await awaited(r.hgetall(self.settings.record_key(message_id)))
                if not hashed:
                    await self._drop_orphan(r, message_id)
                    continue
                if hashed.get(_F_STATUS) not in wanted:
                    continue
                try:
                    records.append(self._from_hash(hashed))
                except (ValueError, KeyError):
                    logger.warning(
                        "conversations: record %r is corrupt and was skipped in the status listing", message_id
                    )
        return records

    async def prune_expired_terminal_indexes(self) -> None:
        """Drop the terminal-status index members whose row has already expired under the
        retention TTL. The live and on-demand indexes are pruned lazily on every read; the
        ``delivered``/``shed`` indexes are read by nothing, so a periodic sweep prunes them
        or they grow without bound. A member's score is its row's exact expiry moment."""
        now = time.time()
        async with client_ctx(RedisClient, self.settings.redis) as r:
            for status in TERMINAL_STATUSES:
                await awaited(r.zremrangebyscore(self.settings.status_index_key(status.value), "-inf", now))

    # -- outbound reverse index (keyspace 3) ---------------------------------

    async def index_outbound(self, channel: str, outbound_ids: list[str], message_id: str) -> None:
        """Map each outbound provider id back to ``message_id`` so an out-of-band receipt
        resolves to the record. Written with the retention TTL, so it is swept with it."""
        if not outbound_ids:
            return
        ttl = self.settings.answer_retention_ttl_seconds
        async with client_ctx(RedisClient, self.settings.redis) as r:
            for outbound_id in outbound_ids:
                await awaited(r.set(self.settings.outbound_index_key(channel, outbound_id), message_id, ex=ttl))

    async def resolve_outbound(self, channel: str, outbound_id: str) -> str | None:
        """The ``message_id`` an outbound provider id maps to, or ``None`` when the index
        holds no such id (an unknown id, or one already swept)."""
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await awaited(r.get(self.settings.outbound_index_key(channel, outbound_id)))
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw


__all__ = ["ConversationRecordStore", "PendingWork"]
