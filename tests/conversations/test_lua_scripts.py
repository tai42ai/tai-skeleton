"""End-to-end execution of the conversation record store's Redis Lua scripts.

Every exactly-once decision the bridge makes is a server-side Lua step. The rest of the
suite drives a Python re-implementation; this module runs the REAL script text against
``fakeredis[lua]``.

The scripts are driven through :class:`ConversationRecordStore`, so the argument
marshalling each one depends on is exercised with them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fakeredis import aioredis
from tai42_contract.conversations import DeliveryReceipt

from tai42_skeleton.conversations import records as records_module
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings

_LEASE = 120.0


@pytest.fixture(autouse=True)
def _conversations_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
async def lua_redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[aioredis.FakeRedis]:
    """A Lua-executing fake Redis wired behind the record store's ``client_ctx`` seam,
    decoding responses exactly as the kit's real client does."""
    client = aioredis.FakeRedis(decode_responses=True)

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield client

    monkeypatch.setattr(records_module, "client_ctx", fake_client_ctx)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def store(lua_redis: aioredis.FakeRedis) -> ConversationRecordStore:
    return ConversationRecordStore(ConversationsSettings())


def _record(message_id: str, *, status: DeliveryStatus = DeliveryStatus.PENDING_DELIVERY) -> ConversationRecord:
    answered = status not in (DeliveryStatus.ACCEPTED, DeliveryStatus.SHED)
    now = time.time()
    return ConversationRecord(
        message_id=message_id,
        route_name="line",
        door="channel",
        thread_id=f"bridge:line:{message_id}",
        client_address="+15550002222",
        channel="twilio",
        our_identity="+15550001111",
        provider_message_id=f"PID-{message_id}",
        delivery_status=status,
        answer_status="answered" if answered else None,
        answer="the answer" if answered else None,
        created_at=now,
        updated_at=now,
    )


def _key(message_id: str) -> str:
    return ConversationsSettings().record_key(message_id)


# -- the inbound claim (conversations:dedupe:claim) ---------------------------


async def test_the_inbound_claim_is_get_or_set(store, lua_redis):
    assert await store.claim_inbound("twilio", "PID1", "first") == "first"
    # A second attempt on the same pair is answered with the id that owns it, and does not
    # overwrite it — the arbitration two concurrent accepts resolve through.
    assert await store.claim_inbound("twilio", "PID1", "second") == "first"
    assert await lua_redis.get(ConversationsSettings().dedupe_key("twilio", "PID1")) == "first"


async def test_the_inbound_claim_carries_the_dedupe_ttl(store, lua_redis):
    await store.claim_inbound("twilio", "PID1", "first")
    ttl = await lua_redis.ttl(ConversationsSettings().dedupe_key("twilio", "PID1"))
    assert 0 < ttl <= ConversationsSettings().inbound_dedupe_ttl_seconds


# -- record creation (conversations:record:create) ----------------------------


async def test_a_non_terminal_record_is_created_without_an_expiry(store, lua_redis):
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    assert await lua_redis.ttl(_key("m1")) == -1
    assert (await store.get_record("m1")).delivery_status is DeliveryStatus.ACCEPTED


async def test_a_record_created_terminal_carries_the_retention_ttl(store, lua_redis):
    await store.create_record(_record("m-shed", status=DeliveryStatus.SHED))
    ttl = await lua_redis.ttl(_key("m-shed"))
    assert 0 < ttl <= ConversationsSettings().answer_retention_ttl_seconds


# -- the guarded turn completion (conversations:record:complete_turn) ---------


async def test_complete_turn_transitions_only_from_intake(store):
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    completed = _record("m1")

    assert await store.complete_turn(completed) == 1
    record = await store.get_record("m1")
    assert record.delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert record.answer == "the answer"
    # The second writer — a turn finishing after a re-drive already resolved the record —
    # is refused rather than overwriting the outcome the client was given.
    assert await store.complete_turn(completed) == 0


async def test_complete_turn_reports_a_missing_record(store):
    assert await store.complete_turn(_record("gone")) == -1


# -- the leased intake claim (conversations:record:intake_claim) ---------------


async def test_an_intake_record_is_created_holding_its_lease_and_no_other_state_is(store, lua_redis):
    await store.create_record(_record("m-intake", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    claim = await lua_redis.hget(_key("m-intake"), "intake_claim")
    assert claim.startswith("worker-1:")
    assert float(claim.split(":", 1)[1]) > time.time()

    await store.create_record(_record("m-pending"))
    assert await lua_redis.hget(_key("m-pending"), "intake_claim") == ""


async def test_creating_a_record_without_the_matching_intake_lease_is_refused(store):
    with pytest.raises(ValueError, match="intake lease"):
        await store.create_record(_record("m-intake", status=DeliveryStatus.ACCEPTED))
    with pytest.raises(ValueError, match="intake lease"):
        await store.create_record(_record("m-pending"), intake_token="worker-1")


async def test_the_intake_claim_refuses_a_live_lease_and_admits_a_lapsed_one(store):
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    now = time.time()

    # A booting sibling cannot adopt a record whose turn is still running...
    assert await store.claim_intake("m1", now, "worker-2", _LEASE) == 0
    # ...while the worker running that turn refreshes its own lease. The refresh must carry
    # the expiry FORWARD, or a turn longer than one lease is reapable the moment it renews.
    assert await store.claim_intake("m1", now + _LEASE - 1, "worker-1", _LEASE) == 1
    assert await store.claim_intake("m1", now + _LEASE + 1, "worker-2", _LEASE) == 0
    # Only once the REFRESHED lease has genuinely lapsed is the record adoptable.
    assert await store.claim_intake("m1", now + 2 * _LEASE, "worker-2", _LEASE) == 1


async def test_the_intake_claim_refuses_a_record_that_has_left_intake(store):
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    assert await store.complete_turn(_record("m1")) == 1
    assert await store.claim_intake("m1", time.time(), "worker-1", _LEASE) == -2


async def test_the_intake_claim_reports_a_missing_record(store):
    assert await store.claim_intake("gone", time.time(), "worker-1", _LEASE) == -1


async def test_completing_a_turn_releases_the_intake_lease(store, lua_redis):
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    assert await store.complete_turn(_record("m1")) == 1
    assert await lua_redis.hget(_key("m1"), "intake_claim") == ""


# -- the leased delivery claim (conversations:record:claim) -------------------


async def test_the_delivery_claim_refuses_a_live_lease_and_admits_a_lapsed_one(store):
    await store.create_record(_record("m1"))
    now = time.time()

    assert await store.claim_delivery("m1", now, "worker-1", _LEASE) == 1
    # Another worker's live lease is not stealable...
    assert await store.claim_delivery("m1", now + 1, "worker-2", _LEASE) == 0
    # ...while the holder refreshes its own lease across a long send.
    assert await store.claim_delivery("m1", now + 1, "worker-1", _LEASE) == 1
    # Once the lease has genuinely lapsed the record is reclaimable by anyone.
    assert await store.claim_delivery("m1", now + 1 + _LEASE + 1, "worker-2", _LEASE) == 1


async def test_the_delivery_claim_refuses_an_intake_record(store):
    await store.create_record(_record("m-intake", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    assert await store.claim_delivery("m-intake", time.time(), "worker-1", _LEASE) == -2


async def test_the_delivery_claim_refuses_a_terminal_record(store):
    await store.create_record(_record("m-shed", status=DeliveryStatus.SHED))
    assert await store.claim_delivery("m-shed", time.time(), "worker-1", _LEASE) == 0


async def test_the_delivery_claim_refuses_a_provisional_record(store):
    """A fully sent record awaits a receipt, not a re-send: only pending_delivery is
    claimable for a send, so a racing re-drive claims nothing."""
    await store.create_record(_record("m-prov"))
    now = time.time()
    assert await store.claim_delivery("m-prov", now, "worker-1", _LEASE) == 1
    assert await store.mark_provisional("m-prov", ["out-1"], 1, now, "worker-1") == 1
    assert await store.claim_delivery("m-prov", now, "worker-2", _LEASE) == 0


async def test_the_delivery_claim_reports_a_missing_record(store):
    assert await store.claim_delivery("gone", time.time(), "worker-1", _LEASE) == -1


# -- provisional + the terminal writes ----------------------------------------


async def test_mark_provisional_records_the_ids_and_releases_the_lease(store, lua_redis):
    await store.create_record(_record("m1"))
    now = time.time()
    assert await store.claim_delivery("m1", now, "worker-1", _LEASE) == 1

    assert await store.mark_provisional("m1", ["out-1", "out-2"], 1, now, "worker-1") == 1
    record = await store.get_record("m1")
    assert record.delivery_status is DeliveryStatus.PROVISIONAL
    assert record.outbound_message_ids == ["out-1", "out-2"]
    assert await lua_redis.hget(_key("m1"), "claim") == ""
    grace = float(await lua_redis.hget(_key("m1"), "grace_deadline"))
    assert grace == pytest.approx(now + ConversationsSettings().delivery_grace_seconds)
    # A provisional record is still non-terminal, so it carries no expiry yet.
    assert await lua_redis.ttl(_key("m1")) == -1


async def test_mark_provisional_refuses_a_terminal_record(store, lua_redis):
    """A receipt or a re-drive can terminalise a record between its last chunk and the
    provisional write; resurrecting it would put a settled answer back in the sweep."""
    await store.create_record(_record("m1"))
    now = time.time()
    assert await store.mark_failed("m1", 1, now, "tok") == 1

    assert await store.mark_provisional("m1", ["out-1"], 1, now, "tok") == 0
    record = await store.get_record("m1")
    assert record.delivery_status is DeliveryStatus.FAILED
    assert record.outbound_message_ids == []
    assert await lua_redis.hget(_key("m1"), "grace_deadline") == ""


async def test_mark_provisional_reports_a_missing_record(store, lua_redis):
    """A record swept out from under the sender must stay gone: writing the fields back
    resurrects a hash with no content blob, which every later read blows up on."""
    assert await store.mark_provisional("gone", ["out-1"], 1, time.time(), "tok") == -1
    assert await lua_redis.exists(_key("gone")) == 0


async def test_the_terminal_writes_are_idempotent_and_refuse_the_opposite_outcome(store, lua_redis):
    await store.create_record(_record("m-ok"))
    now = time.time()

    assert await store.mark_delivered("m-ok", ["out-1"], 1, now, "tok") == 1
    assert await store.mark_delivered("m-ok", ["out-1"], 1, now, "tok") == 0
    assert await store.mark_failed("m-ok", 1, now, "tok") == -2
    assert (await store.get_record("m-ok")).outbound_message_ids == ["out-1"]
    assert 0 < await lua_redis.ttl(_key("m-ok")) <= ConversationsSettings().answer_retention_ttl_seconds

    await store.create_record(_record("m-bad"))
    assert await store.mark_failed("m-bad", 8, now, "tok") == 1
    assert await store.mark_failed("m-bad", 8, now, "tok") == 0
    assert await store.mark_delivered("m-bad", [], 8, now, "tok") == -2
    # BOTH terminal writes carry the retention TTL, or a failed record leaks forever.
    assert 0 < await lua_redis.ttl(_key("m-bad")) <= ConversationsSettings().answer_retention_ttl_seconds

    assert await store.mark_delivered("gone", [], 1, now, "tok") == -1
    assert await store.mark_failed("gone", 1, now, "tok") == -1
    assert await lua_redis.exists(_key("gone")) == 0


@pytest.mark.parametrize("settle", ["provisional", "delivered", "failed", "receipt"])
async def test_a_settled_record_holds_no_lease_and_no_stale_grace_deadline(store, lua_redis, settle):
    """Every write that ends this worker's send releases the lease, and every terminal one
    also drops the grace deadline — leftovers a later pass would act on."""
    await store.create_record(_record("m1"))
    now = time.time()
    assert await store.claim_delivery("m1", now, "worker-1", _LEASE) == 1

    if settle == "delivered":
        assert await store.mark_delivered("m1", ["out-1"], 1, now, "worker-1") == 1
    elif settle == "failed":
        assert await store.mark_failed("m1", 1, now, "worker-1") == 1
    else:
        assert await store.mark_provisional("m1", ["out-1"], 1, now, "worker-1") == 1

    if settle == "receipt":
        assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, now) == 1

    assert await lua_redis.hget(_key("m1"), "claim") == ""
    if settle != "provisional":
        assert await lua_redis.hget(_key("m1"), "grace_deadline") == ""


@pytest.mark.parametrize("write", ["provisional", "delivered", "failed"])
async def test_a_delivery_write_is_refused_under_another_workers_live_lease(store, write):
    """A worker whose lease lapsed and was taken over must not be able to write ANY
    delivery state: its send is no longer the one the record is committed to."""
    await store.create_record(_record("m1"))
    now = time.time()
    assert await store.claim_delivery("m1", now, "worker-2", _LEASE) == 1

    if write == "provisional":
        assert await store.mark_provisional("m1", ["out-1"], 1, now, "worker-1") == -3
    elif write == "delivered":
        assert await store.mark_delivered("m1", ["out-1"], 1, now, "worker-1") == -3
    else:
        assert await store.mark_failed("m1", 1, now, "worker-1") == -3
    assert (await store.get_record("m1")).delivery_status is DeliveryStatus.PENDING_DELIVERY

    # The holder's own write goes through, and a lapsed foreign lease no longer blocks.
    assert await store.mark_provisional("m1", ["out-1"], 1, now, "worker-2") == 1


async def test_a_late_failed_write_cannot_undo_a_fully_sent_answer(store):
    """Once the answer is out the record is ``provisional``; the send path may no longer
    terminalise it ``failed``, or a hung worker's late refusal erases a delivered answer."""
    await store.create_record(_record("m1"))
    now = time.time()
    assert await store.mark_provisional("m1", ["out-1"], 1, now, "worker-2") == 1

    # The claim is released by the provisional write, so only the state guard stands here.
    assert await store.mark_failed("m1", 1, now, "worker-1") == -2
    assert (await store.get_record("m1")).delivery_status is DeliveryStatus.PROVISIONAL


# -- out-of-band receipt ingestion (conversations:record:receipt) -------------


async def test_a_receipt_confirms_a_provisional_record_once(store, lua_redis):
    await store.create_record(_record("m1"))
    now = time.time()
    await store.mark_provisional("m1", ["out-1"], 1, now, "tok")

    assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, now) == 1
    assert (await store.get_record("m1")).delivery_status is DeliveryStatus.DELIVERED
    # The grace-expiry fallback arriving after a receipt is a no-op, not a second write.
    assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, now) == 0
    # A failure receipt for a record already confirmed delivered is the conflict the
    # caller logs, never a silent overwrite.
    assert await store.ingest_receipt("m1", DeliveryReceipt.FAILED, now) == -2
    assert 0 < await lua_redis.ttl(_key("m1")) <= ConversationsSettings().answer_retention_ttl_seconds


async def test_a_receipt_reports_a_missing_record(store):
    assert await store.ingest_receipt("gone", DeliveryReceipt.DELIVERED, time.time()) == -1


@pytest.mark.parametrize("receipt", [DeliveryReceipt.DELIVERED, DeliveryReceipt.FAILED])
@pytest.mark.parametrize("status", [DeliveryStatus.ACCEPTED, DeliveryStatus.PENDING_DELIVERY])
async def test_a_receipt_before_the_send_finished_is_refused(store, lua_redis, status, receipt):
    """Chunk one's provider callback routinely lands while chunks two and three are still
    going out. Settling the record on it would terminalise a half-sent answer."""
    intake = "worker-1" if status is DeliveryStatus.ACCEPTED else None
    await store.create_record(_record("m1", status=status), intake_token=intake)
    now = time.time()

    assert await store.ingest_receipt("m1", receipt, now) == -3
    assert (await store.get_record("m1")).delivery_status is status
    # No terminal write means no retention TTL: the record is still live work.
    assert await lua_redis.ttl(_key("m1")) == -1


# -- the scans that drive the re-drive and the sweep --------------------------


async def test_the_scans_report_unfinished_work_and_skip_a_corrupt_row(store, lua_redis):
    await store.create_record(_record("m-pending"))
    await store.create_record(_record("m-intake", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    await store.create_record(_record("m-shed", status=DeliveryStatus.SHED))
    await store.create_record(_record("m-prov"))
    await store.mark_provisional("m-prov", ["out-1"], 1, time.time(), "tok")
    # A row corrupted under a live index entry — the listing reads the index, so this is
    # the shape a corrupt row reaches it in.
    await lua_redis.hset(_key("m-corrupt"), mapping={"delivery_status": "not-a-status", "attempts": "x"})
    await lua_redis.zadd(
        ConversationsSettings().status_index_key(DeliveryStatus.PENDING_DELIVERY.value), {"m-corrupt": float("inf")}
    )

    work = {item.message_id: item for item in await store.pending_work()}

    # The corrupt row is skipped and every other unfinished record still comes back.
    assert set(work) == {"m-pending", "m-prov"}
    assert work["m-prov"].grace_deadline is not None
    # The intake scan sees exactly the record whose turn never completed.
    intake = await store.list_by_status(frozenset({DeliveryStatus.ACCEPTED}))
    assert [record.message_id for record in intake] == ["m-intake"]


# -- the per-status index those listings read ---------------------------------


async def _indexed_under(lua_redis, message_id: str) -> list[str]:
    """Every status index naming ``message_id`` — exactly one, its current status."""
    settings = ConversationsSettings()
    return [
        status.value
        for status in DeliveryStatus
        if await lua_redis.zscore(settings.status_index_key(status.value), message_id) is not None
    ]


async def test_every_transition_moves_the_record_to_exactly_one_status_index(store, lua_redis):
    """The listings read the index rather than the keyspace, so a record left indexed under
    a status it has LEFT is picked up twice, and one indexed under none is invisible work."""
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")
    assert await _indexed_under(lua_redis, "m1") == ["accepted"]

    assert await store.complete_turn(_record("m1")) == 1
    assert await _indexed_under(lua_redis, "m1") == ["pending_delivery"]
    # A live record's member outlives nothing: it is never swept out from under the listing.
    assert await lua_redis.zscore(ConversationsSettings().status_index_key("pending_delivery"), "m1") == float("inf")

    now = time.time()
    assert await store.mark_provisional("m1", ["out-1"], 1, now, "tok") == 1
    assert await _indexed_under(lua_redis, "m1") == ["provisional"]

    assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, now) == 1
    assert await _indexed_under(lua_redis, "m1") == ["delivered"]


@pytest.mark.parametrize("terminal", ["delivered", "failed"])
async def test_a_terminal_write_indexes_the_member_to_expire_with_its_row(store, lua_redis, terminal):
    await store.create_record(_record("m1"))
    now = time.time()
    if terminal == "delivered":
        assert await store.mark_delivered("m1", ["out-1"], 1, now, "tok") == 1
    else:
        assert await store.mark_failed("m1", 1, now, "tok") == 1

    assert await _indexed_under(lua_redis, "m1") == [terminal]
    # The member's score is the moment its row expires, so the index is swept with it and
    # never outlives what it names.
    score = await lua_redis.zscore(ConversationsSettings().status_index_key(terminal), "m1")
    assert score == pytest.approx(now + ConversationsSettings().answer_retention_ttl_seconds)
    assert await store.pending_work() == []


async def test_deleting_a_record_unindexes_it_in_the_same_step(store, lua_redis):
    """An index entry outliving its row hands every later listing a ``message_id`` with
    nothing behind it."""
    await store.create_record(_record("m1", status=DeliveryStatus.ACCEPTED), intake_token="worker-1")

    assert await store.delete_record("m1") is True
    assert await lua_redis.exists(_key("m1")) == 0
    assert await _indexed_under(lua_redis, "m1") == []
    assert await store.list_by_status(frozenset({DeliveryStatus.ACCEPTED})) == []
    # A second delete removes nothing, and says so.
    assert await store.delete_record("m1") is False
