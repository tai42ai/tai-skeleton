"""The answer/record store — inbound dedupe, the exactly-once delivery claim, the
delivery-state transitions, out-of-band receipt ingestion, the reverse index and the
re-drive scan — against the faked redis hash + string + Lua seam."""

from __future__ import annotations

import time

import pytest
from tai42_contract.conversations import DeliveryReceipt

from tai42_skeleton.conversations import records as records_module
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings

from .fake_record_redis import FakeRecordRedis, make_record_client_ctx


@pytest.fixture(autouse=True)
def _redis_backend(monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")


def _store(monkeypatch, fake: FakeRecordRedis) -> ConversationRecordStore:
    monkeypatch.setattr(records_module, "client_ctx", make_record_client_ctx(fake))
    return ConversationRecordStore(ConversationsSettings())


def _record(message_id: str = "m1", door: str = "channel", **over) -> ConversationRecord:
    now = time.time()
    fields = {
        "message_id": message_id,
        "route_name": "line",
        "door": door,
        "thread_id": f"bridge:line:{message_id}",
        "client_address": "+15550002222",
        "channel": "twilio" if door == "channel" else None,
        "our_identity": "+15550001111" if door == "channel" else None,
        "callback_url": "https://cb.example/x" if door == "api" else None,
        "caller_principal": "alice" if door == "api" else None,
        "answer_status": "answered",
        "answer": "hello there",
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    fields.update(over)
    return ConversationRecord(**fields)  # type: ignore[arg-type]


def _intake(message_id: str = "m1", **over) -> ConversationRecord:
    """A pre-turn intake record — the shape ``accept`` persists before it claims."""
    return _record(
        message_id,
        delivery_status=DeliveryStatus.ACCEPTED,
        answer_status=None,
        answer=None,
        provider_message_id="PID1",
        **over,
    )


def test_in_memory_backend_refuses(monkeypatch):
    monkeypatch.delenv("CONVERSATIONS_REDIS_URL", raising=False)
    from tai42_skeleton.operations.errors import NotSupportedError

    with pytest.raises(NotSupportedError):
        ConversationRecordStore(ConversationsSettings())


async def test_claim_inbound_is_idempotent(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    first = await store.claim_inbound("twilio", "PID1", "msg-a")
    assert first == "msg-a"  # fresh claim keeps the caller's id
    # A provider redelivery of the same (channel, provider_message_id) returns the FIRST id.
    again = await store.claim_inbound("twilio", "PID1", "msg-b")
    assert again == "msg-a"
    # A different provider id on the same channel is independent.
    other = await store.claim_inbound("twilio", "PID2", "msg-c")
    assert other == "msg-c"


async def test_get_inbound_owner_reads_without_claiming(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    # The fast-path read takes nothing: an unclaimed pair stays unclaimed, so a later
    # claim by a real accept still wins it.
    assert await store.get_inbound_owner("twilio", "PID1") is None
    assert await store.get_inbound_owner("twilio", "PID1") is None
    assert await store.claim_inbound("twilio", "PID1", "msg-a") == "msg-a"
    assert await store.get_inbound_owner("twilio", "PID1") == "msg-a"


async def test_complete_turn_is_guarded_on_the_intake_state(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_intake("m1"), intake_token="worker-1")
    completed = _record("m1", answer="the answer")

    assert await store.complete_turn(completed) == 1
    got = await store.get_record("m1")
    assert got is not None
    assert got.delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert got.answer == "the answer"
    assert got.answer_status == "answered"
    # A second writer (a late turn racing the re-drive that already resolved the record)
    # is refused rather than overwriting the outcome the client was given.
    assert await store.complete_turn(_record("m1", answer="a different answer")) == 0
    got = await store.get_record("m1")
    assert got is not None
    assert got.answer == "the answer"
    # A record that no longer exists answers -1.
    assert await store.complete_turn(_record("gone", answer="x")) == -1


async def test_complete_turn_refuses_a_record_that_is_not_pending_delivery(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    with pytest.raises(ValueError, match="pending_delivery"):
        await store.complete_turn(_intake("m1"))


async def test_claim_delivery_refuses_an_intake_record(monkeypatch):
    # An intake record carries no answer, so the delivery machine must never take it.
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_intake("m1"), intake_token="worker-1")
    assert await store.claim_delivery("m1", time.time(), "tok", 120) == -2


async def test_delete_record_removes_an_abandoned_intake_record(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_intake("m1"), intake_token="worker-1")
    assert await store.delete_record("m1") is True
    assert await store.get_record("m1") is None
    assert await store.delete_record("m1") is False


async def test_a_shed_record_is_created_terminal_with_the_retention_ttl(monkeypatch):
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    shed = _record(
        "m1",
        delivery_status=DeliveryStatus.SHED,
        answer_status=None,
        answer=None,
        error="over the rate cap",
    )
    await store.create_record(shed)

    got = await store.get_record("m1")
    assert got is not None
    assert got.delivery_status is DeliveryStatus.SHED
    assert got.answer is None
    # It is terminal at birth: it carries the retention TTL, and no transition moves it.
    assert fake.ttl_ms[settings.record_key("m1")] == settings.answer_retention_ttl_seconds * 1000
    assert await store.claim_delivery("m1", time.time(), "tok", 120) == 0
    assert await store.mark_delivered("m1", [], 1, time.time(), "tok") == -2
    assert await store.mark_failed("m1", 1, time.time(), "tok") == -2


async def test_an_intake_record_never_expires(monkeypatch):
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    await store.create_record(_intake("m1"), intake_token="worker-1")
    assert ConversationsSettings().record_key("m1") not in fake.ttl_ms


async def test_create_and_get_round_trip(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    record = _record(answer="hi", outbound_message_ids=[], attempts=0)
    await store.create_record(record)
    got = await store.get_record("m1")
    assert got is not None
    assert got.answer == "hi"
    assert got.delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert got.door == "channel"
    assert got.our_identity == "+15550001111"
    assert await store.get_record("nope") is None


async def test_claim_delivery_is_exactly_once(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    now = time.time()
    # First worker wins the lease.
    assert await store.claim_delivery("m1", now, "tokA", 120) == 1
    # A second, different worker (a boot re-drive) sees the live lease and is refused.
    assert await store.claim_delivery("m1", now, "tokB", 120) == 0
    # The holder may re-claim (refresh) its own lease.
    assert await store.claim_delivery("m1", now + 1, "tokA", 120) == 1
    # A missing record answers -1.
    assert await store.claim_delivery("gone", now, "tokA", 120) == -1


async def test_claim_delivery_skips_terminal_record(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    await store.mark_delivered("m1", [], 1, time.time(), "tok")
    assert await store.claim_delivery("m1", time.time(), "tok", 120) == 0


async def test_channel_provisional_then_receipt_delivered(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    assert await store.mark_provisional("m1", ["out-1"], 1, time.time(), "tok") == 1
    got = await store.get_record("m1")
    assert got is not None
    assert got.delivery_status is DeliveryStatus.PROVISIONAL
    assert got.outbound_message_ids == ["out-1"]
    # A positive receipt confirms it delivered.
    assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, time.time()) == 1
    got = await store.get_record("m1")
    assert got is not None
    assert got.delivery_status is DeliveryStatus.DELIVERED
    # A repeat receipt is idempotent.
    assert await store.ingest_receipt("m1", DeliveryReceipt.DELIVERED, time.time()) == 0


async def test_receipt_failed_marks_failed(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    await store.mark_provisional("m1", ["out-1"], 1, time.time(), "tok")
    assert await store.ingest_receipt("m1", DeliveryReceipt.FAILED, time.time()) == 1
    got = await store.get_record("m1")
    assert got is not None
    assert got.delivery_status is DeliveryStatus.FAILED


async def test_receipt_conflicts_with_opposite_terminal(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    await store.mark_delivered("m1", [], 1, time.time(), "tok")
    # A late FAILED receipt on an already-delivered record is a conflict, not an override.
    assert await store.ingest_receipt("m1", DeliveryReceipt.FAILED, time.time()) == -2


async def test_mark_delivered_idempotent_and_failed_guard(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    assert await store.mark_delivered("m1", [], 1, time.time(), "tok") == 1
    assert await store.mark_delivered("m1", [], 1, time.time(), "tok") == 0  # already delivered
    assert await store.mark_failed("m1", 1, time.time(), "tok") == -2  # cannot fail a delivered record


async def test_bump_attempt(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record())
    assert await store.bump_attempt("m1") == 1
    assert await store.bump_attempt("m1") == 2


async def test_outbound_reverse_index(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.index_outbound("twilio", ["o-1", "o-2"], "m1")
    assert await store.resolve_outbound("twilio", "o-1") == "m1"
    assert await store.resolve_outbound("twilio", "o-2") == "m1"
    assert await store.resolve_outbound("twilio", "unknown") is None


async def test_pending_work_reports_non_terminal_only(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record("pend"))
    await store.create_record(_record("prov"))
    await store.mark_provisional("prov", ["o"], 1, 1000.0, "tok")
    await store.create_record(_record("done"))
    await store.mark_delivered("done", [], 1, time.time(), "tok")

    work = {w.message_id: w for w in await store.pending_work()}
    assert set(work) == {"pend", "prov"}
    assert work["pend"].delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert work["prov"].delivery_status is DeliveryStatus.PROVISIONAL
    assert work["prov"].grace_deadline is not None


async def test_pending_work_skips_corrupt_rows_and_still_reports_the_rest(monkeypatch):
    # One unreadable row must not abort the pass, or every record behind it is stranded
    # forever.
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    await store.create_record(_record("good"))
    fake.seed_hash(
        settings.record_key("unknown-status"),
        {"data": "{}", "delivery_status": "in_flight", "outbound_ids": "[]", "attempts": "0", "updated_at": "1"},
    )
    fake.seed_hash(
        settings.record_key("non-numeric-attempts"),
        {"data": "{}", "delivery_status": "pending_delivery", "outbound_ids": "[]", "attempts": "?", "updated_at": "1"},
    )
    fake.seed_hash(settings.record_key("foreign-row"), {"something": "else"})
    fake.seed_hash(
        settings.record_key("bad-grace"),
        {
            "data": "{}",
            "delivery_status": "provisional",
            "outbound_ids": "[]",
            "attempts": "1",
            "grace_deadline": "soon",
            "updated_at": "1",
        },
    )
    # The pass reads the status index, so a corrupt row only reaches it while indexed.
    for message_id, indexed_as in (
        ("unknown-status", "pending_delivery"),
        ("non-numeric-attempts", "pending_delivery"),
        ("foreign-row", "pending_delivery"),
        ("bad-grace", "provisional"),
    ):
        await fake.zadd(settings.status_index_key(indexed_as), {message_id: float("inf")})

    work = await store.pending_work()
    assert [w.message_id for w in work] == ["good"]


async def test_list_by_status_failed(monkeypatch):
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record("a"))
    await store.mark_failed("a", 3, time.time(), "tok")
    await store.create_record(_record("b"))
    await store.mark_delivered("b", [], 1, time.time(), "tok")

    failed = await store.list_by_status(frozenset({DeliveryStatus.FAILED}))
    assert [r.message_id for r in failed] == ["a"]


async def test_retention_ttl_applied_only_on_a_terminal_transition(monkeypatch):
    # A record persisted before send carries NO expiry (it must survive until delivered
    # or failed); the retention TTL is applied exactly on the terminal write.
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    key = settings.record_key("m1")
    await store.create_record(_record("m1"))
    assert key not in fake.ttl_ms  # pending_delivery never expires

    await store.mark_provisional("m1", ["o"], 1, time.time(), "tok")
    assert key not in fake.ttl_ms  # provisional still does not expire

    await store.mark_delivered("m1", ["o"], 1, time.time(), "tok")
    assert fake.ttl_ms[key] == settings.answer_retention_ttl_seconds * 1000


async def test_retention_ttl_applied_on_a_failed_transition(monkeypatch):
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    key = settings.record_key("m1")
    await store.create_record(_record("m1"))
    await store.mark_failed("m1", 8, time.time(), "tok")
    assert fake.ttl_ms[key] == settings.answer_retention_ttl_seconds * 1000


async def test_message_id_is_a_uuid4(monkeypatch):
    # A record round-trips whatever id it was created with; the doors mint uuid4.
    from uuid import UUID, uuid4

    minted = str(uuid4())
    assert UUID(minted).version == 4
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record(minted))
    got = await store.get_record(minted)
    assert got is not None
    assert got.message_id == minted


# -- the per-status index the listings read -----------------------------------


async def test_a_transition_moves_the_record_between_status_indexes(monkeypatch):
    # Exactly one index names a record, so a listing costs the work outstanding and never
    # walks the retained keyspace.
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    pending = settings.status_index_key(DeliveryStatus.PENDING_DELIVERY.value)
    failed = settings.status_index_key(DeliveryStatus.FAILED.value)
    await store.create_record(_record("m1"))
    assert await fake.zrange(pending, 0, -1) == ["m1"]

    await store.mark_failed("m1", 1, time.time(), "tok")
    assert await fake.zrange(pending, 0, -1) == []
    assert await fake.zrange(failed, 0, -1) == ["m1"]
    # A terminal member is scored to expire with the row it names.
    assert fake._zsets[failed]["m1"] <= time.time() + settings.answer_retention_ttl_seconds
    assert await store.pending_work() == []


async def test_a_terminal_index_member_expires_with_its_row(monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_ANSWER_RETENTION_TTL_SECONDS", "1")
    store = _store(monkeypatch, FakeRecordRedis())
    await store.create_record(_record("old"))
    # Terminal a full retention window ago: the index must not name it any more.
    await store.mark_failed("old", 1, time.time() - 10, "tok")

    assert await store.list_by_status(frozenset({DeliveryStatus.FAILED})) == []


async def test_a_member_whose_row_is_gone_is_unindexed(monkeypatch):
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    await store.create_record(_record("gone"))
    # The row removed from under the index rather than through ``delete_record``.
    fake._hashes.pop(settings.record_key("gone"))

    assert await store.pending_work() == []
    assert await fake.zrange(settings.status_index_key(DeliveryStatus.PENDING_DELIVERY.value), 0, -1) == []


async def test_delete_record_unindexes_the_row_it_removes(monkeypatch):
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    await store.create_record(_record("m1"))

    assert await store.delete_record("m1") is True
    assert await store.pending_work() == []
    assert (
        await fake.zrange(ConversationsSettings().status_index_key(DeliveryStatus.PENDING_DELIVERY.value), 0, -1) == []
    )


async def test_pending_work_skips_a_row_whose_content_blob_is_corrupt(monkeypatch):
    # Control fields fine, ``data`` unparseable: the whole-row parse must skip it here, not
    # hand it to a delivery that re-reads it unguarded and re-drives it every lease forever.
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    await store.create_record(_record("good"))
    fake.seed_hash(
        settings.record_key("bad-data"),
        {
            "data": "{not json",
            "delivery_status": "pending_delivery",
            "outbound_ids": "[]",
            "attempts": "0",
            "grace_deadline": "",
            "updated_at": "1",
        },
    )
    await fake.zadd(settings.status_index_key("pending_delivery"), {"bad-data": float("inf")})

    work = await store.pending_work()
    assert [w.message_id for w in work] == ["good"]


async def test_prune_expired_terminal_indexes_drops_only_expired_members(monkeypatch):
    # The delivered/shed indexes are read by no listing, so a periodic prune must drop the
    # members whose row has expired or they outgrow the retained keyspace they name.
    fake = FakeRecordRedis()
    store = _store(monkeypatch, fake)
    settings = ConversationsSettings()
    now = time.time()
    for status in (DeliveryStatus.DELIVERED, DeliveryStatus.SHED, DeliveryStatus.FAILED):
        await fake.zadd(
            settings.status_index_key(status.value),
            {f"{status.value}-expired": now - 10, f"{status.value}-live": now + 100_000},
        )

    await store.prune_expired_terminal_indexes()

    for status in (DeliveryStatus.DELIVERED, DeliveryStatus.SHED, DeliveryStatus.FAILED):
        assert await fake.zrange(settings.status_index_key(status.value), 0, -1) == [f"{status.value}-live"]
