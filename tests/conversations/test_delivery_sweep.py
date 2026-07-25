"""The delivery executor's recovery halves: the periodic stalled-delivery sweep, and the
resumable multi-chunk channel send that keeps a re-drive from re-sending what a provider
already accepted.

The sweep decides WHETHER a record is picked up again (only once the dead worker's lease
has lapsed, never while another's is live); the ledger decides WHERE the pick-up resumes
(the first character the provider has not taken).

A crashed worker is a channel raising something the executor does NOT handle, so the send
is abandoned exactly where the process would have died.

The api door's half runs the REAL signed callback against a local httpx transport, so the
request the executor builds is the one asserted."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import time

import httpx
import pytest
from tai42_contract.channels import ChannelDeliveryError
from tai42_contract.conversations import ConversationRoute, DeliveryReceipt

from tai42_skeleton.conversations import delivery as delivery_module
from tai42_skeleton.conversations import ledger as ledger_module
from tai42_skeleton.conversations import records as records_module
from tai42_skeleton.conversations import turn as turn_module
from tai42_skeleton.conversations.ledger import ChannelSendLedger
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings

from .fake_record_redis import FakeRecordRedis, make_record_client_ctx

#: Every answer in this module is chunked at this width, so a handful of characters makes
#: a genuinely multi-chunk send.
_CHUNK_CHARS = 10


class WorkerDied(RuntimeError):
    """What a channel raises to stand in for the worker vanishing mid-send — not a
    ``ChannelDeliveryError``, so the executor does not turn it into a ``failed`` record."""


class FakeChannel:
    """Records every chunk it is asked to send. ``crash_on`` abandons the send on the nth
    chunk (a dead worker), ``fail_on`` refuses it the way a provider does, and ``hang_on``
    never returns from it (a send still in flight)."""

    def __init__(
        self,
        prefix: str = "out",
        *,
        crash_on: int | None = None,
        fail_on: int | None = None,
        hang_on: int | None = None,
        watch=None,
    ) -> None:
        self.sends: list[str] = []
        self._prefix = prefix
        self._crash_on = crash_on
        self._fail_on = fail_on
        self._hang_on = hang_on
        self._watch = watch

    async def notify(self, notification) -> list[str]:
        self.sends.append(notification.message)
        if self._watch is not None:
            watched = self._watch()
            if inspect.isawaitable(watched):
                await watched
        if self._crash_on is not None and len(self.sends) == self._crash_on:
            raise WorkerDied("the worker died mid-send")
        if self._fail_on is not None and len(self.sends) == self._fail_on:
            raise ChannelDeliveryError("the provider refused the chunk")
        if self._hang_on is not None and len(self.sends) == self._hang_on:
            await asyncio.Event().wait()
        return [f"{self._prefix}-{len(self.sends)}"]


class _FakeChannels:
    def __init__(self, channel: FakeChannel) -> None:
        self._channel = channel

    def get(self, name: str) -> FakeChannel:
        return self._channel


class _FakeDeliveryApp:
    def __init__(self, channel: FakeChannel) -> None:
        self.channels = _FakeChannels(channel)


@pytest.fixture(autouse=True)
def _conversations_env(monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_CLAIM_LEASE_SECONDS", "120")
    monkeypatch.setenv("CONVERSATIONS_MAX_MESSAGE_CHARS", f'{{"twilio": {_CHUNK_CHARS}}}')


@pytest.fixture(autouse=True)
def _no_grace_timers(monkeypatch):
    """The hour-long fallback confirmation a completed channel send schedules is a
    different mechanism; stubbing it keeps these tests to the send itself."""

    async def _noop(message_id: str, grace_seconds: float) -> None:
        return None

    monkeypatch.setattr(delivery_module, "_confirm_after_grace", _noop)


@pytest.fixture(autouse=True)
def _drop_leftover_tasks():
    yield
    for task in list(delivery_module._DELIVERY_TASKS):
        task.cancel()
    delivery_module._DELIVERY_TASKS.clear()


@pytest.fixture
def fake(monkeypatch) -> FakeRecordRedis:
    """One faked redis behind both the record store and the send ledger, so a test reads
    the same keyspace the executor writes through."""
    backing = FakeRecordRedis()
    monkeypatch.setattr(records_module, "client_ctx", make_record_client_ctx(backing))
    monkeypatch.setattr(ledger_module, "client_ctx", make_record_client_ctx(backing))
    return backing


@pytest.fixture
def store(fake: FakeRecordRedis) -> ConversationRecordStore:
    return ConversationRecordStore(ConversationsSettings())


def _record(message_id: str, answer: str) -> ConversationRecord:
    now = time.time()
    return ConversationRecord(
        message_id=message_id,
        route_name="line",
        door="channel",
        thread_id=f"bridge:line:{message_id}",
        client_address="+15550002222",
        channel="twilio",
        our_identity="+15550001111",
        answer_status="answered",
        answer=answer,
        created_at=now,
        updated_at=now,
    )


def _wire_channel(monkeypatch, channel: FakeChannel) -> None:
    monkeypatch.setattr(delivery_module, "tai42_app", _FakeDeliveryApp(channel))


def _claim(fake: FakeRecordRedis, message_id: str) -> tuple[str, float]:
    """The record's live lease as ``(token, expiry)``; an empty token when it is free."""
    raw = fake._hashes[ConversationsSettings().record_key(message_id)]["claim"]
    if not raw:
        return "", 0.0
    token, expiry = raw.split(":", 1)
    return token, float(expiry)


def _expire_claim(fake: FakeRecordRedis, message_id: str) -> None:
    """Age the record's lease out — what the passage of time does to a dead worker's."""
    key = ConversationsSettings().record_key(message_id)
    token = fake._hashes[key]["claim"].split(":", 1)[0]
    fake._hashes[key]["claim"] = f"{token}:{time.time() - 1}"


async def _drain_spawned(store: ConversationRecordStore) -> None:
    """Await every delivery the sweep spawned."""
    pending = [task for task in delivery_module._DELIVERY_TASKS if not task.done()]
    if pending:
        await asyncio.gather(*pending)


async def _get(store: ConversationRecordStore, message_id: str) -> ConversationRecord:
    record = await store.get_record(message_id)
    assert record is not None
    return record


# -- the periodic sweep reclaims what a dead worker left -----------------------


async def test_sweep_redrives_a_record_whose_lease_has_lapsed(monkeypatch, fake, store):
    """A worker took the lease and died before sending anything. Once that lease lapses
    the sweep re-drives the record and the answer finally goes out — the boot re-drive
    alone would have found the lease still live and given up for the life of the process.
    """
    channel = FakeChannel()
    _wire_channel(monkeypatch, channel)
    await store.create_record(_record("m-dead", "the answer"))
    # The dead worker's lease, taken 200s ago under a 120s lease: long lapsed.
    assert await store.claim_delivery("m-dead", time.time() - 200, "dead-worker", 120) == 1

    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert channel.sends == ["the answer"]
    assert (await _get(store, "m-dead")).delivery_status is DeliveryStatus.PROVISIONAL


async def test_sweep_leaves_another_workers_live_lease_alone(monkeypatch, fake, store):
    """Under the supported multi-worker deployment a lease found pending may be somebody
    else's send in flight. The sweep must not steal it."""
    channel = FakeChannel()
    _wire_channel(monkeypatch, channel)
    await store.create_record(_record("m-live", "the answer"))
    assert await store.claim_delivery("m-live", time.time(), "other-worker", 120) == 1

    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert channel.sends == []
    assert (await _get(store, "m-live")).delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert _claim(fake, "m-live")[0] == "other-worker"


async def test_sweep_confirms_a_provisional_record_past_its_grace(monkeypatch, fake, store):
    """The worker that scheduled the fallback confirmation died holding it, so nothing in
    the process would ever close the record. The sweep does."""
    _wire_channel(monkeypatch, FakeChannel())
    await store.create_record(_record("m-prov", "the answer"))
    await store.mark_provisional("m-prov", ["out-1"], 1, time.time(), "tok")
    fake._hashes[ConversationsSettings().record_key("m-prov")]["grace_deadline"] = str(time.time() - 1)

    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert (await _get(store, "m-prov")).delivery_status is DeliveryStatus.DELIVERED


async def test_sweep_does_not_confirm_a_provisional_record_still_within_grace(monkeypatch, fake, store):
    _wire_channel(monkeypatch, FakeChannel())
    await store.create_record(_record("m-young", "the answer"))
    await store.mark_provisional("m-young", ["out-1"], 1, time.time(), "tok")

    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert (await _get(store, "m-young")).delivery_status is DeliveryStatus.PROVISIONAL


async def test_sweep_fails_a_record_that_has_spent_every_attempt(monkeypatch, fake, store):
    """A record no re-drive can finish must not be swept forever: once its attempts are
    spent it becomes a loud, retained ``failed`` instead."""
    channel = FakeChannel()
    _wire_channel(monkeypatch, channel)
    await store.create_record(_record("m-spent", "the answer"))
    for _ in range(ConversationsSettings().delivery_max_attempts):
        await store.bump_attempt("m-spent")

    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert channel.sends == []
    assert (await _get(store, "m-spent")).delivery_status is DeliveryStatus.FAILED


# -- a partial multi-chunk send resumes instead of re-sending ------------------


async def test_a_partial_send_resumes_at_the_first_unsent_chunk(monkeypatch, fake, store):
    """The duplicate this exists to prevent: a send whose worker died after two of four
    chunks. The re-drive must send chunks three and four ONLY — a human must never be
    texted chunk one or two a second time."""
    answer = "aaaaaaaaaabbbbbbbbbbccccccccccdddddddddd"
    await store.create_record(_record("m-part", answer))

    dying = FakeChannel("w1", crash_on=3)
    _wire_channel(monkeypatch, dying)
    assert await store.claim_delivery("m-part", time.time(), "worker-1", 120) == 1
    with pytest.raises(WorkerDied):
        await delivery_module._deliver_channel(store, await _get(store, "m-part"), "worker-1")
    assert dying.sends == ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
    assert (await _get(store, "m-part")).delivery_status is DeliveryStatus.PENDING_DELIVERY

    # The dead worker's lease lapses, and the sweep's re-drive picks the record up.
    _expire_claim(fake, "m-part")
    resuming = FakeChannel("w2")
    _wire_channel(monkeypatch, resuming)
    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert resuming.sends == ["cccccccccc", "dddddddddd"]
    record = await _get(store, "m-part")
    assert record.delivery_status is DeliveryStatus.PROVISIONAL
    # The record names every id both workers produced, in send order.
    assert record.outbound_message_ids == ["w1-1", "w1-2", "w2-1", "w2-2"]


async def test_a_resumed_send_reindexes_what_the_ledger_already_knows(monkeypatch, fake, store):
    """A receipt naming a chunk accepted just before the crash must still resolve to its
    record, even if the crash landed between the ledger append and the reverse index."""
    answer = "aaaaaaaaaabbbbbbbbbb"
    await store.create_record(_record("m-index", answer))
    await ChannelSendLedger(ConversationsSettings()).append("m-index", _CHUNK_CHARS, ["w1-1"])
    assert await store.resolve_outbound("twilio", "w1-1") is None

    _wire_channel(monkeypatch, FakeChannel("w2"))
    await delivery_module._deliver_channel(store, await _get(store, "m-index"), "worker-2")

    assert await store.resolve_outbound("twilio", "w1-1") == "m-index"


async def test_a_send_interrupted_after_its_last_chunk_re_sends_nothing(monkeypatch, fake, store):
    """The whole answer was already out and only the record write was lost. The re-drive
    sends nothing at all and just closes the record."""
    answer = "aaaaaaaaaabbbbbbbbbb"
    await store.create_record(_record("m-tail", answer))
    ledger = ChannelSendLedger(ConversationsSettings())
    await ledger.append("m-tail", _CHUNK_CHARS, ["w1-1"])
    await ledger.append("m-tail", _CHUNK_CHARS, ["w1-2"])

    channel = FakeChannel("w2")
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-tail"), "worker-2")

    assert channel.sends == []
    record = await _get(store, "m-tail")
    assert record.delivery_status is DeliveryStatus.PROVISIONAL
    assert record.outbound_message_ids == ["w1-1", "w1-2"]


async def test_a_completed_send_leaves_no_ledger_behind(monkeypatch, fake, store):
    await store.create_record(_record("m-clean", "aaaaaaaaaabbbbbbbbbb"))
    _wire_channel(monkeypatch, FakeChannel())
    await delivery_module._deliver_channel(store, await _get(store, "m-clean"), "worker-1")

    assert await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-clean") == []


async def test_a_provider_refusal_is_terminal_and_clears_the_ledger(monkeypatch, fake, store):
    """A refusal is not a crash: the send has no idempotency key to retry under, so the
    record fails loudly and keeps the ids of the chunks the provider did take."""
    await store.create_record(_record("m-refused", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    _wire_channel(monkeypatch, FakeChannel("w1", fail_on=2))
    await delivery_module._deliver_channel(store, await _get(store, "m-refused"), "worker-1")

    assert (await _get(store, "m-refused")).delivery_status is DeliveryStatus.FAILED
    assert await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-refused") == []
    assert await store.resolve_outbound("twilio", "w1-1") == "m-refused"


async def test_a_ledger_claiming_more_than_the_answer_refuses_loudly(monkeypatch, fake, store):
    """A ledger that cannot describe this answer is corrupt state, not a resume point —
    resuming from it would send the wrong text."""
    await store.create_record(_record("m-bad", "short"))
    await ChannelSendLedger(ConversationsSettings()).append("m-bad", 99, ["w1-1"])
    _wire_channel(monkeypatch, FakeChannel())

    with pytest.raises(RuntimeError, match="claims 99 character"):
        await delivery_module._deliver_channel(store, await _get(store, "m-bad"), "worker-1")


# -- a long send holds its own lease ------------------------------------------


async def test_every_chunk_goes_out_under_a_live_lease(monkeypatch, fake, store):
    """EVERY chunk — the first one included — must be in flight under a lease this worker
    holds, or the sweep can reclaim the record while a chunk is still going out and both
    workers send it."""
    await store.create_record(_record("m-long", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    # The sender's own lease, taken long enough ago that it has ALREADY lapsed: only the
    # per-chunk refresh can carry it through the send.
    assert await store.claim_delivery("m-long", time.time() - 300, "worker-1", 120) == 1
    assert _claim(fake, "m-long")[1] < time.time()

    observed: list[float] = []
    channel = FakeChannel("w1", watch=lambda: observed.append(_claim(fake, "m-long")[1]))
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-long"), "worker-1")

    assert len(observed) == 3
    assert all(expiry > time.time() for expiry in observed)


async def test_a_chunk_the_provider_never_answers_is_left_unledgered(monkeypatch, fake):
    """A provider call that hangs past the send timeout is INDETERMINATE: the executor
    stops there and does not ledger the chunk, so a re-drive re-sends it (a duplicate is
    the cheap side) instead of skipping text that may never have gone out."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_SEND_TIMEOUT_SECONDS", "0.05")
    store = ConversationRecordStore(ConversationsSettings())
    await store.create_record(_record("m-hang", "aaaaaaaaaabbbbbbbbbbcccccccccc"))

    channel = FakeChannel("w1", hang_on=2)
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-hang"), "worker-1")

    assert channel.sends == ["aaaaaaaaaa", "bbbbbbbbbb"]
    # Only the chunk the provider actually answered for is ledgered, so the resume point
    # sits at the start of the indeterminate chunk.
    ledger = ChannelSendLedger(ConversationsSettings())
    assert [c.chars for c in await ledger.sent_chunks("m-hang")] == [_CHUNK_CHARS]
    # The record is left non-terminal for the sweep to re-drive; nothing was truncated.
    assert (await _get(store, "m-hang")).delivery_status is DeliveryStatus.PENDING_DELIVERY


async def test_a_send_timeout_at_or_above_the_lease_is_refused(monkeypatch):
    """A send bounded at or above the lease can still be in flight after the sweep has
    re-claimed the record, which is the duplicate the lease exists to prevent."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_SEND_TIMEOUT_SECONDS", "120")
    with pytest.raises(ValueError, match="DELIVERY_SEND_TIMEOUT_SECONDS"):
        ConversationsSettings()


async def test_a_callback_timeout_at_or_above_the_lease_is_refused(monkeypatch):
    """A callback POST bounded at or above the lease can still be in flight after the sweep
    has re-claimed the record, which re-POSTs the identical signed callback — a duplicate
    delivery to the external receiver."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_CALLBACK_TIMEOUT_SECONDS", "120")
    with pytest.raises(ValueError, match="DELIVERY_CALLBACK_TIMEOUT_SECONDS"):
        ConversationsSettings()


async def test_a_send_in_flight_holds_the_record_against_a_racing_sweep(monkeypatch, fake, store):
    """The refreshed lease is what the sweep actually respects: a sweep pass over a record
    somebody is mid-send on claims nothing and sends nothing."""
    await store.create_record(_record("m-race", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    assert await store.claim_delivery("m-race", time.time() - 300, "worker-1", 120) == 1

    hanging = FakeChannel("w1", hang_on=3)
    _wire_channel(monkeypatch, hanging)
    sender = asyncio.create_task(delivery_module._deliver_channel(store, await _get(store, "m-race"), "worker-1"))
    for _ in range(200):
        if len(hanging.sends) == 3:
            break
        await asyncio.sleep(0.01)
    assert len(hanging.sends) == 3

    sweeper = FakeChannel("sweeper")
    _wire_channel(monkeypatch, sweeper)
    await delivery_module.sweep_stalled_deliveries()
    await _drain_spawned(store)

    assert sweeper.sends == []
    assert _claim(fake, "m-race")[0] == "worker-1"

    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender


async def test_a_send_stops_when_it_loses_the_lease_mid_flight(monkeypatch, fake, store):
    """Another worker takes the record over while chunk one is in flight. The sender must
    stop there rather than keep texting a human under an authority it no longer holds."""
    await store.create_record(_record("m-lost", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    assert await store.claim_delivery("m-lost", time.time(), "worker-1", 120) == 1

    async def _taken_over() -> None:
        if len(channel.sends) == 1:
            _expire_claim(fake, "m-lost")
            assert await store.claim_delivery("m-lost", time.time(), "worker-2", 120) == 1

    channel = FakeChannel("w1", watch=_taken_over)
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-lost"), "worker-1")

    assert channel.sends == ["aaaaaaaaaa"]
    assert (await _get(store, "m-lost")).delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert _claim(fake, "m-lost")[0] == "worker-2"
    # The ledger still names the chunk that did go out, so the new holder resumes after it.
    assert [c.chars for c in await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-lost")] == [_CHUNK_CHARS]


async def test_a_worker_that_lost_its_lease_cannot_terminalise_the_record(monkeypatch, fake, store):
    """The stale-terminal clobber: a worker whose lease lapsed and was taken over raises a
    provider refusal LATE. Its ``failed`` write must not overwrite the holder's record, and
    it must not clear the ledger the holder is resuming from."""
    await store.create_record(_record("m-stale", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    assert await store.claim_delivery("m-stale", time.time(), "worker-1", 120) == 1

    async def _taken_over() -> None:
        if len(channel.sends) == 1:
            _expire_claim(fake, "m-stale")
            assert await store.claim_delivery("m-stale", time.time(), "worker-2", 120) == 1

    # The refusal lands on the SAME chunk the takeover happened on, so worker-1 reaches its
    # terminal write holding nothing.
    channel = FakeChannel("w1", fail_on=1, watch=_taken_over)
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-stale"), "worker-1")

    assert (await _get(store, "m-stale")).delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert _claim(fake, "m-stale")[0] == "worker-2"


async def test_a_mid_send_receipt_does_not_truncate_the_answer(monkeypatch, fake, store):
    """A provider posts chunk one's DELIVERED callback while chunks two and three are still
    going out — a sub-second window Twilio routinely hits. The receipt must be refused
    loudly, not settle the record and strand the rest of the answer unsent."""
    await store.create_record(_record("m-early", "aaaaaaaaaabbbbbbbbbbcccccccccc"))
    assert await store.claim_delivery("m-early", time.time(), "worker-1", 120) == 1

    async def _receipt_arrives() -> None:
        # Chunk one is indexed and chunk two is in flight — the window a status callback
        # for chunk one actually lands in.
        if len(channel.sends) == 2:
            with pytest.raises(RuntimeError, match="has not finished"):
                await delivery_module.record_delivery_status("twilio", "w1-1", DeliveryReceipt.DELIVERED)

    channel = FakeChannel("w1", watch=_receipt_arrives)
    _wire_channel(monkeypatch, channel)
    await delivery_module._deliver_channel(store, await _get(store, "m-early"), "worker-1")

    assert channel.sends == ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
    record = await _get(store, "m-early")
    assert record.delivery_status is DeliveryStatus.PROVISIONAL
    # Now that the send IS finished, the same receipt settles it.
    await delivery_module.record_delivery_status("twilio", "w1-1", DeliveryReceipt.DELIVERED)
    assert (await _get(store, "m-early")).delivery_status is DeliveryStatus.DELIVERED


async def test_a_stale_failed_write_cannot_undo_a_completed_send(monkeypatch, fake, store):
    """The other half of the clobber: the answer is fully out and ``provisional``, so a
    late ``failed`` write from a worker that no longer owns the record is refused."""
    await store.create_record(_record("m-done", "aaaaaaaaaabbbbbbbbbb"))
    _wire_channel(monkeypatch, FakeChannel("w2"))
    await delivery_module._deliver_channel(store, await _get(store, "m-done"), "worker-2")
    assert (await _get(store, "m-done")).delivery_status is DeliveryStatus.PROVISIONAL

    assert await store.mark_failed("m-done", 1, time.time(), "worker-1") == -2
    assert (await _get(store, "m-done")).delivery_status is DeliveryStatus.PROVISIONAL


# -- the api door: the real signed callback and the lease it retries under ------


_CALLBACK_URL = "https://cb.example/hook"
_CALLBACK_SECRET = "sec-1"


def _api_route() -> ConversationRoute:
    return ConversationRoute(
        route_name="support",
        door="api",
        agent_name="echo",
        execution_key="svc",
        callback_url=_CALLBACK_URL,
        callback_secret=_CALLBACK_SECRET,
        execution_key_fingerprint="fp-1",
    )


class _FakeRouteManager:
    def __init__(self, route: ConversationRoute) -> None:
        self._route = route

    async def get_route(self, name: str) -> ConversationRoute | None:
        return self._route if name == self._route.route_name else None


def _api_record(message_id: str, answer: str) -> ConversationRecord:
    now = time.time()
    return ConversationRecord(
        message_id=message_id,
        route_name="support",
        door="api",
        thread_id="bridge:support:alice/user-7",
        client_address="alice/user-7",
        caller_principal="alice",
        callback_url=_CALLBACK_URL,
        answer_status="answered",
        answer=answer,
        created_at=now,
        updated_at=now,
    )


def _wire_callback(monkeypatch, handler) -> tuple[list[httpx.Request], list[dict]]:
    """Drive the REAL ``_post_callback`` against a local transport, so the request the
    executor actually builds is what is asserted. Returns the requests it made and the
    kwargs it constructed its client with."""
    monkeypatch.setattr(delivery_module, "get_conversations_manager", lambda: _FakeRouteManager(_api_route()))
    requests: list[httpx.Request] = []
    client_kwargs: list[dict] = []
    real_client = httpx.AsyncClient

    async def _handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return await handler(request)

    def _client(**kwargs):
        client_kwargs.append(kwargs)
        return real_client(transport=httpx.MockTransport(_handle), **kwargs)

    monkeypatch.setattr(delivery_module.httpx, "AsyncClient", _client)
    return requests, client_kwargs


async def test_the_callback_post_is_signed_under_the_rows_secret(monkeypatch, fake, store):
    """The whole point of the api door: a receiver recomputes the HMAC over the RAW BODY it
    received and compares it against the header the executor sent."""
    await store.create_record(_api_record("m-api", "the answer"))
    record = await _get(store, "m-api")

    async def _ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    requests, client_kwargs = _wire_callback(monkeypatch, _ok)
    assert await store.claim_delivery("m-api", time.time(), "worker-1", 120) == 1
    await delivery_module._deliver_api(store, record, "worker-1")

    assert len(requests) == 1
    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == _CALLBACK_URL
    assert sent.content == record.answer_payload().model_dump_json().encode()
    assert sent.headers["Content-Type"] == "application/json"

    signature = sent.headers["X-Tai-Signature"]
    expected = "sha256=" + hmac.new(_CALLBACK_SECRET.encode(), sent.content, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(signature, expected)
    forged = "sha256=" + hmac.new(b"wrong-secret", sent.content, hashlib.sha256).hexdigest()
    assert not hmac.compare_digest(signature, forged)

    # The callback is bounded and takes no proxy/CA configuration from the environment.
    assert client_kwargs == [{"timeout": ConversationsSettings().delivery_callback_timeout_seconds, "trust_env": False}]
    assert (await _get(store, "m-api")).delivery_status is DeliveryStatus.DELIVERED


async def test_the_callback_timeout_comes_from_settings(monkeypatch, fake, store):
    """The POST is bounded by ``delivery_callback_timeout_seconds``, not a hardcoded
    constant: an operator lowering it below the lease is honoured on the wire. Reverting
    ``_deliver_api`` to a fixed timeout leaves the client built with 15 and reddens this."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_CALLBACK_TIMEOUT_SECONDS", "7")
    store = ConversationRecordStore(ConversationsSettings())
    await store.create_record(_api_record("m-timeout", "the answer"))

    async def _ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    _, client_kwargs = _wire_callback(monkeypatch, _ok)
    assert await store.claim_delivery("m-timeout", time.time(), "worker-1", 120) == 1
    await delivery_module._deliver_api(store, await _get(store, "m-timeout"), "worker-1")

    assert client_kwargs == [{"timeout": 7.0, "trust_env": False}]
    assert (await _get(store, "m-timeout")).delivery_status is DeliveryStatus.DELIVERED


async def test_a_slow_callback_is_bounded_by_the_total_deadline_not_left_in_flight(monkeypatch, fake, store):
    """A receiver that answers slower than the timeout is cut off as a retryable non-2xx, so a
    POST cannot outlive the lease and a sweep re-claim cannot double-deliver. The bound is the
    total request time, not httpx's per-phase timeout: reverting to the plain httpx timeout (no
    asyncio.timeout wall) leaves this POST in flight and hangs the test past the deadline."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_CALLBACK_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_MAX_SECONDS", "0.01")
    store = ConversationRecordStore(ConversationsSettings())
    await store.create_record(_api_record("m-slow", "the answer"))

    async def _slow_then_ok(request: httpx.Request) -> httpx.Response:
        if len(requests) == 1:
            await asyncio.sleep(5)  # far past the 0.05s total deadline
        return httpx.Response(200)

    requests, _ = _wire_callback(monkeypatch, _slow_then_ok)
    assert await store.claim_delivery("m-slow", time.time(), "worker-1", 120) == 1
    async with asyncio.timeout(2):  # the fix bounds the POST; without it this test would hang
        await delivery_module._deliver_api(store, await _get(store, "m-slow"), "worker-1")

    assert len(requests) == 2  # the slow first POST was cut off and retried
    assert (await _get(store, "m-slow")).delivery_status is DeliveryStatus.DELIVERED


async def test_a_transport_failure_is_retried_rather_than_raised(monkeypatch, fake, store):
    """A connection reset is a retryable non-2xx, not an exception that strands the record:
    the record must still reach a terminal state on the next attempt."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_MAX_SECONDS", "0.01")
    store = ConversationRecordStore(ConversationsSettings())
    await store.create_record(_api_record("m-blip", "the answer"))

    async def _reset_then_accept(request: httpx.Request) -> httpx.Response:
        if len(requests) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200)

    requests, _ = _wire_callback(monkeypatch, _reset_then_accept)
    assert await store.claim_delivery("m-blip", time.time(), "worker-1", 120) == 1
    await delivery_module._deliver_api(store, await _get(store, "m-blip"), "worker-1")

    assert len(requests) == 2
    assert (await _get(store, "m-blip")).delivery_status is DeliveryStatus.DELIVERED


async def test_the_api_retry_extends_the_lease_over_its_own_backoff(monkeypatch, fake, store):
    """The backoff is dead time the record must stay claimed through, so the refresh leases
    it for the backoff PLUS a full lease — a plain lease could lapse mid-sleep."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_BASE_SECONDS", "0.5")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_MAX_SECONDS", "0.5")
    settings = ConversationsSettings()
    store = ConversationRecordStore(settings)
    await store.create_record(_api_record("m-backoff", "the answer"))
    observed: list[float] = []

    async def _refuse_then_accept(request: httpx.Request) -> httpx.Response:
        if len(requests) == 1:
            observed.append(time.time())
            return httpx.Response(500)
        observed.append(_claim(fake, "m-backoff")[1])
        return httpx.Response(200)

    requests, _ = _wire_callback(monkeypatch, _refuse_then_accept)
    assert await store.claim_delivery("m-backoff", time.time(), "worker-1", 120) == 1
    await delivery_module._deliver_api(store, await _get(store, "m-backoff"), "worker-1")

    first_post, expiry = observed
    assert expiry >= first_post + 0.5 + settings.delivery_claim_lease_seconds


async def test_the_api_retry_stops_when_it_loses_the_lease_during_backoff(monkeypatch, fake, store):
    """Another worker takes the record over while this one is holding a 500. It must not
    wake up and POST a SECOND callback the customer's endpoint has already been sent."""
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_MAX_SECONDS", "0.01")
    store = ConversationRecordStore(ConversationsSettings())
    await store.create_record(_api_record("m-taken", "the answer"))

    async def _refuse_and_hand_over(request: httpx.Request) -> httpx.Response:
        _expire_claim(fake, "m-taken")
        assert await store.claim_delivery("m-taken", time.time(), "worker-2", 120) == 1
        return httpx.Response(500)

    requests, _ = _wire_callback(monkeypatch, _refuse_and_hand_over)
    assert await store.claim_delivery("m-taken", time.time(), "worker-1", 120) == 1
    await delivery_module._deliver_api(store, await _get(store, "m-taken"), "worker-1")

    assert len(requests) == 1
    assert (await _get(store, "m-taken")).delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert _claim(fake, "m-taken")[0] == "worker-2"


# -- the periodic loop drives both recovery passes -----------------------------


async def test_the_periodic_loop_runs_every_recovery_pass(monkeypatch):
    """A worker that dies and never reboots leaves its ``accepted`` records and its stale
    terminal-index members to a sibling's PERIODIC passes; a failing pass must not skip the
    rest, or a boot-only re-drive would hold those messages unanswered for every process."""
    ran: list[str] = []

    async def _broken_delivery_pass() -> None:
        ran.append("delivery")
        raise RuntimeError("this pass is broken")

    async def _intake_pass() -> None:
        ran.append("intake")

    async def _prune_pass(self) -> None:
        ran.append("prune")

    monkeypatch.setattr(delivery_module, "sweep_stalled_deliveries", _broken_delivery_pass)
    monkeypatch.setattr(turn_module, "redrive_accepted", _intake_pass)
    monkeypatch.setattr(records_module.ConversationRecordStore, "prune_expired_terminal_indexes", _prune_pass)

    loop = asyncio.create_task(delivery_module._sweep_loop(0.01))
    for _ in range(200):
        if ran.count("prune") >= 2:
            break
        await asyncio.sleep(0.01)
    loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop

    # Every pass runs every tick, and a failing one does not skip the rest.
    assert ran[:6] == ["delivery", "intake", "prune", "delivery", "intake", "prune"]


async def test_starting_the_sweep_twice_cancels_the_first_task(monkeypatch):
    """A second start must cancel the task the first started, or two sweep loops run and
    every recovery pass fires twice a tick."""
    delivery_module.start_delivery_sweep()
    first = delivery_module._sweep_task
    assert first is not None
    assert not first.done()

    delivery_module.start_delivery_sweep()
    second = delivery_module._sweep_task
    assert second is not None
    assert second is not first
    with pytest.raises(asyncio.CancelledError):
        await first
    assert first.cancelled()

    await delivery_module.stop_delivery_sweep()
    assert second.cancelled()
    assert delivery_module._sweep_task is None


async def test_stopping_the_sweep_cancels_and_clears_it(monkeypatch):
    """Shutdown cancels the running sweep and clears the handle; a second stop is a no-op."""
    delivery_module.start_delivery_sweep()
    task = delivery_module._sweep_task
    assert task is not None

    await delivery_module.stop_delivery_sweep()
    assert task.cancelled()
    assert delivery_module._sweep_task is None
    await delivery_module.stop_delivery_sweep()  # nothing running: a clean no-op


# -- a send that can never complete leaves pending_delivery -------------------


async def test_a_corrupt_ledger_drives_the_record_terminal(monkeypatch, fake, store):
    """A ledger that cannot describe the answer can never resume, so the record must reach
    a terminal state: left pending_delivery it is re-driven every lease expiry forever."""
    await store.create_record(_record("m-bad", "short"))
    await ChannelSendLedger(ConversationsSettings()).append("m-bad", 99, ["w1-1"])
    _wire_channel(monkeypatch, FakeChannel())

    with pytest.raises(RuntimeError, match="claims 99 character"):
        await delivery_module._deliver_channel(store, await _get(store, "m-bad"), "worker-1")

    record = await _get(store, "m-bad")
    assert record.delivery_status is DeliveryStatus.FAILED
    # The attempt is accounted before the ledger is read, so any later fault is bounded too.
    assert record.attempts == 1
    assert await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-bad") == []


async def test_a_provisional_record_is_not_re_sent_by_a_second_deliver(monkeypatch, fake, store):
    """The sweep/spawn race: a deliver() reaching a record after it went provisional must
    claim nothing and re-send nothing — the ledger is cleared, so a re-send would emit the
    whole answer to a human a second time."""
    channel = FakeChannel("w1")
    _wire_channel(monkeypatch, channel)
    await store.create_record(_record("m-prov2", "aaaaaaaaaabbbbbbbbbb"))
    await delivery_module._deliver_channel(store, await _get(store, "m-prov2"), "worker-1")
    assert channel.sends == ["aaaaaaaaaa", "bbbbbbbbbb"]
    assert (await _get(store, "m-prov2")).delivery_status is DeliveryStatus.PROVISIONAL

    resending = FakeChannel("w2")
    _wire_channel(monkeypatch, resending)
    await delivery_module.deliver("m-prov2")

    assert resending.sends == []
    assert (await _get(store, "m-prov2")).delivery_status is DeliveryStatus.PROVISIONAL


async def test_deliver_refuses_a_record_that_is_not_pending_after_the_claim(monkeypatch, fake, store):
    """Defense in depth over the claim: were the claim ever to admit a record past
    pending_delivery, deliver() refuses loudly rather than re-sending a fully sent answer."""
    channel = FakeChannel()
    _wire_channel(monkeypatch, channel)
    await store.create_record(_record("m-guard", "the answer"))
    await store.mark_provisional("m-guard", ["out-1"], 1, time.time(), "tok")

    async def _admit(self, message_id, now, token, lease):
        return 1

    monkeypatch.setattr(records_module.ConversationRecordStore, "claim_delivery", _admit)

    with pytest.raises(RuntimeError, match="not pending_delivery"):
        await delivery_module.deliver("m-guard")
    assert channel.sends == []


async def test_a_transient_ledger_read_error_leaves_the_record_deliverable(monkeypatch, fake, store):
    """A redis blip on the ledger read is NOT a corrupt ledger: it must propagate so the
    sweep re-drives, never terminal-fail a deliverable answer at attempt one."""
    import redis.exceptions

    await store.create_record(_record("m-blip", "aaaaaaaaaabbbbbbbbbb"))
    _wire_channel(monkeypatch, FakeChannel())

    async def _boom(self, message_id):
        raise redis.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(ledger_module.ChannelSendLedger, "sent_chunks", _boom)

    with pytest.raises(redis.exceptions.ConnectionError):
        await delivery_module._deliver_channel(store, await _get(store, "m-blip"), "worker-1")

    record = await _get(store, "m-blip")
    assert record.delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert record.attempts == 1


async def test_an_unparseable_ledger_entry_drives_the_record_terminal(monkeypatch, fake, store):
    """A stored entry that cannot be decoded is a record-shaped fault, not a transient one:
    the record reaches a terminal ``failed`` rather than being re-driven forever."""
    await store.create_record(_record("m-garbled", "aaaaaaaaaabbbbbbbbbb"))
    fake._lists[ConversationsSettings().chunk_ledger_key("m-garbled")] = ["{not valid json"]
    _wire_channel(monkeypatch, FakeChannel())

    with pytest.raises(ledger_module.LedgerInconsistentError, match="unparseable"):
        await delivery_module._deliver_channel(store, await _get(store, "m-garbled"), "worker-1")

    assert (await _get(store, "m-garbled")).delivery_status is DeliveryStatus.FAILED
    assert await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-garbled") == []


async def test_a_corrupt_ledger_fault_under_a_foreign_lease_spares_the_ledger(monkeypatch, fake, store):
    """The corrupt-ledger fail path honors the foreign-lease guard: a worker whose lease was
    taken over may not clear the ledger the new holder is resuming from."""
    await store.create_record(_record("m-race2", "short"))
    ledger = ChannelSendLedger(ConversationsSettings())
    await ledger.append("m-race2", 99, ["w1-1"])
    # A different worker holds the live lease now.
    assert await store.claim_delivery("m-race2", time.time(), "worker-2", 120) == 1
    _wire_channel(monkeypatch, FakeChannel())

    with pytest.raises(RuntimeError, match="claims 99 character"):
        await delivery_module._deliver_channel(store, await _get(store, "m-race2"), "worker-1")

    # The terminal write is refused (-3) and the holder's ledger survives.
    assert (await _get(store, "m-race2")).delivery_status is DeliveryStatus.PENDING_DELIVERY
    assert [c.chars for c in await ledger.sent_chunks("m-race2")] == [99]


async def test_a_chunk_is_ledgered_before_it_is_reverse_indexed(monkeypatch, fake, store):
    """The load-bearing write order: a crash between the ledger append and the reverse index
    must leave the chunk LEDGERED, so a re-drive resumes after it and never re-sends it.
    Swapping the two calls leaves the sent chunk unledgered and this test goes red."""
    await store.create_record(_record("m-order", "aaaaaaaaaabbbbbbbbbb"))
    channel = FakeChannel("w1")
    _wire_channel(monkeypatch, channel)
    index_outbound = ConversationRecordStore.index_outbound

    async def _raise_first_time(self, channel_name, outbound_ids, message_id):
        raise RuntimeError("crash between the ledger and the reverse index")

    monkeypatch.setattr(ConversationRecordStore, "index_outbound", _raise_first_time)

    with pytest.raises(RuntimeError, match="crash between the ledger"):
        await delivery_module._deliver_channel(store, await _get(store, "m-order"), "worker-1")

    # The first chunk went out and was ledgered before the reverse index was attempted.
    assert channel.sends == ["aaaaaaaaaa"]
    monkeypatch.setattr(ConversationRecordStore, "index_outbound", index_outbound)
    assert [c.chars for c in await ChannelSendLedger(ConversationsSettings()).sent_chunks("m-order")] == [_CHUNK_CHARS]


async def test_a_whitespace_only_chunk_is_ledgered_and_never_sent(monkeypatch, fake, store):
    """A hard cut can leave a chunk of pure whitespace, which the channel contract refuses.
    It is accounted in the ledger and skipped, so the send still completes."""
    await store.create_record(_record("m-ws", "A" * _CHUNK_CHARS + "\n"))
    channel = FakeChannel("w1")
    _wire_channel(monkeypatch, channel)

    await delivery_module._deliver_channel(store, await _get(store, "m-ws"), "worker-1")

    assert channel.sends == ["A" * _CHUNK_CHARS]
    assert (await _get(store, "m-ws")).delivery_status is DeliveryStatus.PROVISIONAL


async def test_an_answer_over_the_fan_out_cap_is_refused_as_an_error_outcome(monkeypatch, fake):
    """A huge answer would fan one inbound message out into many billable provider sends.
    Past the cap the whole answer is refused with ONE client-safe reply and a failed record,
    never a partial or truncated fan-out."""
    monkeypatch.setenv("CONVERSATIONS_MAX_OUTBOUND_CHUNKS", "3")
    store = ConversationRecordStore(ConversationsSettings())
    channel = FakeChannel()
    _wire_channel(monkeypatch, channel)
    # 100 unbreakable characters at width 10 split into 10 chunks, over the cap of 3.
    await store.create_record(_record("m-huge", "x" * (10 * _CHUNK_CHARS)))

    await delivery_module._deliver_channel(store, await _get(store, "m-huge"), "worker-1")

    record = await _get(store, "m-huge")
    assert record.delivery_status is DeliveryStatus.FAILED
    # Exactly one client-safe reply went out; the answer itself was never fanned out.
    assert channel.sends == [delivery_module._OVERSIZED_ANSWER_TEXT]
    assert record.attempts == 1


async def test_a_partial_send_is_not_refused_when_the_cap_is_lowered_mid_flight(monkeypatch, fake):
    """The fan-out cap is an ADMISSION gate, not retroactive. A send already partway out —
    two chunks ledgered — must COMPLETE even after an operator lowers max_outbound_chunks
    below the full answer's chunk count: a human has seen part of the answer and it cannot
    be un-sent. Reverting the 'only refuse when nothing sent' gate refuses this record and
    turns this test red."""
    monkeypatch.setenv("CONVERSATIONS_MAX_OUTBOUND_CHUNKS", "3")
    store = ConversationRecordStore(ConversationsSettings())
    # 4 chunks at width 10, over the newly-lowered cap of 3.
    answer = "aaaaaaaaaabbbbbbbbbbccccccccccdddddddddd"
    await store.create_record(_record("m-resume", answer))
    ledger = ChannelSendLedger(ConversationsSettings())
    await ledger.append("m-resume", _CHUNK_CHARS, ["w1-1"])
    await ledger.append("m-resume", _CHUNK_CHARS, ["w1-2"])

    resuming = FakeChannel("w2")
    _wire_channel(monkeypatch, resuming)
    await delivery_module._deliver_channel(store, await _get(store, "m-resume"), "worker-1")

    # It resumed and finished the remaining chunks — never the client-safe refusal.
    assert resuming.sends == ["cccccccccc", "dddddddddd"]
    record = await _get(store, "m-resume")
    assert record.delivery_status is DeliveryStatus.PROVISIONAL
    assert record.outbound_message_ids == ["w1-1", "w1-2", "w2-1", "w2-2"]
