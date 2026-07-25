"""The delivery executor — sends a produced answer back and drives its record to a
terminal state, exactly once.

``door=channel`` chunks the answer through the channel's ``notify``, resuming an interrupted
send from the per-chunk ledger; ``door=api`` POSTs it to the row's ``callback_url`` under an
HMAC ``X-Tai-Signature``, retried with backoff. Every send is guarded by an atomic per-record
leased claim, and a periodic sweep re-drives the records whose lease has lapsed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import time
from uuid import uuid4

import httpx
from tai42_contract.app import tai42_app
from tai42_contract.channels import Channel, ChannelDeliveryError, ChannelNotification
from tai42_contract.conversations import DeliveryReceipt

from tai42_skeleton.conversations.cache import get_conversations_manager
from tai42_skeleton.conversations.ledger import ChannelSendLedger, LedgerInconsistentError
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings

logger = logging.getLogger(__name__)

# Signed api-door callback header: ``HMAC-SHA256(callback_secret, raw_body)`` in hex.
_SIGNATURE_HEADER = "X-Tai-Signature"
_SIGNATURE_PREFIX = "sha256="

# Client-safe reply when an answer splits into more provider messages than the fan-out cap
# allows; the whole answer is refused rather than fanned out or silently truncated.
_OVERSIZED_ANSWER_TEXT = "Sorry, the answer was too long to send here. Please ask for a shorter response."

# Strong references to in-flight delivery / grace tasks so they are not GC'd mid-flight.
_DELIVERY_TASKS: set[asyncio.Task[None]] = set()

# The periodic stalled-delivery sweep, held so the lifespan can cancel it at shutdown.
_sweep_task: asyncio.Task[None] | None = None


def _store() -> ConversationRecordStore:
    return ConversationRecordStore(ConversationsSettings())


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def _backoff_seconds(settings: ConversationsSettings, attempt: int) -> float:
    """Exponential backoff for retry ``attempt`` (1-based): ``base * 2**(attempt-1)``
    capped at the configured maximum."""
    raw = settings.delivery_backoff_base_seconds * (2 ** (attempt - 1))
    return min(raw, settings.delivery_backoff_max_seconds)


def split_message(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into ordered chunks of at most ``max_chars``, breaking at the last
    newline or space in the window when there is one. The concatenation of the chunks is
    exactly ``text`` — nothing is dropped or reordered."""
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind("\n"), window.rfind(" "))
        # Hard-cut when a whitespace break would leave an empty head, so a run of
        # non-breakable characters still makes progress.
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks


# -- one-record delivery -----------------------------------------------------


async def deliver(message_id: str) -> None:
    """Drive one record's delivery to a terminal or provisional state, exactly once.

    Takes the atomic claim FIRST, so a record already terminal, gone, or under another
    worker's live lease is left untouched. A record still at intake carries no answer and
    refuses loudly."""
    store = _store()
    token = uuid4().hex
    claimed = await store.claim_delivery(message_id, time.time(), token, store.settings.delivery_claim_lease_seconds)
    if claimed == -2:
        raise RuntimeError(
            f"conversation record {message_id!r} is still at intake and carries no answer; the delivery machine "
            "must not be driven on it"
        )
    if claimed != 1:
        # Not an error: terminal, gone, or held by another worker's live lease.
        logger.debug("conversations: record %s was not claimed for delivery (claim returned %d)", message_id, claimed)
        return
    record = await store.get_record(message_id)
    if record is None:
        # Deleted between the claim and this re-read.
        return
    if record.delivery_status is not DeliveryStatus.PENDING_DELIVERY:
        # The claim admits only pending_delivery; a provisional record is fully sent and
        # awaits a receipt. Reaching here means the claim and this guard disagree.
        raise RuntimeError(
            f"conversation record {message_id!r} is {record.delivery_status.value}, not pending_delivery, after "
            "winning the delivery claim; the delivery machine must not re-send it"
        )
    if record.attempts > 0:
        # Earlier attempts on a non-terminal record: this claim is a takeover.
        logger.info(
            "conversations: re-driving record %s (door=%s) after %d unfinished attempt(s)",
            record.message_id,
            record.door,
            record.attempts,
        )
    if record.attempts >= store.settings.delivery_max_attempts:
        # Attempts spent without a terminal state: fail it so the sweep stops re-driving it.
        outcome = await store.mark_failed(record.message_id, record.attempts, time.time(), token)
        logger.error(
            "conversations: record %s exhausted %d delivery attempt(s) without a terminal outcome "
            "(failed write returned %d)",
            record.message_id,
            record.attempts,
            outcome,
        )
        return
    if record.door == "channel":
        await _deliver_channel(store, record, token)
    else:
        await _deliver_api(store, record, token)


async def _deliver_channel(store: ConversationRecordStore, record: ConversationRecord, token: str) -> None:
    settings = store.settings
    channel_name = record.channel
    if channel_name is None:
        raise RuntimeError(f"channel record {record.message_id!r} carries no channel to deliver on")
    answer = record.answer
    if answer is None:
        raise RuntimeError(
            f"channel record {record.message_id!r} is {record.delivery_status.value} and carries no answer to send"
        )
    max_chars = settings.max_message_chars.get(channel_name)
    if max_chars is None:
        # Config error: fail the record so the outcome is visible, then raise loudly.
        await store.mark_failed(record.message_id, await store.bump_attempt(record.message_id), time.time(), token)
        raise RuntimeError(
            f"channel {channel_name!r} has no max_message_chars entry; add it to CONVERSATIONS_MAX_MESSAGE_CHARS"
        )

    try:
        channel = tai42_app.channels.get(channel_name)
    except KeyError as exc:
        # Same config-error treatment: fail the record so the sweep stops re-driving a
        # send that could never complete, then raise loudly.
        await store.mark_failed(record.message_id, await store.bump_attempt(record.message_id), time.time(), token)
        raise RuntimeError(
            f"channel {channel_name!r} is routed but is not registered on this deployment; load its channel "
            "plugin or remove the route"
        ) from exc

    ledger = ChannelSendLedger(settings)
    # Account the attempt BEFORE the first fallible step, so every fault below is bounded
    # by ``delivery_max_attempts`` instead of leaving the record pending_delivery forever.
    attempts = await store.bump_attempt(record.message_id)
    try:
        sent = await ledger.sent_chunks(record.message_id)
        chars_sent = sum(chunk.chars for chunk in sent)
        if chars_sent > len(answer):
            raise LedgerInconsistentError(
                f"send ledger for record {record.message_id!r} claims {chars_sent} character(s) already sent of an "
                f"answer that is {len(answer)} character(s) long"
            )
    except LedgerInconsistentError:
        # A ledger that cannot describe the answer can never resume, so the record is
        # failed here — as a config error is — before the refusal is raised. A transient
        # store fault instead propagates, leaving the record for the sweep to re-drive.
        failed = await store.mark_failed(record.message_id, attempts, time.time(), token)
        if failed == 1:
            # Clear only under this worker's own terminal write: a foreign takeover owns the
            # ledger it is resuming from.
            await ledger.clear(record.message_id)
        raise

    if not sent:
        # Fan-out cap is an ADMISSION decision, never retroactive: refuse an oversized answer
        # ONLY before any chunk has gone out. A resume (sent non-empty) always completes —
        # a human has seen part of the answer and it cannot be un-sent.
        answer_chunks = len(split_message(answer, max_chars))
        if answer_chunks > settings.max_outbound_chunks:
            await _refuse_oversized_answer(store, record, channel, answer_chunks, attempts, token)
            return

    outbound_ids = [outbound_id for chunk in sent for outbound_id in chunk.outbound_ids]
    if sent:
        # Re-index the ledger's ids: a chunk accepted just before a crash may never have
        # reached the reverse index, and a receipt naming an unindexed id resolves to nothing.
        await store.index_outbound(channel_name, outbound_ids, record.message_id)
        logger.info(
            "conversations: resuming channel delivery of record %s on %r at character %d/%d (%d chunk(s) already "
            "accepted by the provider)",
            record.message_id,
            channel_name,
            chars_sent,
            len(answer),
            len(sent),
        )
    remaining = answer[chars_sent:]
    # An answer already fully out goes straight to the provisional write.
    chunks = split_message(remaining, max_chars) if remaining else []
    total_chunks = len(sent) + len(chunks)
    accepted_chunks = len(sent)
    try:
        for chunk in chunks:
            # Refresh the lease BEFORE each send, and bound the send strictly under it, so
            # the whole notify window is covered by a claim this worker holds — an accepted
            # chunk is never ledgered by a worker that has already been taken over.
            held = await store.claim_delivery(
                record.message_id, time.time(), token, settings.delivery_claim_lease_seconds
            )
            if held != 1:
                # Lease lost: stop sending. The ledger tells the new holder where to resume.
                logger.warning(
                    "conversations: lost the delivery lease on record %s after %d/%d chunk(s) (claim returned %d); "
                    "leaving the remainder to the worker that holds it now",
                    record.message_id,
                    accepted_chunks,
                    total_chunks,
                    held,
                )
                return
            if not chunk.strip():
                # The channel contract has no blank message. A split boundary can leave a
                # chunk of pure whitespace: it is ledgered as sent, so the resume arithmetic
                # stays exact, and no provider call is made for it.
                await ledger.append(record.message_id, len(chunk), [])
                accepted_chunks += 1
                continue
            try:
                async with asyncio.timeout(settings.delivery_send_timeout_seconds):
                    ids = await channel.notify(
                        ChannelNotification(
                            message=chunk, recipient=record.client_address, sender_identity=record.our_identity
                        )
                    )
            except TimeoutError:
                # Indeterminate: the provider may have taken the chunk. It is deliberately
                # NOT ledgered, so a re-drive re-sends it — the same asymmetry the ledger
                # ordering takes, where a duplicate message is the cheaper side of a loss.
                logger.error(
                    "conversations: channel %r did not answer within %ss for chunk %d/%d of record %s; the chunk is "
                    "indeterminate and is left unledgered for a re-drive to re-send",
                    channel_name,
                    settings.delivery_send_timeout_seconds,
                    accepted_chunks + 1,
                    total_chunks,
                    record.message_id,
                )
                return
            # ``None`` from a channel off the id-returning contract is an accepted send
            # with no correlatable id, not a dropped one.
            accepted = list(ids or [])
            # Ledger BEFORE the reverse index: a crash between the two costs an
            # unresolvable receipt, a crash before the ledger entry costs a duplicate message.
            await ledger.append(record.message_id, len(chunk), accepted)
            await store.index_outbound(channel_name, accepted, record.message_id)
            outbound_ids.extend(accepted)
            accepted_chunks += 1
    except ChannelDeliveryError:
        # No blind retry: the medium offers no idempotency key, so a retry would re-send
        # the chunks already accepted. A mid-sequence failure is terminal ``failed``.
        failed = await store.mark_failed(record.message_id, attempts, time.time(), token)
        if failed == 1:
            await ledger.clear(record.message_id)
        logger.error(
            "conversations: channel delivery of record %s on %r failed after %d/%d chunk(s) (failed write returned %d)",
            record.message_id,
            channel_name,
            accepted_chunks,
            total_chunks,
            failed,
            exc_info=True,
        )
        return

    outcome = await store.mark_provisional(record.message_id, outbound_ids, attempts, time.time(), token)
    if outcome != 1:
        # The record left this worker's hands; its ledger belongs to whoever holds it now.
        logger.warning(
            "conversations: record %s was not moved to provisional after a full send (provisional write returned "
            "%d); the send ledger is left for the worker that owns it now",
            record.message_id,
            outcome,
        )
        return
    await ledger.clear(record.message_id)
    # Fallback confirmation for a medium whose receipt never arrives.
    _spawn(_confirm_after_grace(record.message_id, settings.delivery_grace_seconds))


async def _refuse_oversized_answer(
    store: ConversationRecordStore,
    record: ConversationRecord,
    channel: Channel,
    chunk_count: int,
    attempts: int,
    token: str,
) -> None:
    """Refuse an answer past the fan-out cap: send ONE client-safe reply and fail the record
    loudly. A best-effort provider refusal is suppressed — the record still fails."""
    settings = store.settings
    logger.error(
        "conversations: record %s answer splits into %d chunk(s), over the max_outbound_chunks cap of %d on "
        "channel %r; refusing with a client-safe reply and failing the record",
        record.message_id,
        chunk_count,
        settings.max_outbound_chunks,
        record.channel,
    )
    with contextlib.suppress(ChannelDeliveryError, TimeoutError):
        async with asyncio.timeout(settings.delivery_send_timeout_seconds):
            await channel.notify(
                ChannelNotification(
                    message=_OVERSIZED_ANSWER_TEXT,
                    recipient=record.client_address,
                    sender_identity=record.our_identity,
                )
            )
    await store.mark_failed(record.message_id, attempts, time.time(), token)


async def _deliver_api(store: ConversationRecordStore, record: ConversationRecord, token: str) -> None:
    settings = store.settings
    route = await get_conversations_manager().get_route(record.route_name)
    if route is None or route.callback_secret is None or record.callback_url is None:
        await store.mark_failed(record.message_id, await store.bump_attempt(record.message_id), time.time(), token)
        logger.error(
            "conversations: api record %s cannot be delivered — route %r is gone or carries no callback secret; "
            "marked failed",
            record.message_id,
            record.route_name,
        )
        return

    body = record.answer_payload().model_dump_json().encode()
    signature = _sign(route.callback_secret, body)
    while True:
        attempt = await store.bump_attempt(record.message_id)
        status = await _post_callback(record.callback_url, body, signature, settings.delivery_callback_timeout_seconds)
        if status is not None and 200 <= status < 300:
            delivered = await store.mark_delivered(record.message_id, [], attempt, time.time(), token)
            if delivered != 1:
                logger.warning(
                    "conversations: api callback for record %s succeeded but the delivered write returned %d; "
                    "the record's outcome stands as another writer left it",
                    record.message_id,
                    delivered,
                )
            return
        if attempt >= settings.delivery_max_attempts:
            failed = await store.mark_failed(record.message_id, attempt, time.time(), token)
            logger.error(
                "conversations: api callback for record %s to %s exhausted %d attempts (last status %s; failed write "
                "returned %d)",
                record.message_id,
                record.callback_url,
                attempt,
                status,
                failed,
            )
            return
        backoff = _backoff_seconds(settings, attempt)
        # Extend the lease over the upcoming backoff, or a re-drive reclaims a record this
        # worker is still retrying.
        held = await store.claim_delivery(
            record.message_id, time.time(), token, backoff + settings.delivery_claim_lease_seconds
        )
        if held != 1:
            # Lease lost: stop retrying rather than POST a second callback for a record
            # another worker now drives.
            logger.warning(
                "conversations: lost the delivery lease on record %s after attempt %d (claim returned %d); "
                "leaving the retry to the worker that holds it now",
                record.message_id,
                attempt,
                held,
            )
            return
        await asyncio.sleep(backoff)


async def _post_callback(url: str, body: bytes, signature: str, timeout_seconds: float) -> int | None:
    """POST the signed answer body, returning the HTTP status, or ``None`` when the request
    never completed — a transport error or a timeout is a retryable non-2xx, logged not raised.

    ``timeout_seconds`` is a hard total-request deadline (validated below the delivery lease), not
    httpx's per-phase timeout, so a slow receiver cannot keep the POST in flight past the lease.
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
                response = await client.post(
                    url,
                    content=body,
                    headers={"Content-Type": "application/json", _SIGNATURE_HEADER: signature},
                )
            return response.status_code
    except (httpx.HTTPError, TimeoutError):
        logger.warning("conversations: callback POST to %s failed or timed out; will retry", url)
        return None


async def _confirm_after_grace(message_id: str, grace_seconds: float) -> None:
    """Confirm a still-``provisional`` record ``delivered`` once its grace window elapses.
    The atomic ingest is a no-op on a record a receipt already made terminal."""
    await asyncio.sleep(grace_seconds)
    await _store().ingest_receipt(message_id, DeliveryReceipt.DELIVERED, time.time())


# -- out-of-band receipt sink ------------------------------------------------


async def record_delivery_status(channel: str, provider_message_id: str, status: DeliveryReceipt) -> None:
    """Ingest a channel's out-of-band receipt for an outbound message.

    Resolves ``provider_message_id`` through the outbound reverse index and applies the
    receipt atomically. Raises when the id maps to no record (unknown or already swept), and
    when it names a record whose send is still in flight — a receipt for one chunk may not
    terminalise a record whose remaining chunks are still going out."""
    store = _store()
    message_id = await store.resolve_outbound(channel, provider_message_id)
    if message_id is None:
        raise LookupError(
            f"conversations: delivery receipt names outbound id {provider_message_id!r} on channel {channel!r} "
            "that maps to no answer record"
        )
    outcome = await store.ingest_receipt(message_id, status, time.time())
    if outcome == -1:
        raise LookupError(
            f"conversations: outbound id {provider_message_id!r} resolved to record {message_id} which no longer exists"
        )
    if outcome == -3:
        raise RuntimeError(
            f"conversations: delivery receipt {status.value} for outbound id {provider_message_id!r} names record "
            f"{message_id}, whose send has not finished; the record keeps its in-flight state"
        )
    if outcome == -2:
        logger.error(
            "conversations: delivery receipt %s for record %s conflicts with an already-terminal state; ignored",
            status.value,
            message_id,
        )
    elif status is DeliveryReceipt.FAILED and outcome == 1:
        logger.error("conversations: outbound delivery of record %s reported FAILED by the channel", message_id)


async def mark_wait_delivered(message_id: str) -> bool:
    """Confirm a record ``delivered`` because the API door's sync wait returned its answer;
    no callback is POSTed. Takes the same atomic claim a background delivery would, so only
    one of the two paths delivers. Returns ``True`` when this call is that one."""
    store = _store()
    token = uuid4().hex
    claimed = await store.claim_delivery(message_id, time.time(), token, store.settings.delivery_claim_lease_seconds)
    if claimed != 1:
        return False
    outcome = await store.mark_delivered(message_id, [], await store.bump_attempt(message_id), time.time(), token)
    return outcome == 1


# -- startup re-drive + periodic sweep ---------------------------------------


async def redrive_pending() -> None:
    """Resume every non-terminal record on boot so nothing is stranded: re-deliver a
    ``pending_delivery`` record through the exactly-once claim, confirm a ``provisional``
    one past its grace, and reschedule the fallback confirmation for one still inside it.

    Only boot can rebuild the in-process grace timers lost with the previous process, so
    the rescheduling lives here and not in the periodic sweep."""
    store = _store()
    now = time.time()
    for work in await store.pending_work():
        if work.delivery_status is DeliveryStatus.PENDING_DELIVERY:
            _spawn(deliver(work.message_id))
        elif work.delivery_status is DeliveryStatus.PROVISIONAL:
            if work.grace_deadline is not None and now >= work.grace_deadline:
                await store.ingest_receipt(work.message_id, DeliveryReceipt.DELIVERED, now)
            elif work.grace_deadline is not None:
                _spawn(_confirm_after_grace(work.message_id, work.grace_deadline - now))
            else:
                _spawn(_confirm_after_grace(work.message_id, store.settings.delivery_grace_seconds))


async def sweep_stalled_deliveries() -> None:
    """One pass over the records a dead worker could have stranded.

    The exactly-once claim is itself the lease-expiry test, so a re-driven
    ``pending_delivery`` record is only ever a genuinely abandoned one. A ``provisional``
    record past its grace is confirmed here because the in-process fallback confirmation
    died with the worker that scheduled it."""
    store = _store()
    now = time.time()
    for work in await store.pending_work():
        if work.delivery_status is DeliveryStatus.PENDING_DELIVERY:
            _spawn(deliver(work.message_id))
        elif (
            work.delivery_status is DeliveryStatus.PROVISIONAL
            and work.grace_deadline is not None
            and now >= work.grace_deadline
        ):
            await store.ingest_receipt(work.message_id, DeliveryReceipt.DELIVERED, now)


def start_delivery_sweep() -> None:
    """Start (or restart) the periodic recovery sweep — stalled deliveries and lapsed
    intakes. Must be called ON the serving loop, so the task attaches to the loop its
    deliveries run on."""
    global _sweep_task
    task = _sweep_task
    if task is not None and not task.done():
        task.cancel()
    interval = ConversationsSettings().delivery_sweep_interval_seconds
    logger.info("conversations: sweeping for stalled deliveries and lapsed intakes every %ss", interval)
    _sweep_task = asyncio.create_task(_sweep_loop(interval), name="tai-conversations-delivery-sweep")
    _sweep_task.add_done_callback(_on_sweep_done)


async def stop_delivery_sweep() -> None:
    """Cancel and await the sweep task. Must be called ON the serving loop the task lives
    on, so the await is loop-safe. Only ``CancelledError`` is suppressed."""
    global _sweep_task
    task = _sweep_task
    _sweep_task = None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _redrive_lapsed_intakes() -> None:
    """Adopt and resolve records whose turn worker died, through the turn engine's
    lease-gated re-drive. Imported inside the call: the turn engine imports this module."""
    from tai42_skeleton.conversations.turn import redrive_accepted

    await redrive_accepted()


async def _prune_terminal_indexes() -> None:
    """Drop the expired members of the terminal-status indexes that no listing reads, so
    they cannot outgrow the retained keyspace they name."""
    await _store().prune_expired_terminal_indexes()


async def _sweep_loop(interval_seconds: float) -> None:
    """Run every recovery pass every ``interval_seconds`` for the life of the process. A
    failing pass is logged at ERROR and the others still run — a dead sweep is the silent
    abandonment it exists to prevent."""
    passes = (
        ("stalled-delivery sweep", sweep_stalled_deliveries),
        ("lapsed-intake re-drive", _redrive_lapsed_intakes),
        ("terminal-index prune", _prune_terminal_indexes),
    )
    while True:
        await asyncio.sleep(interval_seconds)
        for name, run_pass in passes:
            try:
                await run_pass()
            except Exception:
                logger.error("conversations: %s pass failed; retrying in %ss", name, interval_seconds, exc_info=True)


def _on_sweep_done(task: asyncio.Task[None]) -> None:
    """Surface an unexpected death of the sweep task at ERROR; a cancellation is silent."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("conversations: stalled-delivery sweep task died unexpectedly", exc_info=exc)


# -- task spawning -----------------------------------------------------------


def spawn_delivery(message_id: str) -> None:
    """Spawn the background delivery of an already-persisted record (fire-and-forget)."""
    _spawn(deliver(message_id))


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _DELIVERY_TASKS.add(task)
    task.add_done_callback(_on_task_done)


def _on_task_done(task: asyncio.Task[None]) -> None:
    _DELIVERY_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("conversations: delivery task failed", exc_info=exc)


__all__ = [
    "deliver",
    "mark_wait_delivered",
    "record_delivery_status",
    "redrive_pending",
    "spawn_delivery",
    "split_message",
    "start_delivery_sweep",
    "stop_delivery_sweep",
    "sweep_stalled_deliveries",
]
