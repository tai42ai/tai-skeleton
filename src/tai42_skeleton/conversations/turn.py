"""The turn engine — turns an accepted message into an agent turn.

One inbound message (channel door :func:`accept`, authed API door
:func:`submit_api_message`) resolves to its route, runs that route's agent IN-PROCESS under
the route's execution key, and persists the produced answer as a durable record the delivery
executor sends back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import quote
from uuid import uuid4

from tai42_contract.agent import Agent
from tai42_contract.agent.events import InterruptFinal, MessageFinal, StructuredFinal
from tai42_contract.conversations import AnswerStatus, ConversationAnswer, ConversationRoute

from tai42_skeleton.agent.thread_reservation import BRIDGE_THREAD_PREFIX
from tai42_skeleton.authz.execution import authorize_execution_agent_run, bind_execution_identity
from tai42_skeleton.conversations.address import canonical_address
from tai42_skeleton.conversations.cache import get_conversations_manager
from tai42_skeleton.conversations.caps import AddressAdmission, AddressRateLimitedError, TurnCaps, get_turn_caps
from tai42_skeleton.conversations.delivery import mark_wait_delivered, spawn_delivery
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations.errors import NotSupportedError, PermissionDenied

logger = logging.getLogger(__name__)

# Strong references to in-flight turn tasks so one is not GC'd before it persists.
_TURN_TASKS: set[asyncio.Task] = set()

# Client-safe text for a failed turn; the internal detail goes to the record's ``error``.
_ERROR_ANSWER_TEXT = "Sorry, something went wrong handling your message. Please try again."

# Delivered once per refill window to an address over its rate cap.
_SLOW_DOWN_TEXT = "You are sending messages faster than I can answer. Please wait a moment and try again."


class ConversationRouteResolutionError(LookupError):
    """No route matches the inbound message, so it is refused rather than dropped."""


class UnauthenticatedApiCallerError(NotSupportedError):
    """The API door was reached with no authenticated caller principal. The turn is refused:
    every thread and rate bucket on that door is keyed by its caller, and an anonymous one
    would be shared by everybody."""


@dataclass(frozen=True)
class ApiSubmitResult:
    """The outcome the API door turns into its HTTP response. ``answer`` is set only when
    the bounded sync-wait finished the turn in time (→ ``200``); otherwise the turn is
    still running behind the callback (→ ``202``)."""

    message_id: str
    thread_id: str
    answer: ConversationAnswer | None


def _thread_id(route_name: str, client_address: str) -> str:
    return f"{BRIDGE_THREAD_PREFIX}{route_name}:{client_address}"


def _api_client_address(caller_principal: str, address: str) -> str:
    """The API door's address slot: its authenticated caller joined to the caller-supplied
    end-user id. The principal is percent-encoded, so it holds no ``/`` and the join is
    unambiguous for any pair; two callers naming one end user get two addresses."""
    return f"{quote(caller_principal, safe='')}/{address}"


def _channel_bucket_key(route_name: str, address: str) -> str:
    """The channel door's rate-bucket key. The provider-attested address is the accountable
    party there, and the route scopes it so two routes never share a budget."""
    return f"{route_name}|{address}"


def _api_bucket_key(route_name: str, caller_principal: str) -> str:
    """The API door's rate-bucket key. It is the authenticated CALLER, not the composed
    address, whose cardinality the caller still chooses."""
    return f"{route_name}|caller:{caller_principal}"


def _store() -> ConversationRecordStore:
    return ConversationRecordStore(ConversationsSettings())


# -- route resolution --------------------------------------------------------


async def _resolve_channel_route(channel: str, our_identity_canonical: str) -> ConversationRoute:
    """The single ``door=channel`` route matching ``(channel, our_identity)`` by EXACT
    equality on the canonical address form. No match raises
    :class:`ConversationRouteResolutionError`; more than one is a corrupt table and raises
    rather than picking one."""
    routes = await get_conversations_manager().list_routes()
    matches = [
        route
        for route in routes.values()
        if route.door == "channel"
        and route.channel == channel
        and route.our_identity is not None
        and canonical_address(route.our_identity) == our_identity_canonical
    ]
    if not matches:
        raise ConversationRouteResolutionError(
            f"no channel route bound to channel {channel!r} identity {our_identity_canonical!r}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"conversations: {len(matches)} channel routes claim channel {channel!r} identity "
            f"{our_identity_canonical!r}; the routing table is inconsistent"
        )
    return matches[0]


# -- the turn ----------------------------------------------------------------


def _serialize_structured(data: object) -> str:
    if isinstance(data, str):
        return data
    model_dump_json = getattr(data, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())
    return json.dumps(data, default=str)


async def _drain_answer(agent: Agent, text: str, thread_id: str) -> str:
    """Run the agent to its terminal event and return the answer text. A structured final
    is serialized; an interrupt is not answerable by a background turn and is raised."""
    structured: StructuredFinal | None = None
    message: MessageFinal | None = None
    async for event in agent.astream(user_message=text, thread_id=thread_id):
        if isinstance(event, InterruptFinal):
            raise RuntimeError(f"agent raised an interrupt ({event.interrupt_id}) a background turn cannot answer")
        if isinstance(event, StructuredFinal):
            structured = event
        elif isinstance(event, MessageFinal):
            message = event
    if structured is not None:
        return _serialize_structured(structured.data)
    if message is not None:
        return message.text
    return ""


async def _run_agent_turn(route: ConversationRoute, text: str, thread_id: str) -> tuple[AnswerStatus, str, str | None]:
    """Run one turn as the route's execution key and return ``(answer_status, answer,
    error_detail)``. The identity is bound for the turn's duration and the run authorized
    against it before the agent runs. A denied run, a mid-turn error or an empty answer
    becomes a client-safe ``error`` outcome; its detail is returned, never delivered."""
    agent = _agent_registry().get(route.agent_name)
    if agent is None:
        return ("error", _ERROR_ANSWER_TEXT, f"agent {route.agent_name!r} is not registered")
    try:
        async with bind_execution_identity(
            route.execution_key, bound_fingerprint=route.execution_key_fingerprint
        ) as identity:
            await authorize_execution_agent_run(identity, route.agent_name)
            answer = await _drain_answer(agent, text, thread_id)
    except PermissionDenied as exc:
        return ("error", _ERROR_ANSWER_TEXT, f"turn denied: {exc}")
    except Exception as exc:
        # A failed turn becomes a logged error OUTCOME, not a swallowed error.
        logger.error("conversations: turn for route %r failed", route.route_name, exc_info=exc)
        return ("error", _ERROR_ANSWER_TEXT, f"turn error: {exc}")
    if not answer.strip():
        return ("error", _ERROR_ANSWER_TEXT, "agent produced an empty answer")
    return ("answered", answer, None)


def _agent_registry() -> dict[str, Agent]:
    from tai42_skeleton.app import instance

    return instance.app.agents.all_agents()


def _new_record(
    *,
    route: ConversationRoute,
    message_id: str,
    thread_id: str,
    client_address: str,
    caller_principal: str | None,
    provider_message_id: str | None,
    delivery_status: DeliveryStatus,
    answer_status: AnswerStatus | None = None,
    answer: str | None = None,
    error: str | None = None,
) -> ConversationRecord:
    """A freshly minted record for one accepted message, in the state its door commits it
    to (``accepted``, ``pending_delivery`` or ``shed``). An api-door record MUST name the
    authenticated caller its thread and rate bucket are keyed by; a channel-door record
    names none."""
    if (route.door == "api") != bool(caller_principal and caller_principal.strip()):
        raise RuntimeError(
            f"conversations: a {route.door} record cannot carry caller_principal={caller_principal!r}; "
            "the api door requires one and the channel door has none"
        )
    now = time.time()
    return ConversationRecord(
        message_id=message_id,
        route_name=route.route_name,
        door=route.door,
        thread_id=thread_id,
        client_address=client_address,
        channel=route.channel,
        our_identity=route.our_identity,
        callback_url=route.callback_url,
        caller_principal=caller_principal,
        provider_message_id=provider_message_id,
        delivery_status=delivery_status,
        answer_status=answer_status,
        answer=answer,
        error=error,
        created_at=now,
        updated_at=now,
    )


def _with_outcome(
    intake: ConversationRecord, answer_status: AnswerStatus, answer: str, error_detail: str | None
) -> ConversationRecord:
    """``intake`` carrying a produced outcome and moved to ``pending_delivery`` — the shape
    :meth:`ConversationRecordStore.complete_turn` requires."""
    return ConversationRecord.model_validate(
        intake.model_dump()
        | {
            "answer_status": answer_status,
            "answer": answer,
            "error": error_detail,
            "delivery_status": DeliveryStatus.PENDING_DELIVERY,
            "updated_at": time.time(),
        }
    )


async def _complete_turn(*, route: ConversationRoute, intake: ConversationRecord, text: str) -> ConversationRecord:
    """Run the turn and move its intake record to ``pending_delivery`` carrying the answer
    (persist before send); delivery is the caller's to spawn. The transition is guarded on
    the record still being at intake, so a turn finishing after a re-drive resolved its
    record raises rather than overwriting the outcome the client was given."""
    answer_status, answer, error_detail = await _run_agent_turn(route, text, intake.thread_id)
    completed = _with_outcome(intake, answer_status, answer, error_detail)
    outcome = await _store().complete_turn(completed)
    if outcome != 1:
        raise RuntimeError(
            f"conversations: record {intake.message_id} is no longer at intake "
            f"(complete_turn answered {outcome}); its outcome was resolved elsewhere and this turn's "
            "answer is discarded"
        )
    return completed


# -- shed outcomes (the address rate cap) ------------------------------------


async def _shed_with_reply(
    store: ConversationRecordStore,
    *,
    route: ConversationRoute,
    channel: str,
    message_id: str,
    thread_id: str,
    client_address: str,
    provider_message_id: str,
) -> str:
    """Answer an over-limit address with its one paid slow-down reply, committed in the
    turn path's order: the record is persisted at ``accepted`` under an intake lease, the
    inbound pair is claimed, and only then does the guarded transition make it deliverable.
    A record the delivery machine drives must never stand behind an unclaimed pair. No turn
    runs, so no thread slot is reserved."""
    intake_token = uuid4().hex
    intake = _new_record(
        route=route,
        message_id=message_id,
        thread_id=thread_id,
        client_address=client_address,
        caller_principal=None,
        provider_message_id=provider_message_id,
        delivery_status=DeliveryStatus.ACCEPTED,
    )
    try:
        await store.create_record(intake, intake_token=intake_token)
        owner = await store.claim_inbound(channel, provider_message_id, message_id)
    except asyncio.CancelledError:
        _spawn_intake_resolution(message_id)
        raise
    except Exception:
        # The claim may have been APPLIED with only its reply lost, so the record is
        # resolved against its inbound pair now instead of waiting out its intake lease.
        await _resolve_stranded_intake(message_id)
        raise
    if owner != message_id:
        await store.delete_record(message_id)
        return owner
    completed = _with_outcome(intake, "answered", _SLOW_DOWN_TEXT, None)
    outcome = await store.complete_turn(completed)
    if outcome != 1:
        raise RuntimeError(
            f"conversations: shed record {message_id} is no longer at intake (complete_turn answered {outcome}); "
            "its outcome was resolved elsewhere and the slow-down reply is discarded"
        )
    spawn_delivery(message_id)
    return message_id


async def _shed_silently(
    store: ConversationRecordStore,
    *,
    route: ConversationRoute,
    channel: str,
    message_id: str,
    thread_id: str,
    client_address: str,
    provider_message_id: str,
) -> str:
    """Drop a message from an address already given its slow-down reply this window,
    leaving a terminal ``shed`` record. The claim behind that record is what makes a
    provider redelivery resolve to it instead of buying the address another turn."""
    record = _new_record(
        route=route,
        message_id=message_id,
        thread_id=thread_id,
        client_address=client_address,
        caller_principal=None,
        provider_message_id=provider_message_id,
        delivery_status=DeliveryStatus.SHED,
        error=f"address {client_address!r} was over its rate cap after a prior slow-down reply",
    )
    await store.create_record(record)
    owner = await store.claim_inbound(channel, provider_message_id, message_id)
    if owner != message_id:
        await store.delete_record(message_id)
        return owner
    logger.warning(
        "conversations: address %r on route %r is over its rate cap; message dropped after a prior slow-down reply",
        client_address,
        route.route_name,
    )
    return message_id


# -- the channel door: accept ------------------------------------------------


async def accept(channel: str, our_identity: str, client_address: str, text: str, provider_message_id: str) -> str:
    """Accept one inbound channel message, persist-and-deliver its answer, and return its
    ``message_id`` (a uuid4). See :meth:`AppConversations.accept`.

    Idempotent on ``(channel, provider_message_id)``: a redelivery returns the existing
    ``message_id`` and starts no second turn. Every gate that can refuse runs before any
    state is written, so a refusal leaves the pair unclaimed. The turn runs in the
    background; the caller gets the id immediately."""
    channel_identity = canonical_address(our_identity)
    address = canonical_address(client_address)
    route = await _resolve_channel_route(channel, channel_identity)
    thread_id = _thread_id(route.route_name, address)
    store = _store()

    owner = await store.get_inbound_owner(channel, provider_message_id)
    if owner is not None:
        # Redelivery of a message already accepted: return the prior turn's id.
        return owner

    message_id = str(uuid4())
    admission = get_turn_caps().admit_address(_channel_bucket_key(route.route_name, address))
    if admission is AddressAdmission.SHED_WITH_REPLY:
        return await _shed_with_reply(
            store,
            route=route,
            channel=channel,
            message_id=message_id,
            thread_id=thread_id,
            client_address=address,
            provider_message_id=provider_message_id,
        )
    if admission is AddressAdmission.SHED_SILENT:
        return await _shed_silently(
            store,
            route=route,
            channel=channel,
            message_id=message_id,
            thread_id=thread_id,
            client_address=address,
            provider_message_id=provider_message_id,
        )

    return await _accept_for_turn(
        store,
        route=route,
        channel=channel,
        message_id=message_id,
        thread_id=thread_id,
        client_address=address,
        text=text,
        provider_message_id=provider_message_id,
    )


async def _accept_for_turn(
    store: ConversationRecordStore,
    *,
    route: ConversationRoute,
    channel: str,
    message_id: str,
    thread_id: str,
    client_address: str,
    text: str,
    provider_message_id: str,
) -> str:
    """Commit an admitted channel message to a turn in the one order that keeps the
    release-less inbound claim sound: reserve the per-thread FIFO slot (the last gate that
    can refuse, and it refuses with nothing written), persist the intake record, claim the
    inbound pair, schedule the turn. Losing the claim means a concurrent attempt committed
    first, so this one releases its slot, discards its record and returns the winner's id."""
    caps = get_turn_caps()
    caps.reserve_thread_slot(thread_id)
    intake_token = uuid4().hex
    intake = _new_record(
        route=route,
        message_id=message_id,
        thread_id=thread_id,
        client_address=client_address,
        caller_principal=None,
        provider_message_id=provider_message_id,
        delivery_status=DeliveryStatus.ACCEPTED,
    )
    try:
        await store.create_record(intake, intake_token=intake_token)
        owner = await store.claim_inbound(channel, provider_message_id, message_id)
    except asyncio.CancelledError:
        # A cancelled task cannot await the round-trips the resolution needs, so it is
        # handed to a fresh task.
        caps.release_thread_slot(thread_id)
        _spawn_intake_resolution(message_id)
        raise
    except Exception:
        # The claim may have been APPLIED with only its reply lost, so the record is
        # resolved against its inbound pair now instead of waiting out its intake lease.
        caps.release_thread_slot(thread_id)
        await _resolve_stranded_intake(message_id)
        raise
    if owner != message_id:
        caps.release_thread_slot(thread_id)
        await store.delete_record(message_id)
        return owner

    _schedule_turn(caps, route=route, intake=intake, text=text, intake_token=intake_token, deliver_on_completion=True)
    return message_id


# -- the authed API door -----------------------------------------------------


async def submit_api_message(
    route_name: str, external_user_id: str, text: str, caller_principal: str | None, wait_seconds: int
) -> ApiSubmitResult:
    """Accept one authed API-door message and run its turn.

    ``wait_seconds`` (clamped to ``sync_wait_max_seconds`` by the door, ``0`` for the
    pure-async path) bounds a sync wait: a turn that finishes inside it answers in the
    ``200`` with the callback suppressed, otherwise the door returns ``202`` and the answer
    is POSTed to the callback.

    Admission runs in the channel door's order — rate cap, thread reservation, intake
    record — so a refusal writes nothing and a returned ``message_id`` always names a
    durable record. ``external_user_id`` is matched VERBATIM after a whitespace trim: two
    spellings are two threads.

    ``caller_principal`` is MANDATORY: it qualifies the thread (so no caller can reach
    another's conversation memory by naming its ``external_user_id``) and it alone keys the
    rate bucket (so the cap bounds the accountable party, not a value the caller picks)."""
    if caller_principal is None or not caller_principal.strip():
        raise UnauthenticatedApiCallerError(
            f"api conversation route {route_name!r} needs an authenticated caller principal and this "
            "deployment resolved none; the api door requires access control to be enabled"
        )
    address = canonical_address(external_user_id)
    route = await _get_api_route(route_name)
    client_address = _api_client_address(caller_principal, address)
    thread_id = _thread_id(route.route_name, client_address)
    message_id = str(uuid4())

    caps = get_turn_caps()
    admission = caps.admit_address(_api_bucket_key(route.route_name, caller_principal))
    if admission is not AddressAdmission.ADMIT:
        raise AddressRateLimitedError(
            f"caller {caller_principal!r} is over its rate cap of "
            f"{caps.settings.per_address_turns_per_hour}/hour on route {route.route_name!r}; "
            "retry after a short wait"
        )

    caps.reserve_thread_slot(thread_id)
    intake_token = uuid4().hex
    intake = _new_record(
        route=route,
        message_id=message_id,
        thread_id=thread_id,
        client_address=client_address,
        caller_principal=caller_principal,
        provider_message_id=None,
        delivery_status=DeliveryStatus.ACCEPTED,
    )
    try:
        await _store().create_record(intake, intake_token=intake_token)
    except BaseException:
        caps.release_thread_slot(thread_id)
        raise

    task = _schedule_turn(
        caps, route=route, intake=intake, text=text, intake_token=intake_token, deliver_on_completion=False
    )

    if wait_seconds > 0:
        done, _pending = await asyncio.wait({task}, timeout=wait_seconds)
        if task in done and task.exception() is None:
            record = task.result()
            if await mark_wait_delivered(message_id):
                return ApiSubmitResult(message_id=message_id, thread_id=thread_id, answer=record.answer_payload())
            # Lost the claim to a racing delivery — fall through to the async shape.

    # Async path: attach the delivery spawn to the task's completion, so exactly one of the
    # wait path and the callback delivers.
    _deliver_when_done(task, message_id)
    return ApiSubmitResult(message_id=message_id, thread_id=thread_id, answer=None)


async def _get_api_route(route_name: str) -> ConversationRoute:
    route = await get_conversations_manager().get_route(route_name)
    if route is None or route.door != "api":
        raise ConversationRouteResolutionError(f"no api conversation route named {route_name!r}")
    return route


# -- turn scheduling under the caps ------------------------------------------


def _schedule_turn(
    caps: TurnCaps,
    *,
    route: ConversationRoute,
    intake: ConversationRecord,
    text: str,
    intake_token: str,
    deliver_on_completion: bool,
) -> asyncio.Task[ConversationRecord]:
    """Schedule ``intake``'s turn as a background task consuming the caller's reservation;
    returns the task whose result is the completed :class:`ConversationRecord`.

    The intake lease is refreshed OUTSIDE the caps, so a turn queued behind the FIFO reads
    as live too. ``caps`` MUST be the instance the caller reserved on, or the reservation is
    released on a different instance and the slot leaks."""

    async def _run() -> ConversationRecord:
        async with _intake_lease_held(intake.message_id, intake_token), caps.run_reserved(intake.thread_id):
            return await _complete_turn(route=route, intake=intake, text=text)

    task = asyncio.create_task(_run())
    _TURN_TASKS.add(task)
    if deliver_on_completion:
        task.add_done_callback(lambda t: _spawn_delivery_on_success(t, intake.message_id))
    else:
        task.add_done_callback(_TURN_TASKS.discard)
    return task


@contextlib.asynccontextmanager
async def _intake_lease_held(message_id: str, token: str) -> AsyncIterator[None]:
    """Refresh ``message_id``'s intake lease for the body's duration, so the intake re-drive
    reads the turn as LIVE and leaves the record to this worker — whether it is running or
    still queued behind the caps."""
    refresher = asyncio.create_task(_refresh_intake_lease(message_id, token))
    try:
        yield
    finally:
        refresher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresher


async def _refresh_intake_lease(message_id: str, token: str) -> None:
    """Re-take the intake lease every ``intake_claim_refresh_seconds`` until the record
    leaves intake or the lease is lost. A refresh that fails is logged and retried — a
    heartbeat that died quietly would let a live turn be reaped as stranded."""
    store = _store()
    settings = store.settings
    while True:
        await asyncio.sleep(settings.intake_claim_refresh_seconds)
        try:
            held = await store.claim_intake(message_id, time.time(), token, settings.intake_claim_lease_seconds)
        except Exception:
            logger.error(
                "conversations: refreshing the intake lease on record %s failed; retrying in %ss",
                message_id,
                settings.intake_claim_refresh_seconds,
                exc_info=True,
            )
            continue
        if held != 1:
            logger.warning(
                "conversations: record %s no longer holds this worker's intake lease (claim returned %d); its "
                "outcome is another worker's to write",
                message_id,
                held,
            )
            return


def _deliver_when_done(task: asyncio.Task[ConversationRecord], message_id: str) -> None:
    """Spawn the record's delivery when its turn task completes successfully."""
    if task.done():
        _spawn_delivery_on_success(task, message_id)
        return
    task.add_done_callback(lambda t: _spawn_delivery_on_success(t, message_id))


def _spawn_delivery_on_success(task: asyncio.Task[ConversationRecord], message_id: str) -> None:
    _TURN_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "conversations: turn task for record %s failed before it wrote an outcome; the record is given an "
            "error outcome and the turn is not re-run",
            message_id,
            exc_info=exc,
        )
        _spawn_intake_resolution(message_id)
        return
    spawn_delivery(message_id)


# -- resolving a record left mid-turn ----------------------------------------


def _spawn_intake_resolution(message_id: str) -> None:
    """Resolve a record this worker left at intake, in this worker, now — this worker owns
    it, and waiting out its intake lease would hold the message unanswered for that long."""
    task = asyncio.create_task(_resolve_stranded_intake(message_id))
    _TURN_TASKS.add(task)
    task.add_done_callback(_on_intake_resolution_done)


def _on_intake_resolution_done(task: asyncio.Task[None]) -> None:
    _TURN_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("conversations: resolving a record left at intake by a failed turn task failed", exc_info=exc)


async def _resolve_stranded_intake(message_id: str) -> None:
    """Resolve a record this worker left at intake — an interrupted commit or a turn task
    that died. A record that has already left intake keeps the outcome it carries; one still
    at intake is arbitrated against its inbound pair."""
    store = _store()
    record = await store.get_record(message_id)
    if record is None:
        logger.warning(
            "conversations: record %s is gone, so the turn that failed on it leaves nothing to resolve", message_id
        )
        return
    if record.delivery_status is not DeliveryStatus.ACCEPTED:
        logger.info(
            "conversations: record %s already carries a %s outcome; the turn task's failure resolves nothing",
            message_id,
            record.delivery_status.value,
        )
        return
    await _arbitrate_stranded_intake(store, record)


async def _arbitrate_stranded_intake(store: ConversationRecordStore, record: ConversationRecord) -> None:
    """Resolve one record left at intake against its inbound pair: the record the pair is
    committed to takes the error outcome and delivers it; one that lost the pair to another
    attempt owns nothing and is discarded."""
    if await _owns_inbound_claim(store, record):
        await _fail_stranded_turn(store, record)
        return
    await store.delete_record(record.message_id)
    logger.warning(
        "conversations: intake record %s lost the inbound claim for %r on channel %r to another attempt and was "
        "discarded",
        record.message_id,
        record.provider_message_id,
        record.channel,
    )


async def redrive_accepted() -> None:
    """Resolve every record left in ``accepted`` by a worker that DIED mid-turn.

    The intake lease is the liveness test and it is taken FIRST: a record whose lease is
    still live belongs to a turn running on a sibling worker and is left untouched, so the
    sweep never reaps an in-flight turn. Only a record whose lease has LAPSED is adopted,
    then arbitrated against the inbound claim (a get-or-set): one the claim names someone
    else for is discarded. An adopted record takes the ``error`` outcome and its turn is
    never re-run — a turn dispatches authorized tools, so it is not idempotent."""
    store = _store()
    token = uuid4().hex
    for record in await store.list_by_status(frozenset({DeliveryStatus.ACCEPTED})):
        try:
            adopted = await store.claim_intake(
                record.message_id, time.time(), token, store.settings.intake_claim_lease_seconds
            )
            if adopted != 1:
                logger.info(
                    "conversations: intake record %s was not adopted by the re-drive (claim returned %d); its turn "
                    "is live on another worker, or its outcome has already landed",
                    record.message_id,
                    adopted,
                )
                continue
            await _arbitrate_stranded_intake(store, record)
        except Exception:
            # One failing record must not abandon every other stranded record in the pass;
            # the next sweep re-drives this one.
            logger.error(
                "conversations: re-driving stranded intake record %s failed; skipped this pass",
                record.message_id,
                exc_info=True,
            )
            continue


async def _owns_inbound_claim(store: ConversationRecordStore, record: ConversationRecord) -> bool:
    """Whether ``record`` is the one its inbound pair is committed to. An api-door record
    has no provider id to dedupe on, so it is its own authority."""
    if record.channel is None or record.provider_message_id is None:
        return True
    owner = await store.claim_inbound(record.channel, record.provider_message_id, record.message_id)
    return owner == record.message_id


async def _fail_stranded_turn(store: ConversationRecordStore, record: ConversationRecord) -> None:
    """Give an intake record the error outcome its interrupted turn never produced and
    spawn its delivery — the one resolution both the in-process watcher and the periodic
    re-drive apply. Losing the guarded transition leaves the existing outcome standing."""
    completed = _with_outcome(record, "error", _ERROR_ANSWER_TEXT, "turn was interrupted before it produced an answer")
    outcome = await store.complete_turn(completed)
    if outcome != 1:
        logger.warning(
            "conversations: intake record %s left intake while it was being re-driven (complete_turn answered %d); "
            "its outcome stands as written",
            record.message_id,
            outcome,
        )
        return
    logger.error(
        "conversations: record %s was stranded mid-turn; the turn is NOT re-run and a client-safe error outcome "
        "is delivered instead",
        record.message_id,
    )
    spawn_delivery(record.message_id)


__all__ = [
    "ApiSubmitResult",
    "ConversationRouteResolutionError",
    "UnauthenticatedApiCallerError",
    "accept",
    "redrive_accepted",
    "submit_api_message",
]
