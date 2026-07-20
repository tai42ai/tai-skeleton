"""HTTP routes for the ask_user interactions surface — ``/api/interactions/*``.

Four doors:

* ``GET /api/interactions/stream`` — authenticated SSE feed: the pending-question
  backlog on connect, a ``backlog_done`` marker, then a live tail of add/answered/
  removed events.
* ``POST /api/interactions/{interaction_id}/answer`` — the authenticated human
  answer door. The value is validated server-side against the stored question's
  ``answer_format`` before the blocked caller is woken; an invalid answer is
  rejected loudly and the caller stays blocked. An EXTERNAL question is answered
  through its callback URL, never here.
* ``POST /api/interactions/callback/{ticket}`` — the UNAUTHENTICATED data door
  for external-format answers (the server-to-server / confirm-form claim path).
  Sensitive data rides the JSON body here.
* ``GET /api/interactions/callback/{ticket}`` — the UNAUTHENTICATED redirect
  door. GET never mutates state (link scanners prefetch these URLs); it serves a
  byte-constant page: for confirm/external questions the confirm page whose form
  POSTs back to the same URL, for text/select an informational awaiting-reply
  page (a bare confirm tap carries no value answer).

The callback ticket is a bearer capability minted by the ``ask_user`` helper;
it is never deleted, single-use is enforced by the answered-state guard in
``record_answer``, and a duplicate callback resolves idempotently to 200. The
callback door is NOT audience-gated — the ticket IS the authorization (an
external-answer flow deliberately addresses outsiders). That is sound because the
ticket is delivered ONLY over the configured channel: no stream or read frame ever
carries it (``_add_data`` emits no ``ticket`` field), so a restricted caller
filtered out of a question's stream can never obtain that question's ticket, and
the answer-door audience gate cannot be bypassed through this door.

Success bodies are ``{"data": {...}}``; failures are ``{"error": "<message>"}``.
Add frames are at-least-once across the cursor/backlog window — an add landing
between cursor capture and the backlog snapshot can appear in both the backlog
and the tail; clients de-duplicate by ``interaction_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from tai_contract.app import tai_app
from tai_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
    InteractionState,
)
from tai_contract.webhooks import WebhookVerificationError
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.access_control.user import request_identity
from tai_skeleton.app.http import http_surface
from tai_skeleton.app.route_registry import DeclaredRouteMetadata
from tai_skeleton.interactions.settings import InteractionsSettings, interactions_settings
from tai_skeleton.interactions.store import (
    ADD_EVENT,
    ANSWERED_EVENT,
    REMOVED_EVENT,
    InteractionStore,
    as_str,
)
from tai_skeleton.operations import (
    BadRequestError,
    PayloadTooLargeError,
    operation_metadata_of,
    register_operation_route,
)

# The human answer door is an operation in ``tai_skeleton.operations.interactions``;
# the still-handler callback door shares its answer-validation, reply-TTL, and
# serializer-guarded claim helpers, imported from that module.
from tai_skeleton.operations.interactions import (
    _AnswerInvalid,
    _claim_or_serialization_error,
    _reply_ttl,
    _schema_mismatch,
    _validate_answer,
)
from tai_skeleton.operations.interactions import answer_interaction as _answer_interaction_op

logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15


def _now() -> float:
    """The monotonic loop clock the keepalive deadline reads. A module-level seam so
    a test can drive the deadline without real wall-clock waits."""
    return asyncio.get_running_loop().time()


# no-store + nosniff ride EVERY callback response: capability-bearing URLs must
# never be cached, and the schema-mismatch 400 reflects attacker-influenced
# content that browsers reach via the confirm flow.
_BASE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
# HTML pages add the anti-injection headers for the platform's only
# unauthenticated HTML route.
_HTML_HEADERS = {
    **_BASE_HEADERS,
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

# Byte-constant pages: zero interpolation of any request-derived value. The
# confirm form posts to the SAME URL (empty action preserves the query string).
_CONFIRM_PAGE = (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8"><title>Confirm</title>\n'
    "<style>body{font-family:system-ui,sans-serif;margin:3rem;text-align:center}"
    "button{font-size:1rem;padding:.6rem 1.4rem}</style></head><body>\n"
    "<h1>Confirm</h1>\n"
    "<p>Click confirm to submit your response.</p>\n"
    '<form method="post"><button type="submit">Confirm</button></form>\n'
    "</body></html>\n"
)
_DONE_PAGE = (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8"><title>Done</title>\n'
    "<style>body{font-family:system-ui,sans-serif;margin:3rem;text-align:center}</style></head><body>\n"
    "<h1>Done</h1>\n"
    "<p>This interaction has already been answered.</p>\n"
    "</body></html>\n"
)
# Served on GET for a ticketed text/select question (channel delivery mints a
# ticket for every format): a value answer is required, so the page presents no
# form — the confirm button's empty-body POST could never succeed here.
_REPLY_PAGE = (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8"><title>Awaiting reply</title>\n'
    "<style>body{font-family:system-ui,sans-serif;margin:3rem;text-align:center}</style></head><body>\n"
    "<h1>Awaiting your reply</h1>\n"
    "<p>Answer this question by replying on the channel where you received it.</p>\n"
    "</body></html>\n"
)

# The free-form label recorded on an answer delivered through the callback door.
_EXTERNAL_ANSWERED_BY = "external-callback"

# A ``post_only`` (body-signature) verifier authenticates the raw body only, so an
# empty-body POST carries no signed answer — the query string is unauthenticated
# and must never supply the answer. The deny is a constant, request-independent
# message: the answer must ride the signed body.
_POST_ONLY_EMPTY_BODY_DENY = "verified callback requires a signed request body"


class _PayloadTooLarge(Exception):
    """Raised when a request body or query string exceeds the configured cap."""


# -- shared helpers ----------------------------------------------------------


def _callback_json(payload: dict, status_code: int) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers=dict(_BASE_HEADERS))


# Invalid JSON in an untrusted body converts to a loud 400; any exception outside
# this set is a server bug and propagates as a 500. RecursionError covers a
# deeply-nested body blowing up the parser; ``json.JSONDecodeError`` is a
# ``ValueError``.
_JSON_PARSE_ERRORS = (ValueError, RecursionError)


# -- SSE stream (route 1) ----------------------------------------------------


def _frame(event: str, data: dict) -> str:
    # ``json.dumps`` of the whole payload is what stops an attacker-supplied
    # answer (with newlines / ``data:`` sequences) from injecting extra frames.
    # Frame values are JSON-native by construction (they round-trip through the
    # contract models), so a serialization failure is a server bug and raises.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _add_data(request: InteractionRequest) -> dict:
    # A verifier config rides ``format_payload`` server-side; STRIP it from the
    # client frame (the browser never needs the verifier name / secret_env) and
    # in its place emit ``server_verified`` so the UI renders a non-actionable
    # "awaiting a verified server callback" card instead of a dead confirm link.
    format_payload = request.format_payload
    server_verified = False
    if format_payload is not None and "verifier" in format_payload:
        format_payload = {k: v for k, v in format_payload.items() if k != "verifier"}
        server_verified = True

    data = {
        "interaction_id": request.interaction_id,
        "group_id": request.group_id,
        "question": request.question,
        "answer_format": request.answer_format.value,
        "format_payload": format_payload,
        "created_at": request.created_at.isoformat(),
        "timeout_at": request.timeout_at.isoformat(),
        # Rides every add frame (backlog + live tail) so the UI can label the
        # answered state of a sensitive question, whose body is never persisted.
        "sensitive": request.sensitive,
    }
    if server_verified:
        data["server_verified"] = True
    if request.channel is not None:
        data["channel"] = request.channel
    # Display-only media rides the add frame as plain JSON dicts when present
    # (absent — no key — when the question has none); ``exclude_none`` keeps a
    # caption-less item lean, and the client treats a missing caption as absent.
    if request.media is not None:
        data["media"] = [item.model_dump(mode="json", exclude_none=True) for item in request.media]
    return data


async def _stream_events(request: Request, store: InteractionStore, settings: InteractionsSettings):
    # Resolve the caller's isolation identity once. A RESTRICTED caller (owner claim
    # present) sees ONLY interactions addressed to it: backlog/add frames are filtered
    # on the record's ``audience == <own id>``. ``answered``/``removed`` frames carry
    # no audience and the record may already be gone, so the connection keeps a
    # ``visible`` set of the interaction_ids it emitted an ``audience==self`` frame
    # for and emits a later ``answered``/``removed`` ONLY when its id is in that set —
    # failing CLOSED for anything else (never leaking another identity's interaction
    # id or lifecycle timing) while never dropping a frame for the caller's own
    # addressed interaction. An UNRESTRICTED caller sees every frame (today's
    # operator inbox). The interactions backlog is the FULL pending index (not a
    # bounded window), so — unlike tool runs — no completeness truncation exists and
    # a plain per-frame filter suffices.
    _user_id, restricted_id = request_identity()
    restricted = restricted_id is not None
    visible: set[str] = set()

    async with client_ctx(RedisClient, settings.redis) as r:
        # Capture the events cursor BEFORE the backlog read so no live event that
        # arrives during the backlog is missed. Empty stream -> tail from "0-0"
        # ("$" would drop an event written before the first XREAD call).
        tail = await r.xrevrange(store.events_key, count=1)
        cursor = as_str(tail[0][0]) if tail else "0-0"

        # The store owns the backlog read: pending-group order, per-group batched
        # state reads (no N+1), and the phantom/abandoned reconciliation side
        # effects. Read the WHOLE backlog under the pooled connection, then exit
        # the block to return the connection BEFORE yielding — a slow SSE client
        # suspends the generator between frames and must never pin the shared pool.
        backlog = await store.backlog(r)

    for req in backlog:
        if restricted and req.audience != restricted_id:
            continue
        if restricted:
            visible.add(req.interaction_id)
        yield _frame(ADD_EVENT, _add_data(req))
    yield _frame("interaction.backlog_done", {})

    # The tail blocks ~15s per iteration; a dedicated fresh connection keeps it
    # off the shared pool the answer door needs. The socket read timeout is
    # stripped on this connection only — the keepalive XREAD blocks legitimately
    # for the keepalive window, which a blanket 5s read timeout would kill; the
    # outer wait_for below bounds a black-holed redis instead.
    tail_redis = settings.redis.model_copy(update={"socket_timeout": None})
    # Keepalive cadence is governed by a monotonic loop-clock DEADLINE, not by
    # whether a given XREAD returned events. Any global-stream event (including one
    # filtered out for a restricted caller) makes XREAD return early; tying the
    # keepalive to that would let a restricted caller infer other identities'
    # activity timing from keepalive jitter. Instead XREAD blocks only until the
    # next deadline, the keepalive fires when the deadline passes, and the deadline
    # resets only on a frame actually yielded to THIS caller — so the cadence is
    # identical whether or not other identities are active.
    next_keepalive = _now() + _KEEPALIVE_SECONDS
    async with client_ctx(RedisClient, tail_redis, fresh=True) as tail_conn:
        while True:
            if await request.is_disconnected():
                break
            block_seconds = max(0.0, next_keepalive - _now())
            try:
                result = await asyncio.wait_for(
                    tail_conn.xread({store.events_key: cursor}, block=max(1, int(block_seconds * 1000))),
                    timeout=block_seconds + settings.blocking_grace_seconds,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "interactions SSE tail: redis XREAD did not return within the keepalive "
                    f"window + {settings.blocking_grace_seconds}s grace — connection presumed stalled"
                ) from exc
            yielded = False
            for _stream, messages in result or ():
                for message_id, data in messages:
                    cursor = as_str(message_id)
                    fields = {as_str(k): as_str(v) for k, v in data.items()}
                    event_type = fields.get("type")
                    interaction_id = fields.get("interaction_id")
                    group_id = fields.get("group_id")
                    if event_type is None or interaction_id is None or group_id is None:
                        # A malformed entry (partial XADD, older/newer schema,
                        # seeded frame) is skipped, never fatal — one bad event
                        # must not tear down the whole SSE tail.
                        logger.debug("skipping malformed stream event %s: missing required field", cursor)
                        continue
                    if event_type == ADD_EVENT:
                        state = await store.get_state(tail_conn, interaction_id)
                        # A state pruned/expired between the event and this read
                        # has nothing left to show — the add frame is skipped,
                        # matching the backlog's pending-only filter.
                        if state is None:
                            continue
                        if restricted and state.request.audience != restricted_id:
                            continue
                        if restricted:
                            visible.add(interaction_id)
                        yield _frame(ADD_EVENT, _add_data(state.request))
                        yielded = True
                    elif event_type in (ANSWERED_EVENT, REMOVED_EVENT):
                        # Fail closed: a restricted caller is emitted a terminal
                        # frame only for an interaction it saw an addressed add for.
                        if restricted and interaction_id not in visible:
                            continue
                        yield _frame(event_type, {"interaction_id": interaction_id, "group_id": group_id})
                        yielded = True
                        # An interaction fires exactly one terminal event, and its id
                        # is never reused — drop it from the visible set so the set
                        # stays bounded to currently-open interactions. A later
                        # duplicate terminal frame then fails closed (suppressed).
                        visible.discard(interaction_id)
            # The keepalive is deadline-driven: it fires whenever the monotonic
            # deadline passes and no frame reached THIS caller this window — whether
            # XREAD returned nothing OR only events filtered out for a restricted
            # caller. A frame that DID reach the caller restarts the countdown (it
            # doubles as liveness); a window of only filtered events leaves the
            # deadline untouched, so the caller's cadence stays independent of other
            # identities' volume and the connection never goes silent.
            if yielded:
                next_keepalive = _now() + _KEEPALIVE_SECONDS
            elif _now() >= next_keepalive:
                yield ": keepalive\n\n"
                next_keepalive = _now() + _KEEPALIVE_SECONDS


@http_surface().custom_route(
    "/api/interactions/stream",
    methods=["GET"],
    summary="Stream the interactions inbox (backlog then live)",
    tags=["interactions"],
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(401,),
        success_status=200,
    ),
)
async def stream(request: Request) -> Response:
    settings = interactions_settings()
    store = InteractionStore(settings.key_prefix)
    return StreamingResponse(
        _stream_events(request, store, settings),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# -- human answer door (route 2) — an operation adapter ----------------------


async def _extract_answer(request: Request) -> dict:
    """Read + bound the human-answer body at the HTTP edge into the operation's flat
    ``answer`` argument. The byte cap (413), invalid JSON (400), and a missing
    ``answer`` key (400) are the same loud rejections the door has always answered —
    reproduced here so the operation receives an already-parsed answer value (the
    adapter's plain parse would yield 422)."""
    settings = interactions_settings()
    try:
        raw = await _read_bounded_body(request, settings.callback_max_body_bytes)
    except _PayloadTooLarge as exc:
        raise PayloadTooLargeError("payload too large") from exc
    try:
        body = json.loads(raw)
    except _JSON_PARSE_ERRORS as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict) or "answer" not in body:
        raise BadRequestError("body must contain 'answer'")
    return {"answer": body["answer"]}


answer = register_operation_route(
    tai_app,
    operation_metadata_of(_answer_interaction_op),
    path="/api/interactions/{interaction_id}/answer",
    method="POST",
    context_extractor=_extract_answer,
)


# -- callback doors (routes 3 & 4) -------------------------------------------


async def _read_bounded_body(request: Request, cap: int) -> bytes:
    """Read the request body on ACTUAL bytes, never a client ``Content-Length``.
    Raise ``_PayloadTooLarge`` past ``cap`` before parsing — loud, never truncated."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise _PayloadTooLarge("request body exceeds the configured cap")
        chunks.append(chunk)
    return b"".join(chunks)


def _params_to_answer(request: Request) -> dict:
    """The query params as the delivered answer: a single occurrence yields the
    scalar string (``?a=1`` -> ``{"a": "1"}``), a repeated key yields a list
    (``?tag=a&tag=b`` -> ``{"tag": ["a", "b"]}``) — never silent last-wins."""
    result: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[key] = [existing, value]
        else:
            result[key] = value
    return result


async def _record_callback_answer(
    r: Any,
    store: InteractionStore,
    settings: InteractionsSettings,
    ticket: str,
    interaction_id: str,
    state: InteractionState,
    answer: Any,
) -> JSONResponse:
    """Atomically claim ``answer`` for the ticketed question. A lost race maps
    to the same idempotent 200 already_answered."""
    response = InteractionResponse(
        interaction_id=interaction_id,
        answer=answer,
        answered_by=_EXTERNAL_ANSWERED_BY,
        answered_at=datetime.now(UTC),
    )
    claimed = await _claim_or_serialization_error(
        store,
        r,
        response,
        state.group_id,
        _reply_ttl(state.request),
        ticket=ticket,
        ticket_ttl=settings.idle_ttl_seconds,
    )
    if claimed is None:
        return _callback_json({"error": "answer payload could not be serialized"}, 400)
    if not claimed:
        return _callback_json({"data": {"status": "already_answered"}}, 200)
    return _callback_json({"data": {"interaction_id": interaction_id, "status": "answered"}}, 200)


async def _claim_external(
    r: Any,
    store: InteractionStore,
    settings: InteractionsSettings,
    ticket: str,
    interaction_id: str,
    state: InteractionState,
    answer: Any,
) -> JSONResponse:
    """Validate (if a schema was declared), then atomically claim the answer."""
    schema = (state.request.format_payload or {}).get("schema")
    if schema is not None:
        message = _schema_mismatch(answer, schema)
        if message is not None:
            return _callback_json({"error": message}, 400)
    return await _record_callback_answer(r, store, settings, ticket, interaction_id, state, answer)


def _callback_verifier(state: InteractionState) -> dict | None:
    """The verifier binding stashed in the external ``format_payload`` (``{"name",
    "config"}``), or ``None`` for an unbound ticket-only external question."""
    binding = (state.request.format_payload or {}).get("verifier")
    return binding if isinstance(binding, dict) else None


async def _verify_callback(request: Request, raw: bytes, state: InteractionState) -> tuple[Response | None, bool]:
    """Run the question's bound verifier over the raw callback body. Return
    ``(deny_response, False)`` on any failure (nothing recorded, ticket
    unconsumed), or ``(None, post_only)`` when the question is unbound or
    verification passes — ``post_only`` tells the caller whether the verifier
    signs only the body (True for a body-signature verifier), so an empty-body
    POST must not draw its answer from the unauthenticated query string.

    The unbound question returns ``(None, False)`` so its ticket-only query-param
    path stays open; only a passing body-signature verifier returns
    ``(None, True)``.

    Fails CLOSED: a signature failure -> 401 (constant message); an unknown
    verifier name / missing secret env / verifier bug -> 500."""
    binding = _callback_verifier(state)
    if binding is None:
        return None, False
    name = binding.get("name")
    config = binding.get("config") or {}
    interaction_id = state.request.interaction_id
    if not isinstance(name, str):
        logger.error("callback verify: malformed verifier binding (no name) for interaction %s", interaction_id)
        return _callback_json({"error": "webhook verification error"}, 500), False
    try:
        verifier = tai_app.webhook_verifiers.get(name)
    except Exception:
        logger.error("callback verify: no registered verifier %r for interaction %s", name, interaction_id)
        return _callback_json({"error": "webhook verification error"}, 500), False
    post_only = bool(getattr(verifier, "post_only", False))
    try:
        await verifier.verify(raw, request.headers, config)
    except WebhookVerificationError as exc:
        # Log the reason (never the raw body verbatim); return a constant message.
        logger.warning("callback verification failed for interaction %s: %s", interaction_id, exc)
        return _callback_json({"error": "webhook verification failed"}, 401), False
    except Exception:
        logger.error("callback verifier error for interaction %s", interaction_id, exc_info=True)
        return _callback_json({"error": "webhook verification error"}, 500), False
    return None, post_only


async def _callback_post(request: Request, r: Any, store: InteractionStore, settings: InteractionsSettings) -> Response:
    ticket = request.path_params["ticket"]

    # Size caps: the raw query string (the confirm-flow answer rides the URL) and
    # the actual body bytes both obey ``callback_max_body_bytes``.
    if len(request.url.query.encode()) > settings.callback_max_body_bytes:
        return _callback_json({"error": "payload too large"}, 413)
    try:
        raw = await _read_bounded_body(request, settings.callback_max_body_bytes)
    except _PayloadTooLarge:
        return _callback_json({"error": "payload too large"}, 413)

    interaction_id = await store.resolve_ticket(r, ticket)
    if interaction_id is None:
        # Uniform 404: same status and body whether the ticket never existed or
        # expired — no timing or body distinction.
        return _callback_json({"error": "not found"}, 404)
    state = await store.get_state(r, interaction_id)
    if state is None:
        # A cancel/timeout prune deletes the state while the ticket lives out its
        # TTL — the same uniform 404, never a None dereference.
        return _callback_json({"error": "not found"}, 404)
    if state.status == "answered":
        # Idempotent duplicate handling (incl. provider retries after success);
        # the ticket is never deleted and its TTL was refreshed on claim.
        return _callback_json({"data": {"status": "already_answered"}}, 200)

    # Verify the signed server-to-server answer over the RAW body BEFORE parsing
    # or recording. Failure denies without consuming the ticket or recording the
    # answer — the idempotency window is untouched, so a legitimate retry still
    # works.
    denied, post_only = await _verify_callback(request, raw, state)
    if denied is not None:
        return denied

    fmt = state.request.answer_format

    if fmt is not AnswerFormat.EXTERNAL:
        # A ticketed non-external question is channel-delivered: the plugin
        # forwards the human's reply as {"answer": <value>}. Validate it
        # against the STORED format (the authed door's exact rules) and record
        # the TYPED value. Query params never carry the answer here.
        value: Any
        if not raw:
            if fmt is AnswerFormat.CONFIRM:
                # The GET-confirm page's form POSTs an empty body — an
                # affirmative tap.
                value = True
            else:
                # text/select: an answer is required.
                return _callback_json({"error": "body must contain 'answer'"}, 400)
        else:
            try:
                parsed = json.loads(raw)
            except _JSON_PARSE_ERRORS:
                return _callback_json({"error": "body must be a JSON object"}, 400)
            if not isinstance(parsed, dict):
                return _callback_json({"error": "body must be a JSON object"}, 400)
            if "answer" not in parsed:
                return _callback_json({"error": "body must contain 'answer'"}, 400)
            value = parsed["answer"]
        try:
            validated = _validate_answer(state.request, value)
        except _AnswerInvalid as exc:
            return _callback_json({"error": str(exc)}, 400)
        return await _record_callback_answer(r, store, settings, ticket, interaction_id, state, validated)

    # EXTERNAL: verbatim payload semantics. Dispatch on the body FIRST —
    # empty-body branch first (``json.loads("")`` raises). Body wins: query
    # params alongside a JSON-object body are ignored as routing metadata
    # (webhook providers decorate the URL with their own params while POSTing
    # the event body).
    if not raw:
        # A body-signature verifier signs only the raw body; a replayed signature
        # over an empty body must never let ``?approved=true`` inject an answer.
        # Reject with a constant deny — nothing recorded, ticket unconsumed.
        if post_only:
            return _callback_json({"error": _POST_ONLY_EMPTY_BODY_DENY}, 400)
        answer = _params_to_answer(request)
    else:
        try:
            parsed = json.loads(raw)
        except _JSON_PARSE_ERRORS:
            return _callback_json({"error": "body must be a JSON object"}, 400)
        if not isinstance(parsed, dict):
            return _callback_json({"error": "body must be a JSON object"}, 400)
        answer = parsed

    return await _claim_external(r, store, settings, ticket, interaction_id, state, answer)


async def _callback_get(request: Request, r: Any, store: InteractionStore) -> Response:
    ticket = request.path_params["ticket"]
    interaction_id = await store.resolve_ticket(r, ticket)
    if interaction_id is None:
        return PlainTextResponse("Not Found", status_code=404, headers=_BASE_HEADERS)
    state = await store.get_state(r, interaction_id)
    if state is None:
        # State pruned while the ticket lives out its TTL tail — plain 404, never
        # a confirm page for a dead interaction.
        return PlainTextResponse("Not Found", status_code=404, headers=_BASE_HEADERS)
    if state.status == "answered":
        return HTMLResponse(_DONE_PAGE, headers=_HTML_HEADERS)
    if _callback_verifier(state) is not None:
        # A verifier-bound question is server-to-server ONLY: the browser confirm
        # form posts an EMPTY body (answer in query params) and can never carry a
        # provider signature, so no confirm page is served — the answer arrives
        # only via a signed server POST.
        return PlainTextResponse(
            "This question is verified server-to-server; no browser confirm.",
            status_code=404,
            headers=_BASE_HEADERS,
        )
    if state.request.answer_format in (AnswerFormat.CONFIRM, AnswerFormat.EXTERNAL):
        # Only these formats map the confirm form's empty-body POST to an
        # answer (confirm -> True, external -> the query-param payload).
        return HTMLResponse(_CONFIRM_PAGE, headers=_HTML_HEADERS)
    # A ticketed text/select question needs a VALUE answer, which the channel
    # plugin forwards as a POST body — never a bare confirm tap, so no page
    # with an action that would be rejected.
    return HTMLResponse(_REPLY_PAGE, headers=_HTML_HEADERS)


@http_surface().custom_route(
    "/api/interactions/callback/{ticket}",
    methods=["GET", "POST"],
    summary="External interaction callback door",
    tags=["interactions"],
    response_model=None,
    authed=False,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(400, 401, 404, 413, 500),
        success_status=200,
    ),
)
async def callback(request: Request) -> Response:
    # Rate limiting for this public door lives in the app-level
    # ``RateLimitMiddleware``, registered at app construction so it is always on
    # (tunable/disable via ``TAI_RATE_LIMIT_*``); it runs ahead of this route for
    # the interactions-callback door family.
    settings = interactions_settings()
    store = InteractionStore(settings.key_prefix)

    async with client_ctx(RedisClient, settings.redis) as r:
        if request.method == "GET":
            return await _callback_get(request, r, store)
        return await _callback_post(request, r, store, settings)
