"""HTTP surface for the hooks feature — the inbound event ingress plus the
authed management doors the Studio's hooks UI consumes.

- ``POST|GET /universal_webhook/{topic}`` (PUBLIC ingress) — external systems
  deliver events here; the payload is parsed and dispatched to the topic's
  registered hooks in the background. This is the most exposed door in the
  system, so the ingress is bounded (body + query cap -> 413), parses hostile
  content types safely (entity-expansion off), and answers with ``nosniff`` +
  ``no-store``. Rate limiting lives in the app-level ``RateLimitMiddleware``.
  A topic with NO verifier binding is OPEN BY DESIGN (today's behavior); bind a
  verifier to lock one. A bound topic verifies the RAW body BEFORE parsing;
  failure -> 401 with a constant message (no oracle), nothing dispatched. A
  body-signature (``post_only``) verifier rejects GET delivery — a GET door would
  sign an empty body while the real payload rides the URL unauthenticated. This
  ingress carries a discriminated status body (413/401/405/500/accepted) with
  custom headers and a background dispatch task, so it stays a native handler.
- ``GET /api/hooks`` (AUTHED) — list registered hooks (``?topic=`` filters) plus
  the per-topic verifier bindings under ``data.topic_verifiers``.
- ``POST /api/hooks`` (AUTHED) — register a hook from a ``HookParams`` body.
- ``DELETE /api/hooks/{name}`` (AUTHED) — unregister a hook by name; a missing
  name is a loud 404.
- ``PUT /api/hooks/topics/{topic}/verifier`` (AUTHED) — set/replace a topic's
  verifier binding; an unknown verifier name is rejected at bind time (400).
- ``DELETE /api/hooks/topics/{topic}/verifier`` (AUTHED) — remove a binding;
  a missing binding is a loud 404.

The management doors are thin adapters over the operations in
``tai42_skeleton.operations.hooks`` (the same manager the MCP hook-management tools
drive) — no hook logic lives here. Each management door's body is parsed and
structurally validated at the HTTP edge (a strict 400 surface) into the
operation's flat arguments; the operation owns the logical guards. Success bodies are ``{"data": ...}``; failures are
``{"error": "<message>"}``.
"""

from __future__ import annotations

import logging

from fastapi import Request
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams
from tai42_contract.webhooks import WebhookVerificationError

from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.payload_parser import parse_any_payload
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.hooks import delete_topic_verifier as _delete_topic_verifier_op
from tai42_skeleton.operations.hooks import list_hooks as _list_hooks_op
from tai42_skeleton.operations.hooks import list_verifiers as _list_verifiers_op
from tai42_skeleton.operations.hooks import register_hook as _register_hook_op
from tai42_skeleton.operations.hooks import set_topic_verifier as _set_topic_verifier_op
from tai42_skeleton.operations.hooks import unregister_hook as _unregister_hook_op
from tai42_skeleton.webhooks.settings import webhook_ingress_settings

logger = logging.getLogger(__name__)

# nosniff + no-store ride every ingress response: the door is public and its
# body reflects attacker-influenced content the caller should never cache or
# have content-sniffed.
_INGRESS_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}

# The constant failure message returned to the caller on a verification failure:
# no oracle detail (a distinguishing message would leak which check failed). The
# server-side log records the reason; the raw body is NEVER logged verbatim.
_VERIFY_FAILED = "webhook verification failed"


class _PayloadTooLarge(Exception):
    """Raised when the request body or query string exceeds the configured cap."""


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code, headers=dict(_INGRESS_HEADERS))


def _ingress_json(payload: dict, status_code: int = 200, background: BackgroundTask | None = None) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers=dict(_INGRESS_HEADERS), background=background)


def _sanitize_topic_for_log(topic: str) -> str:
    """Strip CR and LF so a crafted topic path param cannot forge extra log lines
    (log injection). Both are removed — a lone carriage return can rewrite a line
    on many terminals, not only a newline."""
    return topic.replace("\r", "").replace("\n", "")


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


# -- Inbound event ingress (PUBLIC, native) ----------------------------------


@tai42_app.http.custom_route(
    "/universal_webhook/{topic}",
    methods=["POST", "GET"],
    summary="Public webhook ingress door for a topic",
    tags=["hooks"],
    response_model=None,
    authed=False,
)
async def universal_webhook(request: Request) -> Response:
    topic = request.path_params["topic"]
    logger.info("--- INCOMING EVENT ON TOPIC: %s ---", _sanitize_topic_for_log(topic))

    cap = webhook_ingress_settings().max_body_bytes
    # Payload rides the query string too (a GET/POST decorated URL), so cap it on
    # actual bytes exactly like the body.
    if len(request.url.query.encode()) > cap:
        return _error("payload too large", 413)
    try:
        raw = await _read_bounded_body(request, cap)
    except _PayloadTooLarge:
        return _error("payload too large", 413)
    # Cache the bounded bytes on the request so ``parse_any_payload`` re-reads them
    # (json/xml/form) without re-consuming the already-drained stream. The verifier
    # runs over these exact raw bytes BEFORE any parse.
    request._body = raw

    manager = get_hooks_manager()
    binding = await manager.get_topic_verifier(topic)
    # A body-signature (``post_only``) verifier authenticates the raw body only;
    # its topic's dispatched payload must therefore exclude the unauthenticated
    # query string (else a captured signed delivery replays with appended params).
    strip_query = False
    if binding is not None:
        denied, strip_query = await _verify_ingress(request, raw, binding, topic)
        if denied is not None:
            return denied

    try:
        payload = await parse_any_payload(request, include_query=not strip_query)
    except ValueError as e:
        return _ingress_json({"status": "rejected", "topic": topic, "error": str(e)}, status_code=400)

    task = BackgroundTask(manager.on_event, topic=topic, payload=payload)
    return _ingress_json({"status": "accepted", "topic": topic}, background=task)


async def _verify_ingress(request: Request, raw: bytes, binding: dict, topic: str) -> tuple[Response | None, bool]:
    """Run the topic's bound verifier over the raw body. Return
    ``(deny_response, False)`` on any failure (nothing dispatched), or
    ``(None, post_only)`` when verification passes — ``post_only`` tells the
    caller whether to drop the unauthenticated query string from the payload
    (True for a body-signature verifier, whose signature covers only the body).

    Fails CLOSED on every path: a signature failure -> 401 (constant message);
    an unknown verifier name / missing secret env / verifier bug -> 500. A
    ``post_only`` (body-signature) verifier rejects GET delivery -> 405."""
    # ``get_topic_verifier`` validates the stored binding against
    # ``TopicVerifierBinding`` (whose ``verifier`` carries ``min_length=1``), so
    # ``verifier`` is a guaranteed non-empty str and ``config`` a dict here — no
    # shape re-check needed.
    name = binding["verifier"]
    config = binding["config"]
    safe_topic = _sanitize_topic_for_log(topic)
    try:
        verifier = tai42_app.webhook_verifiers.get(name)
    except Exception:
        # A bound name that no longer resolves (verifier module dropped from the
        # manifest) must deny, not dispatch unverified. Loud 500, logged.
        logger.error("webhook verify: no registered verifier %r for topic %s", name, safe_topic)
        return _error("webhook verification error", 500), False

    post_only = bool(getattr(verifier, "post_only", False))
    if post_only and request.method == "GET":
        # A body-signature verifier signs the raw body; a GET door would sign an
        # empty body while the real payload rides the URL unauthenticated.
        return _error("this verified topic accepts POST delivery only", 405), False

    try:
        await verifier.verify(raw, request.headers, config)
    except WebhookVerificationError as exc:
        # Log the reason (never the raw body verbatim — a payload can itself hold
        # sensitive data); return the constant no-oracle message.
        logger.warning("webhook verification failed for topic %s: %s", safe_topic, exc)
        return _error(_VERIFY_FAILED, 401), False
    except Exception:
        # A missing secret env var / verifier bug fails CLOSED as a loud 500, not
        # a soft-open dispatch.
        logger.error("webhook verifier error for topic %s", safe_topic, exc_info=True)
        return _error("webhook verification error", 500), False
    return None, post_only


# -- Hook management (AUTHED) — HTTP-edge extractors --------------------------


async def _extract_list_query(request: Request) -> dict:
    """The optional ``?topic=`` filter as the operation's flat ``topic`` argument
    (a GET reads its parameters from the query string, never a body)."""
    return {"topic": request.query_params.get("topic")}


async def _extract_hook_params(request: Request) -> dict:
    """Parse + validate the ``HookParams`` body into the operation's flat fields,
    rejecting a malformed body before the operation runs (the adapter's plain parse
    would yield 422; this yields an explicit 400 surface)."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object of hook params") from None
    try:
        params = HookParams.model_validate(body)
    except ValidationError as exc:
        raise BadRequestError(f"invalid hook params: {exc}") from exc
    return params.model_dump()


async def _extract_binding(request: Request) -> dict:
    """Parse + structurally validate a PUT binding body into the operation's flat
    ``verifier`` / ``config`` arguments (the unknown-verifier check is the
    operation's, so the projected tool and CLI carry it too)."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object") from None
    name = body.get("verifier")
    if not isinstance(name, str) or not name:
        raise BadRequestError("binding requires a non-empty 'verifier' name") from None
    config = body.get("config", {})
    if not isinstance(config, dict):
        raise BadRequestError("binding 'config' must be a JSON object") from None
    return {"verifier": name, "config": config}


list_hooks = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_hooks_op),
    path="/api/hooks",
    method="GET",
    context_extractor=_extract_list_query,
    action="read",
)

register_hook = register_operation_route(
    tai42_app,
    operation_metadata_of(_register_hook_op),
    path="/api/hooks",
    method="POST",
    context_extractor=_extract_hook_params,
    action="write",
)

list_verifiers = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_verifiers_op),
    path="/api/hooks/verifiers",
    method="GET",
    action="read",
)

unregister_hook = register_operation_route(
    tai42_app,
    operation_metadata_of(_unregister_hook_op),
    path="/api/hooks/{name}",
    method="DELETE",
    action="write",
)

set_topic_verifier = register_operation_route(
    tai42_app,
    operation_metadata_of(_set_topic_verifier_op),
    path="/api/hooks/topics/{topic}/verifier",
    method="PUT",
    context_extractor=_extract_binding,
    action="write",
)

delete_topic_verifier = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_topic_verifier_op),
    path="/api/hooks/topics/{topic}/verifier",
    method="DELETE",
    action="write",
)
