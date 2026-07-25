"""HTTP surface for the hooks feature — the inbound event ingress plus the
authed management doors the Studio's hooks UI consumes.

- ``POST|GET /universal_webhook/{topic}`` (PUBLIC ingress) — external systems
  deliver events here; the payload is parsed and dispatched to the topic's
  registered hooks in the background. This is the most exposed door in the
  system, so the ingress is bounded (body + query cap -> 413), parses hostile
  content types safely (entity-expansion off), and answers with ``nosniff`` +
  ``no-store``. Rate limiting lives in the app-level ``RateLimitMiddleware``.
  A topic with NO verifier binding is OPEN BY DESIGN; bind a verifier to lock
  one. A bound topic verifies the RAW body BEFORE parsing; failure -> 401 with a
  constant message (no oracle), nothing dispatched. A
  body-signature (``post_only``) verifier rejects GET delivery — a GET door would
  sign an empty body while the real payload rides the URL unauthenticated. This
  ingress carries a discriminated status body (413/401/405/500/accepted) with
  custom headers and a background dispatch task, so it stays a native handler.
- ``GET /api/hooks`` (AUTHED) — list registered hooks (``?topic=`` filters) plus
  the per-topic verifier bindings under ``data.topic_verifiers`` and, under
  ``data.trigger_auth``, how a topic's webhook ingress door authenticates its caller
  (derived from those live bindings, never stored) — keyed by every topic among the
  listed hooks plus every topic currently carrying a binding.
- ``POST /api/hooks`` (AUTHED) — register a hook from a ``HookRegister`` body.
- ``DELETE /api/hooks/{name}`` (AUTHED) — unregister a hook by name; a missing
  name is a loud 404.
- ``PUT /api/hooks/topics/{topic}/verifier`` (AUTHED, ``fenced``) — set/replace a
  topic's verifier binding; an unknown verifier name is rejected at bind time (400).
- ``DELETE /api/hooks/topics/{topic}/verifier`` (AUTHED, ``fenced``) — remove a
  binding; a missing binding is a loud 404.

  Both verifier doors are ADMIN-ONLY. A binding is the only authentication
  ``/universal_webhook/{topic}`` has, and the topic namespace has no owner: any
  ``hooks``-write holder could otherwise take the lock off a topic whose hooks fire
  under keys it could never bind, or replace that lock with one whose secret it
  chooses. Removing a lock and replacing one reach the same state, so they carry the
  same fence — gating only the unbind would leave the rebind as its bypass.
- ``GET|POST /trigger/{token}`` (PUBLIC) — the trigger-link door: resolve a minted
  token to a hook topic and dispatch the payload, exactly like the ingress door but
  hiding its topic and merging the link's stored ``tool_kwargs`` in BELOW each fired
  hook's own static ``tool_kwargs``, so the link fills only the arguments that hook's
  author left unpinned. Every miss is
  the uniform 404 (no oracle). A link minted ``require_api_key`` demands an
  authenticated principal beside the token — a token holder without one gets a 403,
  and a credential the authentication backend does not admit (invalid, disabled, or
  role-governed and refused at this door's method) never reaches this handler. The
  dispatch is gated on the link's bound execution key; each fired hook's tool call is
  authorized against the HOOK's key.
- ``POST /api/hooks/trigger-links`` (AUTHED, ``write``) — mint a trigger link.
- ``GET /api/hooks/trigger-links`` (AUTHED, ``read``) — list trigger links.
- ``DELETE /api/hooks/trigger-links/{name}`` (AUTHED, ``write``) — revoke by name.

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
from tai42_contract.hooks import HookRegister
from tai42_contract.webhooks import WebhookVerificationError

from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.authz.execution import bind_execution_identity
from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.payload_parser import parse_any_payload
from tai42_skeleton.hooks.trigger_links import ResolvedTrigger, TriggerLinkError, resolve_trigger_token
from tai42_skeleton.operations import (
    BadRequestError,
    PermissionDenied,
    operation_metadata_of,
    register_operation_route,
)
from tai42_skeleton.operations.hooks import TriggerLinkCreate
from tai42_skeleton.operations.hooks import create_trigger_link as _create_trigger_link_op
from tai42_skeleton.operations.hooks import delete_topic_verifier as _delete_topic_verifier_op
from tai42_skeleton.operations.hooks import delete_trigger_link as _delete_trigger_link_op
from tai42_skeleton.operations.hooks import list_hooks as _list_hooks_op
from tai42_skeleton.operations.hooks import list_trigger_links as _list_trigger_links_op
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

# The refusal a token-holder gets on an api-key-authed trigger link they rang with no
# authenticated principal — actionable, and reachable only by a token holder.
_API_KEY_REQUIRED = "this trigger link requires an authenticated api key"


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


@tai42_app.http.custom_route(
    "/trigger/{token}",
    methods=["POST", "GET"],
    summary="Public trigger-link door",
    tags=["hooks"],
    response_model=None,
    authed=False,
)
async def trigger_link(request: Request) -> Response:
    """Fire a hook topic from a minted, token-bearing PUBLIC URL (a QR scan is a GET;
    POST comes free for curl symmetry — an anonymous token holder reaches both alike,
    while a caller that PRESENTS a credential is additionally subject to the ordinary
    route gate, which for a ROLE-governed principal derives ``read`` from GET and
    ``write`` from POST — the governing policy is the OWNER's for an owned key, so a key
    escapes that pass exactly when its governing policy is admin or carries no role
    pointer).
    The token is the capability — whoever holds the URL fires the topic's registered
    hooks.

    Every miss is the SAME 404 ``"unknown or expired trigger link"`` (unknown /
    expired / revoked / verifier-bound / in-memory-mode are deliberately
    indistinguishable — no oracle). A link minted ``require_api_key`` answers 403 to a
    token holder presenting no authenticated principal. The token is resolved BEFORE
    the payload is parsed, so an unknown token 404s without ever reaching a parse-400.
    The accepted response carries NO topic (a trigger link hides its topic from the URL
    holder); the payload rides query/body under the ingress byte cap; and the link's
    stored ``tool_kwargs`` merge into each fired hook's input BELOW that hook's own
    static ``tool_kwargs``, filling only the arguments its author left unpinned — a
    link can never restate an argument the hook pinned. Every route-emitted
    response (accepted / 404 / 403 / 400 / 413) carries ``nosniff`` + ``no-store`` — a
    capability-URL response must never be cached."""
    token = request.path_params["token"]

    cap = webhook_ingress_settings().max_body_bytes
    if len(request.url.query.encode()) > cap:
        return _error("payload too large", 413)
    try:
        raw = await _read_bounded_body(request, cap)
    except _PayloadTooLarge:
        return _error("payload too large", 413)
    request._body = raw

    # Resolve first: an unknown/expired/revoked/verifier-bound/in-memory token is the
    # uniform 404 before any parse work.
    try:
        resolved = await resolve_trigger_token(token)
    except TriggerLinkError as exc:
        return _error(exc.message, exc.status)

    # A principal ON TOP of the token. Only a token holder reaches this branch, so the 403
    # is actionable for them and invisible to everyone else (uniform 404 above).
    if resolved.require_api_key and not _authenticated_caller(request):
        # Logged: the resolver's preceding line reports only that the token resolved,
        # which reads as a successful fire without this.
        logger.warning("hooks: trigger door refused topic=%r cause=api-key-required", resolved.topic)
        return _error(_API_KEY_REQUIRED, 403)

    try:
        payload = await parse_any_payload(request, include_query=True)
    except ValueError as e:
        # No topic echoed on the rejection — the link hides its topic.
        return _ingress_json({"status": "rejected", "error": str(e)}, status_code=400)

    task = BackgroundTask(_dispatch_trigger_link, resolved, payload)
    return _ingress_json({"status": "accepted"}, background=task)


def _authenticated_caller(request: Request) -> bool:
    """Whether the request carries a valid authenticated principal — and, with access
    control DISABLED, unconditionally ``True``.

    The door is registered ``authed=False`` (a static flag a per-record requirement cannot
    flip), but the authentication backend still runs and has already denied every
    credential it does not admit — so ``request.user`` is the decision here."""
    if not access_control_settings().enable:
        return True
    return bool(request.user.is_authenticated)


async def _dispatch_trigger_link(resolved: ResolvedTrigger, payload: dict) -> None:
    """Dispatch a resolved link's topic under the LINK's own execution identity.

    The bind wraps the whole fan-out, so nothing here runs with the server's unbounded
    authority and a deleted/disabled/policy-less key refuses the dispatch before any hook
    runs; each fired hook then re-binds its OWN key inside its own task.

    That refusal is a routine revocation outcome landing after the ``accepted`` response,
    so it is logged as an error outcome rather than raised. Everything else propagates."""
    try:
        async with bind_execution_identity(
            resolved.execution_key, bound_fingerprint=resolved.execution_key_fingerprint
        ):
            await get_hooks_manager().on_event(
                topic=resolved.topic, payload=payload, tool_kwargs_override=resolved.tool_kwargs
            )
    except PermissionDenied as exc:
        logger.error(
            "hooks: trigger dispatch refused topic=%r execution_key=%r cause=%s",
            resolved.topic,
            resolved.execution_key,
            exc,
        )


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
    """Parse + validate the client-facing hook body into the operation's flat fields,
    rejecting a malformed body before the operation runs (the adapter's plain parse
    would yield 422; this yields an explicit 400 surface).

    Validated against ``HookRegister``, which carries no ``execution_key_fingerprint`` —
    the operation derives that server-side; a client-set one would pin an authorization
    anchor the server never verified."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object of hook params") from None
    try:
        params = HookRegister.model_validate(body)
    except ValidationError as exc:
        raise BadRequestError(f"invalid hook params: {exc}") from exc
    return params.model_dump()


async def _extract_trigger_link_params(request: Request) -> dict:
    """Parse + validate the ``TriggerLinkCreate`` body into the operation's flat
    fields, rejecting a malformed body with an explicit 400 (the adapter's plain
    parse would yield 422; the ttl contract demands 400 for an absent/invalid
    ``ttl_seconds``)."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object of trigger-link params") from None
    try:
        params = TriggerLinkCreate.model_validate(body)
    except ValidationError as exc:
        raise BadRequestError(f"invalid trigger link params: {exc}") from exc
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
    action="fenced",
)

delete_topic_verifier = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_topic_verifier_op),
    path="/api/hooks/topics/{topic}/verifier",
    method="DELETE",
    action="fenced",
)

create_trigger_link = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_trigger_link_op),
    path="/api/hooks/trigger-links",
    method="POST",
    context_extractor=_extract_trigger_link_params,
    action="write",
)

list_trigger_links = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_trigger_links_op),
    path="/api/hooks/trigger-links",
    method="GET",
    action="read",
)

delete_trigger_link = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_trigger_link_op),
    path="/api/hooks/trigger-links/{name}",
    method="DELETE",
    action="write",
)
