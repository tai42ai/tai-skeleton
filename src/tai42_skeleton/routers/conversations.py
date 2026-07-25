"""HTTP surface for the conversation-route management feature — the authed CRUD doors
the operator and Studio drive over the routing table.

- ``GET /api/conversations`` (AUTHED) — list the stored routes, each with its
  ``callback_secret`` withheld.
- ``GET /api/conversations/{route_name}`` (AUTHED) — read one route by name; an unknown
  name is a loud 404.
- ``POST /api/conversations/{route_name}`` (AUTHED, ``authority_changing``) — create or
  replace a route from a ``ConversationRouteCreate`` body. An ``api`` row's minted
  ``callback_secret`` is returned ONCE here and never again. Binding the route's
  ``execution_key`` is a pass-role decision the operation takes before any write.
- ``DELETE /api/conversations/{route_name}`` (AUTHED) — delete a route by name; an
  unknown name is a loud 404.

Thin adapters over ``tai42_skeleton.operations.conversations`` — no routing logic here.
The POST body is structurally validated at the edge (a strict 400 surface); the operation
owns the logical guards. The channel door (``accept`` / ``record_delivery_status``) lives
on the ``conversations`` facet, not here.
"""

from __future__ import annotations

import logging

from fastapi import Request
from pydantic import ValidationError
from starlette.responses import JSONResponse, Response
from tai42_contract.access_control import get_current_user_id
from tai42_contract.app import tai42_app
from tai42_contract.conversations import ConversationMessage, ConversationRouteCreate

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.conversations.caps import AddressRateLimitedError, ThreadQueueOverflowError
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.conversations.turn import ConversationRouteResolutionError
from tai42_skeleton.operations import (
    BadRequestError,
    operation_metadata_of,
    register_operation_route,
)
from tai42_skeleton.operations.conversations import create_conversation_route as _create_conversation_route_op
from tai42_skeleton.operations.conversations import delete_conversation_route as _delete_conversation_route_op
from tai42_skeleton.operations.conversations import get_conversation_message as _get_conversation_message_op
from tai42_skeleton.operations.conversations import get_conversation_route as _get_conversation_route_op
from tai42_skeleton.operations.conversations import list_conversation_routes as _list_conversation_routes_op
from tai42_skeleton.operations.conversations import list_failed_conversations as _list_failed_conversations_op
from tai42_skeleton.operations.errors import NotSupportedError

logger = logging.getLogger(__name__)


async def _extract_route_create(request: Request) -> dict:
    """Parse + validate the client-facing route body into the operation's flat fields,
    rejecting a malformed body with an explicit 400 (the adapter's plain parse would
    yield 422).

    The ``route_name`` rides the URL path, not the body, so it is injected from the path
    param before validation — a body that also carries a ``route_name`` disagreeing with
    the path is rejected rather than silently overriding either."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object of route params") from None
    path_route_name = request.path_params["route_name"]
    body_route_name = body.get("route_name")
    if body_route_name is not None and body_route_name != path_route_name:
        raise BadRequestError("route_name in the body must match the route_name in the path") from None
    body = {**body, "route_name": path_route_name}
    try:
        create = ConversationRouteCreate.model_validate(body)
    except ValidationError as exc:
        raise BadRequestError(f"invalid conversation route: {exc}") from exc
    return create.model_dump()


list_conversation_routes = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_conversation_routes_op),
    path="/api/conversations",
    method="GET",
    action="read",
)

get_conversation_route = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_conversation_route_op),
    path="/api/conversations/{route_name}",
    method="GET",
    action="read",
)

create_conversation_route = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_conversation_route_op),
    path="/api/conversations/{route_name}",
    method="POST",
    context_extractor=_extract_route_create,
    action="write",
)

delete_conversation_route = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_conversation_route_op),
    path="/api/conversations/{route_name}",
    method="DELETE",
    action="write",
)

# The admin-tier failed-delivery listing sits on a literal path so the ``{route_name}``
# get/delete doors above never capture it. It is registered BEFORE the read-one door for
# the same reason it reads a literal segment.
list_failed_conversations = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_failed_conversations_op),
    path="/api/conversations/messages/failed",
    method="GET",
    action="read",
)

get_conversation_message = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_conversation_message_op),
    path="/api/conversations/{route_name}/messages/{message_id}",
    method="GET",
    action="read",
)


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@http_surface().custom_route(
    "/api/conversations/{route_name}/messages",
    methods=["POST"],
    summary="Send a message to a conversation route",
    tags=["conversations"],
    request_model=ConversationMessage,
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=True,
        reads_body=True,
        error_statuses=(400, 401, 404, 429, 501, 503),
        success_status=202,
        additional_success_statuses=(200,),
    ),
    action="write",
)
async def send_conversation_message(request: Request) -> Response:
    """Accept one authed message for ``route_name`` and run its turn AS the route's
    execution key.

    The auth gate authorizes who may SEND; the turn's own authority is the route's
    execution key, not the caller. Default answer is ``202 {message_id, thread_id}`` with
    the answer delivered later by signed callback. With a ``wait_seconds`` body field
    (clamped here to ``sync_wait_max_seconds``) a turn finishing in time answers ``200``
    with the answer inline and its callback suppressed, so it never double-fires; a turn
    still running when the wait elapses falls back to ``202``.
    """
    if reload_gate.locked:
        return reload_gate.reject_response()
    route_name = request.path_params["route_name"]
    try:
        body = await request.json()
    except ValueError:
        return _error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _error("body must be a JSON object", 400)
    try:
        message = ConversationMessage.model_validate(body)
    except ValidationError as exc:
        return _error(f"invalid conversation message: {exc}", 400)

    cap = ConversationsSettings().sync_wait_max_seconds
    wait_seconds = 0 if message.wait_seconds is None else min(message.wait_seconds, cap)

    from tai42_skeleton.conversations import submit_api_message

    try:
        result = await submit_api_message(
            route_name, message.external_user_id, message.text, get_current_user_id(), wait_seconds
        )
    except ConversationRouteResolutionError as exc:
        return _error(str(exc), 404)
    except AddressRateLimitedError as exc:
        return _error(str(exc), 429)
    except ThreadQueueOverflowError as exc:
        return _error(str(exc), 503)
    except NotSupportedError as exc:
        return _error(str(exc), 501)

    payload: dict[str, object] = {"message_id": result.message_id, "thread_id": result.thread_id}
    if result.answer is not None:
        payload["answer"] = result.answer.model_dump(mode="json")
        return JSONResponse({"data": payload}, status_code=200)
    return JSONResponse({"data": payload}, status_code=202)


@tai42_app.lifecycle.on_startup
async def _redrive_pending_conversations() -> None:
    """Resume every unfinished conversation record on boot and start the periodic
    stalled-delivery sweep, so nothing is stranded across a restart.

    Intake re-drive must run FIRST: it gives every stranded ``accepted`` record a terminal
    outcome, which is work the delivery re-drive then picks up. The sweep is what recovers
    a record whose worker died holding a still-live lease. No-op with no backend."""
    if ConversationsSettings().in_memory:
        return
    from tai42_skeleton.conversations import redrive_accepted, redrive_pending, start_delivery_sweep

    await redrive_accepted()
    await redrive_pending()
    start_delivery_sweep()


@tai42_app.lifecycle.on_shutdown
async def _stop_conversations_delivery_sweep() -> None:
    """Cancel and await the stalled-delivery sweep on the serving loop it lives on. A
    backend-less deployment never started one."""
    if ConversationsSettings().in_memory:
        return
    from tai42_skeleton.conversations import stop_delivery_sweep

    await stop_delivery_sweep()
