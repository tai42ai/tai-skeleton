"""Conversation-route management operations — the routing-table surface behind the
``/api/conversations*`` doors, the ``tai conversations`` CLI and the MCP tools.

A route binds an inbound door (``api`` or ``channel``) to the ``agent_name`` a turn runs
and the ``execution_key`` that turn runs AS. A row's ``callback_secret`` is shown ONCE at
create and withheld from every read. ``create_conversation_route`` DELEGATES authority, so
it carries the ``authority_changing`` tier and binds the key BEFORE any write.

Every routing operation requires the redis conversations backend and otherwise refuses
with a loud 501 (``NotSupportedError``).
"""

from __future__ import annotations

import secrets
from typing import Any

from tai42_contract.conversations import ROUTE_NAME_RE, ConversationRoute, ConversationRouteCreate

from tai42_skeleton.conversations.address import canonical_address
from tai42_skeleton.conversations.cache import get_conversations_manager
from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager
from tai42_skeleton.operations import BadRequestError, NotFoundError, operation
from tai42_skeleton.operations._authority import assert_execution_key_bindable, require_admin, resolve_caller
from tai42_skeleton.operations.errors import ForbiddenError, NotSupportedError

# Surfaced before a create does any bind work it would then have to discard.
_NO_BACKEND = "conversation routes require the redis conversations backend"


def _require_backend() -> BaseConversationsManager:
    manager = get_conversations_manager()
    if isinstance(manager, InMemoryConversationsManager):
        raise NotSupportedError(_NO_BACKEND)
    return manager


def _public_route_view(route: ConversationRoute) -> dict[str, Any]:
    """A stored row for a read response, its ``callback_secret`` stripped."""
    data = route.model_dump(mode="json")
    data.pop("callback_secret", None)
    return data


def _validate_route_name(route_name: str) -> None:
    if not ROUTE_NAME_RE.fullmatch(route_name):
        raise BadRequestError(f"route_name must be a slug matching {ROUTE_NAME_RE.pattern!r}: {route_name!r}")


async def _unclaimed_channel_identity(
    manager: BaseConversationsManager, *, route_name: str, channel: str, our_identity: str
) -> str:
    """The canonical ``our_identity`` a ``channel`` row is STORED under, refused when
    another route already claims that ``(channel, identity)`` pair. Inbound routing matches
    the canonical form, so a second claimant would make every message to that identity
    unresolvable; it is refused here at the write instead.
    """
    try:
        identity = canonical_address(our_identity)
    except ValueError as exc:
        raise BadRequestError(f"invalid our_identity: {exc}") from exc
    for row in (await manager.list_routes()).values():
        if (
            row.route_name != route_name
            and row.door == "channel"
            and row.channel == channel
            and row.our_identity is not None
            and canonical_address(row.our_identity) == identity
        ):
            raise BadRequestError(f"channel {channel!r} identity {identity!r} is already routed by {row.route_name!r}")
    return identity


@operation(summary="List conversation routes", tags=["conversations"], errors=[NotSupportedError])
async def list_conversation_routes() -> dict[str, Any]:
    """Every stored conversation route, each with its ``callback_secret`` withheld.
    Returns ``{"items", "total"}``.
    """
    manager = _require_backend()
    routes = await manager.list_routes()
    items = [_public_route_view(route) for route in routes.values()]
    return {"items": items, "total": len(items)}


@operation(
    summary="Get a conversation route",
    tags=["conversations"],
    errors=[BadRequestError, NotFoundError, NotSupportedError],
)
async def get_conversation_route(route_name: str) -> dict[str, Any]:
    """One conversation route by name, with its ``callback_secret`` withheld. An
    unknown name is a loud 404; a name that is not a valid slug is a 400."""
    _validate_route_name(route_name)
    manager = _require_backend()
    route = await manager.get_route(route_name)
    if route is None:
        raise NotFoundError(f"conversation route not found: {route_name!r}")
    return _public_route_view(route)


@operation(
    summary="Create or replace a conversation route",
    tags=["conversations"],
    destructive=True,
    authority_changing=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError, NotSupportedError],
    request_model=ConversationRouteCreate,
)
async def create_conversation_route(
    route_name: str,
    door: str,
    agent_name: str,
    execution_key: str,
    channel: str | None = None,
    our_identity: str | None = None,
    callback_url: str | None = None,
) -> dict[str, Any]:
    """Create a conversation route from its flat parameters — an UPSERT, so this is the
    create path AND the edit path for a route of that name.

    ``execution_key`` is the api-key identity the turn runs AS; the caller must be allowed
    to delegate it and it must be usable by a tokenless fire, both decided BEFORE the write
    so a refusal leaves any existing row untouched. ``agent_name`` must merely EXIST — the
    key's live grants bound the turn at fire. A ``channel`` row's ``our_identity`` is stored
    canonicalized and must not already be routed on that channel. An ``api`` row's
    ``callback_secret`` is minted here and returned ONCE. Returns ``{"created",
    "route_name", "route", "callback_secret"}``.
    """
    # Validate the whole body shape at the operation, not the edge: the MCP tool and a
    # direct call take these flat parameters and bypass the HTTP extractor.
    try:
        create = ConversationRouteCreate(
            route_name=route_name,
            door=door,  # pyright: ignore[reportArgumentType]
            agent_name=agent_name,
            execution_key=execution_key,
            channel=channel,
            our_identity=our_identity,
            callback_url=callback_url,
        )
    except ValueError as exc:
        raise BadRequestError(f"invalid conversation route: {exc}") from exc

    manager = _require_backend()

    # Existence only: the turn runs as the key, so the key's authority over the agent is
    # deliberately not checked here.
    from tai42_skeleton.app import instance

    if create.agent_name not in instance.app.agents.all_agents():
        raise NotFoundError(f"agent not found: {create.agent_name!r}")

    stored = create.model_dump()
    # Only a ``channel`` row carries both fields; its identity is stored canonicalized.
    if create.channel is not None and create.our_identity is not None:
        stored["our_identity"] = await _unclaimed_channel_identity(
            manager, route_name=create.route_name, channel=create.channel, our_identity=create.our_identity
        )

    execution_key_fingerprint = await assert_execution_key_bindable(await resolve_caller(), create.execution_key)

    # Signs the api-door callback; a ``channel`` row signs nothing and carries no secret.
    callback_secret = secrets.token_urlsafe(32) if create.door == "api" else None

    route = ConversationRoute(
        **stored,
        callback_secret=callback_secret,
        execution_key_fingerprint=execution_key_fingerprint,
    )
    created = await manager.put_route(route)
    return {
        "created": created,
        "route_name": route.route_name,
        "route": _public_route_view(route),
        "callback_secret": callback_secret,
    }


@operation(
    summary="Read one conversation answer record",
    tags=["conversations"],
    errors=[BadRequestError, ForbiddenError, NotFoundError, NotSupportedError],
)
async def get_conversation_message(route_name: str, message_id: str) -> dict[str, Any]:
    """One conversation answer record by ``message_id`` under ``route_name``, caller-scoped.

    An api-door record is readable by the caller that invoked the turn or by an admin; a
    channel-door record is admin-only. A missing record or one on another route is a 404; a
    record that exists but is not the caller's is a 403. An admin reads the whole record;
    the caller reads the caller-safe projection, which withholds the internal detail of the
    route key's run.
    """
    _validate_route_name(route_name)
    _require_backend()
    from tai42_skeleton.conversations.records import ConversationRecordStore
    from tai42_skeleton.conversations.settings import ConversationsSettings

    record = await ConversationRecordStore(ConversationsSettings()).get_record(message_id)
    if record is None or record.route_name != route_name:
        raise NotFoundError(f"conversation record not found: {message_id!r}")
    caller = await resolve_caller()
    if caller.is_admin:
        return record.view()
    if record.caller_principal != caller.caller_id:
        # A channel record's ``None`` never equals a real caller id, so this refuses it too.
        raise ForbiddenError("you may only read conversation records from turns you invoked")
    return record.caller_view()


@operation(
    summary="List failed conversation deliveries",
    tags=["conversations"],
    errors=[ForbiddenError, NotSupportedError],
)
async def list_failed_conversations() -> dict[str, Any]:
    """Every answer record whose delivery ended ``failed``. Admin-only — the listing spans
    every route and caller, so it is not caller-scoped. Returns ``{"items", "total"}``."""
    _require_backend()
    require_admin(await resolve_caller())
    from tai42_skeleton.conversations.models import DeliveryStatus
    from tai42_skeleton.conversations.records import ConversationRecordStore
    from tai42_skeleton.conversations.settings import ConversationsSettings

    records = await ConversationRecordStore(ConversationsSettings()).list_by_status(frozenset({DeliveryStatus.FAILED}))
    items = [record.view() for record in records]
    return {"items": items, "total": len(items)}


@operation(
    summary="Delete a conversation route",
    tags=["conversations"],
    errors=[BadRequestError, NotFoundError, NotSupportedError],
)
async def delete_conversation_route(route_name: str) -> dict[str, Any]:
    """Delete a conversation route by name. An unknown name is a loud 404; a name that
    is not a valid slug is a 400. Returns ``{"removed", "route_name"}``."""
    _validate_route_name(route_name)
    manager = _require_backend()
    removed = await manager.delete_route(route_name)
    if not removed:
        raise NotFoundError(f"conversation route not found: {route_name!r}")
    return {"removed": True, "route_name": route_name}
