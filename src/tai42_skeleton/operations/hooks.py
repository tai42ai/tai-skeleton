"""Hook-management operations — the authed hooks surface the Studio drives, over
the shared ``get_hooks_manager()``.

* ``list_hooks`` lists registered hooks (filtered to ``topic`` when given) plus the
  per-topic verifier bindings — one read the Studio hooks UI consumes.
* ``register_hook`` registers a hook from its FLAT parameters (the ``HookParams``
  fields spread as arguments) and reports whether it was newly stored.
* ``unregister_hook`` removes a hook by name; an unknown name is a loud 404.
* ``list_verifiers`` lists the registered webhook-verifier names (the bind catalog).
* ``set_topic_verifier`` binds (or replaces) a topic's webhook verifier; an unknown
  verifier name is rejected at bind time (400).
* ``delete_topic_verifier`` removes a topic's binding; a missing binding is a 404.

These operations are the single source the ``/api/hooks*`` routes, the ``tai hooks``
CLI, and the MCP tool surface derive from — one definition of the hook-management
tools (``list_hooks`` / ``register_hook`` / ``unregister_hook``) behind both the MCP
tool surface and the HTTP doors. A hook's ``tool_kwargs`` can carry caller-supplied values, so the
list read is secret-bearing; that is acceptable on this authed surface (as with
config env).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from tai42_contract.access_control import get_current_user_id
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams

from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.hooks import trigger_links
from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.trigger_links import TriggerLinkError
from tai42_skeleton.operations import BadRequestError, NotFoundError, operation
from tai42_skeleton.operations.errors import ConflictError, NotSupportedError, OperationError


class TopicVerifierBinding(BaseModel):
    """Bind a registered webhook ``verifier`` (with optional ``config``) to a
    hook topic so its deliveries are signature-verified."""

    verifier: str
    config: dict[str, Any] = {}


class TriggerLinkCreate(BaseModel):
    """Mint a trigger link for a hook ``topic``.

    ``ttl_seconds`` is REQUIRED (no default ⇒ the key must be present) and STRICT
    (``"3600"``, ``3600.0``, ``3600.5`` and bools all reject under one regime): a
    positive int is a timed link, ``null`` is a permanent link, and ``0``/negative
    is a loud 400 — expiry is the creator's explicit choice with no default and no
    product ceiling. ``tool_kwargs`` (optional) is stored on the link and merged
    LAST into every fired hook's input, winning on a colliding key over the hook's
    own ``tool_kwargs``."""

    topic: str
    name: str | None = None
    ttl_seconds: int | None = Field(strict=True)
    tool_kwargs: dict[str, Any] | None = None


# The status → operation-error mapping shared by the three trigger-link adapters,
# mirroring how ``ClaimLinkError`` maps to the operation error surface.
_TRIGGER_ERROR_BY_STATUS: dict[int, type[OperationError]] = {
    400: BadRequestError,
    404: NotFoundError,
    409: ConflictError,
    501: NotSupportedError,
}


def _map_trigger_error(exc: TriggerLinkError) -> OperationError:
    error_cls = _TRIGGER_ERROR_BY_STATUS.get(exc.status, OperationError)
    return error_cls(exc.message)


async def _resolve_trigger_caller_id() -> str | None:
    """The AMBIENT caller identity stamped as ``created_by`` — never a request field
    (which would be caller-spoofable). With the gate OFF there is no principal, so it
    resolves to ``None`` (the record's nullable branch); with the gate ON but the
    identity contextvar UNSET it RAISES (a 500), the same fail-closed posture the
    key-management surface takes rather than a silent unattributed record."""
    if not access_control_settings().enable:
        return None
    caller_id = get_current_user_id()
    if caller_id is None:
        raise RuntimeError(
            "access_control: caller user id is unset on an authed trigger-link request — "
            "the guard middleware must bind it; refusing to proceed"
        )
    return caller_id


@operation(summary="List registered hooks", tags=["hooks"])
async def list_hooks(topic: str | None = None) -> dict[str, Any]:
    """List registered hooks plus the per-topic verifier bindings.

    With a topic, lists only the hooks registered for that topic; without one,
    lists every registered hook. The response carries the hooks under ``items``
    (with ``total``) and the per-topic verifier bindings under ``topic_verifiers``.
    """
    manager = get_hooks_manager()
    hooks = await manager.list_hooks_by_topic(topic=topic) if topic else await manager.list_hooks()
    items = [params.model_dump(mode="json") for params in hooks.values()]
    topic_verifiers = await manager.all_topic_verifiers()
    return {"items": items, "total": len(items), "topic_verifiers": topic_verifiers}


@operation(
    summary="Register a hook",
    tags=["hooks"],
    destructive=True,
    errors=[BadRequestError],
    request_model=HookParams,
)
async def register_hook(
    name: str,
    topic: str,
    tool: str,
    tool_kwargs: dict[str, Any] | None = None,
    condition: str | None = None,
    condition_id: str | None = None,
    condition_kwargs: dict[str, Any] | None = None,
    expr: str | None = None,
    expr_id: str | None = None,
    expr_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a new hook from its flat parameters.

    Supports conditional execution (via ``condition`` or ``condition_id``),
    payload transformation (via ``expr`` or ``expr_id``), and dynamic tool
    arguments. Returns ``{"registered", "name"}`` — whether the hook was newly
    stored, and its name.
    """
    try:
        params = HookParams(
            name=name,
            topic=topic,
            tool=tool,
            tool_kwargs=tool_kwargs or {},
            condition=condition,
            condition_id=condition_id,
            condition_kwargs=condition_kwargs or {},
            expr=expr,
            expr_id=expr_id,
            expr_kwargs=expr_kwargs or {},
        )
        registered = await get_hooks_manager().register(params)
    except ValueError as exc:
        # The manager compiles the condition/expr jq at registration; a bad
        # expression is client input, so it is a 400 (as is a flat-field validation
        # failure reaching this operation from the MCP/CLI edge, which has no HTTP
        # extractor). Store/transport failures raise other types and surface as 500.
        raise BadRequestError(f"invalid hook params: {exc}") from exc
    return {"registered": registered, "name": name}


@operation(summary="Unregister a hook", tags=["hooks"], errors=[NotFoundError])
async def unregister_hook(name: str) -> dict[str, Any]:
    """Unregister a hook by name. An unknown hook name is a loud 404.

    Returns ``{"removed", "name"}``.
    """
    removed = await get_hooks_manager().unregister(name=name)
    if not removed:
        raise NotFoundError(f"hook not found: {name!r}")
    return {"removed": True, "name": name}


@operation(summary="List registered webhook verifiers", tags=["hooks"])
async def list_verifiers() -> list[str]:
    """The sorted names of every registered webhook verifier — the catalog the
    Studio bind form offers instead of free text. Names ONLY: a verifier object and
    a bound config are secret-adjacent and never leave this door. An empty registry
    (no verifier lifecycle module loaded) is a valid state, so it returns ``[]``."""
    # Reached through the concrete app singleton because ``names`` rides the skeleton
    # facet, not the tai42-contract ``AppWebhookVerifiers`` protocol.
    from tai42_skeleton.app import instance

    return instance.app.webhook_verifiers.names()


@operation(
    summary="Bind a webhook verifier to a topic",
    tags=["hooks"],
    destructive=True,
    errors=[BadRequestError],
    request_model=TopicVerifierBinding,
)
async def set_topic_verifier(topic: str, verifier: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind (or replace) a topic's webhook verifier.

    The verifier name is resolved against the registry here — an unknown name is
    rejected at BIND time (400), never left to fail at delivery.
    """
    try:
        tai42_app.webhook_verifiers.get(verifier)
    except Exception as exc:
        raise BadRequestError(f"unknown webhook verifier: {verifier!r}") from exc
    await get_hooks_manager().set_topic_verifier(topic, {"verifier": verifier, "config": config or {}})
    return {"topic": topic, "verifier": verifier}


@operation(summary="Unbind a topic's webhook verifier", tags=["hooks"], errors=[NotFoundError])
async def delete_topic_verifier(topic: str) -> dict[str, Any]:
    """Remove a topic's verifier binding; a missing binding is a loud 404."""
    removed = await get_hooks_manager().delete_topic_verifier(topic)
    if not removed:
        raise NotFoundError(f"no verifier bound to topic: {topic!r}")
    return {"removed": True, "topic": topic}


# -- Trigger links -----------------------------------------------------------
#
# A trigger link is a PUBLIC capability URL that fires a hook topic; the CRUD lives
# under the ``hooks`` tag (grantable read/write). Both mutating ops are
# ``authority_changing`` so an agent never silently mints or revokes a public
# capability through the MCP tool surface. The redis hooks backend is required — an
# in-memory deployment refuses with a loud 501 (``NotSupportedError``), a capability
# the deployment does not provide, distinct from a transient 503 outage.


@operation(
    summary="Create a trigger link",
    tags=["hooks"],
    destructive=True,
    authority_changing=True,
    errors=[BadRequestError, ConflictError, NotSupportedError],
    request_model=TriggerLinkCreate,
)
async def create_trigger_link(
    topic: str, name: str | None, ttl_seconds: int | None, tool_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """Mint a trigger link for ``topic``.

    ``ttl_seconds`` is the creator's explicit choice (``null`` permanent, positive
    timed; ``0``/negative → 400); a unique name is generated when omitted; a
    verifier-bound topic is refused (400); ``tool_kwargs`` rides every
    fire. Returns ``{"name", "trigger_path", "token", "topic", "expires_at"}``
    — the token appears ONLY here (nothing else stores or lists it)."""
    created_by = await _resolve_trigger_caller_id()
    try:
        return await trigger_links.create_trigger_link(
            topic=topic, name=name, ttl_seconds=ttl_seconds, tool_kwargs=tool_kwargs, created_by=created_by
        )
    except TriggerLinkError as exc:
        raise _map_trigger_error(exc) from exc


@operation(summary="List trigger links", tags=["hooks"], errors=[NotSupportedError])
async def list_trigger_links() -> dict[str, Any]:
    """Every live trigger link's record plus its hash PREFIX — never a raw token
    (none is stored; a listed link's QR is unrecoverable by design). Returns
    ``{"items", "total"}``."""
    try:
        return await trigger_links.list_trigger_links()
    except TriggerLinkError as exc:
        raise _map_trigger_error(exc) from exc


@operation(
    summary="Delete a trigger link",
    tags=["hooks"],
    destructive=True,
    authority_changing=True,
    errors=[NotFoundError, NotSupportedError],
)
async def delete_trigger_link(name: str) -> dict[str, Any]:
    """Revoke a trigger link by name — immediate and DURABLE (a permanent tombstone
    keeps a restored backup from re-arming it). An unknown name is a loud 404.
    Returns ``{"removed", "name"}``."""
    try:
        await trigger_links.revoke_trigger_link(name)
    except TriggerLinkError as exc:
        raise _map_trigger_error(exc) from exc
    return {"removed": True, "name": name}
