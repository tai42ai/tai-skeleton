"""Hook-management operations — the authed hooks surface the Studio drives, over
the shared ``get_hooks_manager()``.

* ``list_hooks`` lists registered hooks (filtered to ``topic`` when given) plus the
  per-topic verifier bindings and each covered topic's derived ``trigger_auth`` — one
  read the Studio hooks UI consumes.
* ``register_hook`` registers a hook from its FLAT parameters (the client-facing
  ``HookRegister`` fields spread as arguments; the server-derived fingerprint is not
  among them) — an upsert, so it is the edit path too.
* ``unregister_hook`` removes a hook by name; an unknown name is a loud 404.
* ``list_verifiers`` lists the registered webhook-verifier names (the bind catalog).
* ``set_topic_verifier`` binds (or replaces) a topic's webhook verifier; an unknown
  verifier name is rejected at bind time (400).
* ``delete_topic_verifier`` removes a topic's binding — reopening its public ingress
  door to anyone who knows the topic name; a missing binding is a 404.

These operations are the single source the ``/api/hooks*`` routes, the ``tai hooks``
CLI, and the MCP tool surface derive from — one definition behind every edge. Not all of
them project as MCP tools by DEFAULT: the five ``authority_changing`` ones —
``register_hook``, ``set_topic_verifier``, ``delete_topic_verifier``,
``create_trigger_link`` and ``delete_trigger_link`` — are tier-2 and reachable as tools
only through an explicit ``api_tools.include``, because each wires, mints or revokes
authority on a public URL, or decides the lock on one. The rest project normally, and
the HTTP doors and the CLI cover every one of them either way. A hook's ``tool_kwargs``
can carry caller-supplied values, so the list read is secret-bearing; that is acceptable
on this authed surface (as with config env).

The two verifier operations are additionally ADMIN-ONLY: their routes are ``fenced``, so
the per-tag level pass hard-fences them at the HTTP edge and at both tool edges. A
binding is the only authentication a topic's public ingress door has, and the topic
namespace carries no owner to key a per-caller rule on.

Both record writers — hook registration (an upsert, so the create AND the edit path)
and trigger-link mint — bind an ``execution_key``. A HOOK's key is the identity that
hook's fire runs as; a LINK's key is the identity its dispatch is gated on, the hooks
it fans out to each still firing as their own. Binding either is a pass-role decision
(:func:`~tai42_skeleton.operations._authority.assert_execution_key_bindable`), taken
BEFORE the write, so a refusal leaves no record behind.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams, HookRegister

from tai42_skeleton.hooks import trigger_links
from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.trigger_auth import webhook_trigger_auth
from tai42_skeleton.hooks.trigger_links import TriggerLinkError
from tai42_skeleton.operations import BadRequestError, NotFoundError, operation
from tai42_skeleton.operations._authority import assert_execution_key_bindable, resolve_caller
from tai42_skeleton.operations.errors import ConflictError, ForbiddenError, NotSupportedError, OperationError


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
    product ceiling. ``execution_key`` is the api-key identity the link's dispatch is
    gated on — the link dies with the key — and the creator must be allowed to
    delegate it; each hook the dispatch reaches still fires as its OWN bound key.
    ``require_api_key`` makes the link's door demand an authenticated principal ON TOP
    of the token — one the authentication backend admits, and, when that principal is
    governed by a ROLE, one the ordinary ``hooks``-tag level pass admits at the request's
    method (``read`` for GET, ``write`` for POST); the governing policy is the OWNER's for
    an owned key, so a key escapes that pass exactly when its governing policy is admin or
    carries no role pointer. The
    default is token-only — the QR-on-a-wall case. ``topic`` must be non-empty, the same
    rule the stored record enforces, so the mint door never writes a link its own restore
    would refuse. ``tool_kwargs`` (optional) is stored on the link and merged into every
    fired hook's input BELOW that hook's own static ``tool_kwargs``, so it supplies only
    the arguments the hook's author left unpinned — a colliding key stays the author's."""

    topic: str = Field(min_length=1)
    execution_key: str = Field(min_length=1)
    name: str | None = None
    ttl_seconds: int | None = Field(strict=True)
    tool_kwargs: dict[str, Any] | None = None
    require_api_key: bool = False


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


@operation(summary="List registered hooks", tags=["hooks"])
async def list_hooks(topic: str | None = None) -> dict[str, Any]:
    """List registered hooks plus the per-topic verifier bindings.

    With a topic, lists only the hooks registered for that topic; without one,
    lists every registered hook. The response carries the hooks under ``items``
    (with ``total``), the per-topic verifier bindings under ``topic_verifiers``, and
    under ``trigger_auth`` how a topic's webhook ingress door authenticates its caller —
    DERIVED from the live verifier bindings just read, never stored. ``trigger_auth`` is
    keyed by every topic among the LISTED hooks plus every topic currently carrying a
    verifier binding (one may be bound before any hook exists on it), so under a
    ``topic`` filter it is neither a subset nor a superset of the listed topics.
    """
    manager = get_hooks_manager()
    hooks = await manager.list_hooks_by_topic(topic=topic) if topic else await manager.list_hooks()
    items = [params.model_dump(mode="json") for params in hooks.values()]
    topic_verifiers = await manager.all_topic_verifiers()
    # Every topic the reader can see here: the listed hooks' topics plus every topic
    # carrying a binding (one may be bound before any hook is registered on it).
    topics = {params.topic for params in hooks.values()} | set(topic_verifiers)
    trigger_auth = {name: webhook_trigger_auth(verifier_bound=name in topic_verifiers) for name in sorted(topics)}
    return {"items": items, "total": len(items), "topic_verifiers": topic_verifiers, "trigger_auth": trigger_auth}


# ``authority_changing``: registering binds an api-key identity onto a topic the PUBLIC
# ``/universal_webhook/{topic}`` door fires unauthenticated.
@operation(
    summary="Register a hook",
    tags=["hooks"],
    destructive=True,
    authority_changing=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError],
    request_model=HookRegister,
)
async def register_hook(
    name: str,
    topic: str,
    tool: str,
    execution_key: str,
    tool_kwargs: dict[str, Any] | None = None,
    condition: str | None = None,
    condition_id: str | None = None,
    condition_kwargs: dict[str, Any] | None = None,
    expr: str | None = None,
    expr_id: str | None = None,
    expr_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a hook from its flat parameters — an UPSERT, so this is the create
    path AND the edit path for a hook of that name.

    Supports conditional execution (via ``condition`` or ``condition_id``), payload
    transformation (via ``expr`` or ``expr_id``), and dynamic tool arguments.
    ``execution_key`` is the api-key identity every fire runs as; the caller's authority
    to delegate it and its usability by a tokenless fire are decided BEFORE the upsert, so
    a refusal stores nothing and leaves any existing hook of that name untouched. Returns
    ``{"registered", "name"}``; ``registered`` is ``True`` for a create and a replace
    alike.
    """
    execution_key_fingerprint = await assert_execution_key_bindable(await resolve_caller(), execution_key)
    try:
        params = HookParams(
            name=name,
            topic=topic,
            tool=tool,
            execution_key=execution_key,
            execution_key_fingerprint=execution_key_fingerprint,
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


# ``authority_changing``: binding is also REBINDING — it replaces the lock a topic
# already carries, so fencing only the unbind would leave the rebind as its bypass.
@operation(
    summary="Bind a webhook verifier to a topic",
    tags=["hooks"],
    destructive=True,
    authority_changing=True,
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


# ``authority_changing``: the binding is the ONLY authentication
# ``/universal_webhook/{topic}`` has — unbinding reopens that public door and takes every
# out-of-service trigger link on the topic back into service.
@operation(
    summary="Unbind a topic's webhook verifier",
    tags=["hooks"],
    destructive=True,
    authority_changing=True,
    errors=[NotFoundError],
)
async def delete_topic_verifier(topic: str) -> dict[str, Any]:
    """Remove a topic's verifier binding, reopening its public ingress door; a missing
    binding is a loud 404."""
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
    errors=[BadRequestError, ConflictError, ForbiddenError, NotFoundError, NotSupportedError],
    request_model=TriggerLinkCreate,
)
async def create_trigger_link(
    topic: str,
    execution_key: str,
    name: str | None,
    ttl_seconds: int | None,
    tool_kwargs: dict[str, Any] | None,
    require_api_key: bool = False,
) -> dict[str, Any]:
    """Mint a trigger link for ``topic``.

    ``ttl_seconds`` is the creator's explicit choice (``null`` permanent, positive
    timed; ``0``/negative → 400); a unique name is generated when omitted; a
    verifier-bound topic is refused (400); ``tool_kwargs`` rides every fire, filling
    only the arguments each fired hook's author left
    unpinned. ``execution_key`` is the api-key identity the link's dispatch is gated on,
    decided BEFORE the mint so a refusal leaves no live URL. ``require_api_key`` makes
    the link's door demand an authenticated caller beside the token. Returns
    ``{"name", "trigger_path", "token", "topic", "expires_at"}``
    — the token appears ONLY here (nothing else stores or lists it).

    ``created_by`` is stamped from the AMBIENT caller identity, never a request field
    (which would be caller-spoofable); with the gate off there is no principal and it
    is stored ``null``."""
    caller = await resolve_caller()
    execution_key_fingerprint = await assert_execution_key_bindable(caller, execution_key)
    try:
        return await trigger_links.create_trigger_link(
            topic=topic,
            name=name,
            ttl_seconds=ttl_seconds,
            tool_kwargs=tool_kwargs,
            execution_key=execution_key,
            execution_key_fingerprint=execution_key_fingerprint,
            require_api_key=require_api_key,
            created_by=caller.caller_id,
        )
    except TriggerLinkError as exc:
        raise _map_trigger_error(exc) from exc


@operation(summary="List trigger links", tags=["hooks"], errors=[NotSupportedError])
async def list_trigger_links() -> dict[str, Any]:
    """Every live trigger link's record plus its hash PREFIX and its derived
    ``trigger_auth`` — never a raw token (none is stored; a listed link's QR is
    unrecoverable by design). Returns ``{"items", "total"}``."""
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
