"""Operations for the access-control keys/scopes surface — the authed doors the
Studio's API-keys settings tab consumes, projected from one declaration.

Nineteen operations for the access-control keys/scopes surface: the scope
catalog + CRUD, the route catalog, the public-route pins, the key CRUD (create /
edit / revoke), one-time claim-link creation, the mint-capability + role reads, the
caller's own capability projection (``get_me``), the fail-closed condition validator,
and the admin-only policy version-history + rollback. Most route their core work
through the access-control ``management`` module; ``create_claim_link`` delegates to
:mod:`~tai42_skeleton.access_control.claim_links` and ``get_me`` to
:mod:`~tai42_skeleton.access_control.projection`. Every one is a route under
``/api/auth/*``, so the eighteen non-``get_me`` ops are tier-2 (default-excluded from the MCP surface,
includable by an explicit ``api_tools.include``) by the projection module's
route-prefix predicate — no ``authority_changing`` flag needed. ``get_me`` is
tier-1 (``caller_context=True``, NEVER projectable): its params are the caller's own
edge-derived identity, which an MCP caller could supply to spoof another principal.

**Owner-aware ownership rules.** The caller is resolved from the request-scoped
``user_id`` contextvar and classified admin/non-admin: admin iff a condition-free
``"*"`` policy that is not itself an owned key. A non-admin may create only self-owned
keys with scopes ⊆ its own; may edit/revoke only keys it owns; and sees only its own
keys in ``tokens-payload``. An owned key can mint nothing. A key's owner claim is
immutable through the edit surface (re-mint to change ownership). With the gate off the
caller is treated as admin (nothing to attenuate against).

Every mutation bumps the policy version so a running worker's policy cache re-reads
the edit instead of serving a stale copy. Each policy write is ordered
enforced-store first (the ``management`` write lands, then the cache-buster bump,
then the durable PG version history) so enforcement is current the instant the
authority changes even if the audit write then fails.
"""

from __future__ import annotations

from typing import Any

from jinja2 import TemplateError
from pydantic import BaseModel, Field, ValidationError
from starlette.routing import Mount, Route
from tai42_contract.access_control.models import JqAuthContext
from tai42_contract.app import tai42_app
from tai42_contract.versioning.errors import DocumentNotFoundError, DocumentVersionNotFoundError
from tai42_kit.utils.data import run_jq_first
from tai42_kit.utils.data.jq_util import get_compiled_jq

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.claim_links import ClaimLinkError
from tai42_skeleton.access_control.claim_links import create_claim_link as _create_claim_link
from tai42_skeleton.access_control.policy_store import ac_policy_store
from tai42_skeleton.access_control.projection import ProjectionResult, build_projection, synthetic_full_projection
from tai42_skeleton.access_control.roles import role_store
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.operations import BadRequestError, ForbiddenError, NotFoundError, operation
from tai42_skeleton.operations._authority import (
    Caller,
    owner_of,
    require_admin,
    require_owned_by_caller,
    resolve_caller,
)
from tai42_skeleton.template import TemplateNotFoundError

# -- Request models (spec metadata; the route extractors do the byte-stable parse) --


class ScopeUrlAdd(BaseModel):
    """Add a URL (optionally a match ``pattern``) to a scope."""

    scope_id: str
    url: str
    pattern: str | None = None


class ScopeUrlRemove(BaseModel):
    """Remove a URL from every scope that references it."""

    url: str


class ApiKeyCreate(BaseModel):
    """Create an api key for ``user_id`` with a scope set and an optional jq
    authorization condition (inline ``condition`` or stored ``condition_id``)."""

    user_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scopes: list[str]
    policy_data: dict[str, Any] | None = None
    condition: str | None = None
    condition_id: str | None = None
    condition_kwargs: dict[str, Any] | None = None
    # The account that owns this key. A non-admin caller may only mint self-owned keys
    # (an explicit different owner is rejected); an admin may set any owner or None.
    owner_user_id: str | None = None


class ApiKeyEdit(BaseModel):
    """A partial api-key edit — only the fields present are overwritten; a
    ``null``/``{}``/``""`` value clears an optional gate."""

    description: str | None = Field(default=None, min_length=1)
    scopes: list[str] | None = None
    policy_data: dict[str, Any] | None = None
    condition: str | None = None
    condition_id: str | None = None
    condition_kwargs: dict[str, Any] | None = None


class ConditionValidation(BaseModel):
    """A fail-closed jq policy-condition check — compile and (with a
    ``sample_context``) sample-evaluate a condition without persisting it."""

    condition: str | None = None
    condition_id: str | None = None
    condition_kwargs: dict[str, Any] | None = None
    sample_context: dict[str, Any] | None = None


class PolicyRollback(BaseModel):
    """Re-point a user's enforced policy to a prior version."""

    version: int


class ClaimLinkCreate(BaseModel):
    """Create a one-time claim link for an existing API key. The ``api_key`` is a raw
    key the caller holds; ``ttl_seconds`` overrides the default lifetime (capped at the
    settings ceiling)."""

    api_key: str = Field(min_length=1)
    ttl_seconds: int | None = None


class PublicRoutePin(BaseModel):
    """Pin a URL public (optionally with a dynamic match ``pattern``)."""

    url: str
    pattern: str | None = None


class PublicRouteUnpin(BaseModel):
    """Unpin a public URL."""

    url: str


# -- Ownership rules ---------------------------------------------------------


def _check_scope_subset(caller: Caller, scopes: list[str]) -> None:
    """A non-admin caller may only grant scopes ⊆ its OWN current scopes (a ``"*"``
    caller may grant anything). Raises ``BadRequestError`` naming the offending scopes."""
    if "*" in caller.policy.scopes:
        return
    excess = sorted(set(scopes) - set(caller.policy.scopes))
    if excess:
        raise BadRequestError(f"requested scopes exceed your own: {excess}")


async def _record_policy_version(user_id: str, body: dict[str, Any]) -> None:
    """Record ``body`` — the exact policy the mutation just committed to the enforced
    store — as durable version history.

    The mutation returns the body it wrote inside its own transaction, so this
    appends that precise body to the ``ac_policy`` document (create-or-append)
    without re-reading the store: two concurrent edits A→B each record their own body
    rather than both reading B and dropping A's version. A no-op when the body is
    unchanged (e.g. a description-only key edit re-writes an identical policy
    record), so history is not polluted. Any store error propagates loudly — the
    enforced store then leads the history (the safe direction: enforcement is already
    current, since the bump ran first), and the operator is told the audit write
    failed rather than it being swallowed."""
    await ac_policy_store().write(user_id, body)


# -- Scopes ------------------------------------------------------------------


@operation(summary="List all scopes", tags=["access-control"])
async def list_scopes() -> dict[str, str]:
    """Every non-public route mapping as ``{url: scope_id}``."""
    return await management.get_all_existing_scopes()


@operation(
    summary="Add a URL to a scope",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError],
    request_model=ScopeUrlAdd,
)
async def add_scope_url(scope_id: str, url: str, pattern: str | None) -> dict[str, str]:
    """Map ``url`` to ``scope_id`` (optionally with a dynamic match ``pattern``)."""
    marker = access_control_settings().public_resource_id
    if scope_id == marker:
        # The public marker is a column value, not a scope. Routing it through the
        # generic scope setter would write a public pin behind the scope machinery's
        # back; the dedicated public-routes door is the only public-pin writer.
        raise BadRequestError(f"{marker!r} is the public marker, not a scope; use POST /api/auth/public-routes")
    try:
        await management.add_url_to_scope(scope_id, url, pattern)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    await management.bump_policy_version()
    return {"scope_id": scope_id, "url": url}


@operation(
    summary="Remove a URL from all scopes",
    tags=["access-control"],
    errors=[BadRequestError, NotFoundError],
    request_model=ScopeUrlRemove,
)
async def remove_scope_url(url: str) -> dict[str, str]:
    """Unmap ``url`` from every scope that references it; a url that was never mapped
    is a loud 404 (a typo, not a silent success)."""
    existed, affected = await management.remove_url_from_scope(url)
    if not existed:
        raise NotFoundError(f"url not mapped: {url!r}")
    # The store cascade has landed; bump the cache-buster first so enforcement follows
    # immediately, then record each rewritten policy as a new PG version so the durable
    # history's ``is_current`` stays honest against enforcement (a rollback target that
    # still listed the removed scope would silently re-grant it).
    await management.bump_policy_version()
    for affected_user, body in affected:
        await _record_policy_version(affected_user, body)
    return {"url": url}


@operation(summary="Delete a scope", tags=["access-control"], errors=[BadRequestError, NotFoundError])
async def delete_scope(scope_id: str) -> dict[str, Any]:
    """Delete a scope, cascading it out of every referencing key; an unknown scope
    (no urls) is a loud 404."""
    try:
        deleted, affected = await management.remove_scope(scope_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    if deleted == 0:
        raise NotFoundError(f"scope not found: {scope_id!r}")
    # The delete cascades the scope out of every referencing key's stored policy; bump
    # the cache-buster first so enforcement follows immediately, then record each
    # rewritten policy as a new PG version so the durable history's ``is_current`` stays
    # honest against enforcement.
    await management.bump_policy_version()
    for affected_user, body in affected:
        await _record_policy_version(affected_user, body)
    return {"scope_id": scope_id, "deleted_keys": deleted}


# -- Route catalog -----------------------------------------------------------


@operation(summary="List the app's HTTP routes and their scope mappings", tags=["access-control"])
async def list_routes(routes: list[Any]) -> list[dict[str, Any]]:
    """Enumerate the app's own HTTP routes with each route's current scope mapping —
    the mapper's route picker and its "unassigned routes" bucket (the ``mapped: null``
    entries).

    ``routes`` is the app's live route table (the route-adapter extractor hands the
    operation ``request.app.routes`` so it stays request-free). One entry per
    :class:`starlette.routing.Route`, sorted by ``path``. ``Mount`` entries (the sub-MCP
    mount, the MCP mount) are EXCLUDED — a mount is not a bindable url and the sub-MCP
    surface has its own door. The exclusion is an explicit ``isinstance`` filter: any
    route-table entry that is neither a ``Route`` nor a ``Mount`` raises loudly (a new
    Starlette routing type must be classified here, never silently dropped). ``methods``
    is the route's method set sorted with ``HEAD`` removed (Starlette auto-adds it to
    every GET route — noise for the mapper). ``mapped`` is the url's value from
    ``get_all_route_mappings`` looked up by the EXACT path string — a scope id, the
    public marker for a public pin, or ``null`` when the path has no mapping. Exact-key
    lookup only: this door does not attempt dynamic-pattern matching.

    Each entry is JOINED with its :class:`RouteMetadata` (from
    ``route_registry.load_all_routes()``, a separate collection) to also carry the
    route's feature ``tags`` + ``summary`` + authorization ``action`` — the data the
    Studio Roles page groups the per-tag tri-state by and marks the ``fenced``/``secret``
    (admin-only) routes it must never offer a grant for. The join is keyed on the route
    TEMPLATE + its method set with ``HEAD`` stripped on BOTH sides (Starlette auto-adds
    ``HEAD`` to a GET), so the two collections compare like-for-like; a registered route
    whose method set fails to join is a loud STOP (a normalization drift), never a
    silently dropped row."""
    from tai42_skeleton.app.route_registry import load_all_routes

    mappings = await management.get_all_route_mappings()
    all_meta = load_all_routes()
    meta_by_key = {(meta.path, frozenset(m for m in meta.methods if m != "HEAD")): meta for meta in all_meta}
    registered_paths = {meta.path for meta in all_meta}
    entries: list[dict[str, Any]] = []
    for route in routes:
        if isinstance(route, Mount):
            continue
        if not isinstance(route, Route):
            raise TypeError(
                f"unclassified route-table entry {type(route).__name__!r}: {route!r} — a new starlette "
                "routing type must be classified in list_routes, not silently dropped"
            )
        methods = sorted(m for m in (route.methods or set()) if m != "HEAD")
        entry: dict[str, Any] = {"path": route.path, "methods": methods, "mapped": mappings.get(route.path)}
        meta = meta_by_key.get((route.path, frozenset(methods)))
        if meta is not None:
            entry.update(tags=list(meta.tags), summary=meta.summary, action=meta.action)
        elif route.path in registered_paths:
            # The PATH is a registered route but this method set did not join — a genuine
            # method-normalization drift between the two collections. A loud STOP, never a
            # silently dropped metadata row.
            raise TypeError(
                f"gated route {'/'.join(methods)} {route.path} did not join its route-registry metadata — "
                "the method-set normalization drifted; a gated route must never be a dropped row"
            )
        else:
            # A path not in the registry at all carries no metadata to join (the boot
            # action-class audit guarantees every registered gated route is classified).
            entry.update(tags=[], summary="", action=None)
        entries.append(entry)
    entries.sort(key=lambda entry: entry["path"])
    return entries


# -- Public route pins -------------------------------------------------------


@operation(summary="List public-pinned routes", tags=["access-control"])
async def list_public_routes() -> list[str]:
    """Every route pinned to the public marker."""
    return await management.get_public_route_pins()


@operation(
    summary="Pin a route public",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError],
    request_model=PublicRoutePin,
)
async def pin_public_route(url: str, pattern: str | None) -> dict[str, str]:
    """Pin ``url`` public (optionally with a dynamic match ``pattern``)."""
    try:
        await management.pin_route_public(url, pattern)
    except ValueError as exc:
        # A url under a reserved management prefix cannot be pinned public — the
        # control plane must not be usable to de-authenticate itself.
        raise BadRequestError(str(exc)) from exc
    # Bump AFTER the write so the version-keyed enforcer route cache re-reads the pin.
    await management.bump_policy_version()
    return {"url": url}


@operation(
    summary="Unpin a public route",
    tags=["access-control"],
    errors=[BadRequestError, NotFoundError],
    request_model=PublicRouteUnpin,
)
async def unpin_public_route(url: str) -> dict[str, str]:
    """Unpin a public ``url``; a url that is absent or scope-mapped is a loud 404."""
    if not await management.unpin_public_route(url):
        raise NotFoundError(f"url is not pinned public: {url!r}")
    await management.bump_policy_version()
    return {"url": url}


# -- Keys --------------------------------------------------------------------


@operation(summary="List api-key token payloads", tags=["access-control"])
async def list_tokens_payload() -> list[dict[str, Any]]:
    """Every provisioned key's identity + policy (NEVER key material). Non-admin callers
    see ONLY the keys they own (management/listing owner home); admin sees every key."""
    caller = await resolve_caller()
    payload = await management.get_all_existing_tokens_payload()
    if not caller.is_admin:
        payload = [p for p in payload if owner_of(p.get("policy_data")) == caller.caller_id]
    return payload


@operation(
    summary="Create an api key",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError, ForbiddenError],
    request_model=ApiKeyCreate,
)
async def create_api_key(
    user_id: str,
    description: str,
    scopes: list[str],
    policy_data: dict[str, Any] | None,
    condition: str | None,
    condition_id: str | None,
    condition_kwargs: dict[str, Any] | None,
    owner_user_id: str | None,
) -> dict[str, Any]:
    """Provision a key, returning ``{"api_key", "key_fingerprint"}``. The raw ``sk-…``
    ``api_key`` is surfaced ONCE; ``key_fingerprint`` is the key's immutable per-mint
    identity a caller binds a hook against so the binding survives only this exact mint."""
    caller = await resolve_caller()
    # An owned key cannot mint keys — ownership is exactly one level deep.
    if caller.owner_claim is not None:
        raise ForbiddenError("an owned API key may not mint API keys")
    if not caller.is_admin:
        # Non-admin: force self-ownership (reject an explicit different owner, never
        # silently overwrite) and cap the grant to the caller's own scopes.
        if owner_user_id is not None and owner_user_id != caller.caller_id:
            raise ForbiddenError("a non-admin caller may only create keys owned by itself")
        owner_user_id = caller.caller_id
        _check_scope_subset(caller, scopes)

    try:
        raw_key, committed_body, key_fingerprint = await management.add_user_api_key(
            user_id=user_id,
            description=description,
            scopes=scopes,
            policy_data=policy_data,
            condition=condition,
            condition_id=condition_id,
            condition_kwargs=condition_kwargs,
            owner_user_id=owner_user_id,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    # Store-first: ``add_user_api_key`` has written the policy to the enforced store
    # (the authority) and returned the exact body it committed. Bump the
    # cache-invalidation key immediately so enforcement follows, then record that body
    # as durable PG history. A store failure above raised before this, so neither the
    # bump nor the history is touched.
    await management.bump_policy_version()
    await _record_policy_version(user_id, committed_body)
    return {"api_key": raw_key, "key_fingerprint": key_fingerprint}


@operation(
    summary="Edit an api key",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError],
    request_model=ApiKeyEdit,
)
async def edit_api_key(user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """A PATCH-style partial edit: only the fields present in ``updates`` are
    overwritten; a field absent is preserved at its stored value, so saving a
    description or scope change never silently drops an authorization ``condition`` or
    ``policy_data`` gate. ``updates`` is the sparse set of present fields (a single dict
    rather than flattened params, so "field absent" stays distinct from "field is
    ``null``" — the partial-edit semantics a flat signature cannot express)."""
    caller = await resolve_caller()
    # Ownership + owner-claim immutability pre-checks, reading the stored body once when
    # any check needs it (a non-admin ownership gate, or a policy_data edit whose owner
    # claim must not change).
    if (not caller.is_admin) or ("policy_data" in updates):
        stored_body = await management.get_policy_body(user_id)
        if stored_body is None:
            raise NotFoundError(f"user not found: {user_id!r}")
        stored_owner = owner_of(stored_body.get("policy_data"))
        if not caller.is_admin:
            if stored_owner != caller.caller_id:
                raise ForbiddenError("you may only edit API keys you own")
            if "scopes" in updates:
                _check_scope_subset(caller, updates["scopes"])
        if "policy_data" in updates:
            # Echo-tolerant immutability: an unchanged owner claim is accepted (Studio
            # echoes policy_data back verbatim), but a CHANGED, newly-introduced, or
            # absent/cleared owner claim is rejected — ownership never changes post-mint
            # (re-mint instead), and a silent strip would orphan the owner's visibility.
            new_owner = owner_of(updates["policy_data"])
            if new_owner != stored_owner:
                raise ForbiddenError("the owner of an API key is immutable; re-mint to change ownership")

    try:
        updated = await management.edit_user_payload(user_id=user_id, **updates)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    if not updated:
        raise NotFoundError(f"user not found: {user_id!r}")
    # Store-first: the edit has landed in the enforced store and returned the exact
    # committed body. Bump the cache key immediately so enforcement follows, then
    # record that body as a new PG version (see ``_record_policy_version``).
    await management.bump_policy_version()
    await _record_policy_version(user_id, updated)
    return {"user_id": user_id, "updated": True}


@operation(
    summary="Revoke an api key",
    tags=["access-control"],
    errors=[BadRequestError, ForbiddenError, NotFoundError],
)
async def revoke_api_key(user_id: str) -> dict[str, Any]:
    """Revoke a key (immediate: next request fails to auth). Deletes the key record, its
    enforced policy row, and its live context; the user's ``ac_policy`` version history
    is deliberately NOT touched (it belongs to the identity, so a key later re-created
    for the same ``user_id`` resumes that history).

    No version bump here, unlike every other mutation on this surface: revocation's
    cache-buster is atomic with the policy-row delete inside
    :func:`~tai42_skeleton.access_control.management.revoke_api_key`, so no fault in the
    steps behind it can leave the revoked key's authority live in a warm cache slot."""
    caller = await resolve_caller()
    if not caller.is_admin:
        # A non-admin may revoke only a key it owns.
        await require_owned_by_caller(caller, user_id)
    try:
        revoked = await management.revoke_api_key(user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    if not revoked:
        raise NotFoundError(f"user not found: {user_id!r}")
    return {"user_id": user_id, "revoked": True}


@operation(
    summary="Create a one-time claim link for an API key",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError, ForbiddenError],
    request_model=ClaimLinkCreate,
)
async def create_claim_link(api_key: str, ttl_seconds: int | None) -> dict[str, Any]:
    """Mint a one-time claim link that carries ``api_key`` to another device (the QR
    onboarding leg). The submitted key is resolved through the gate's own verifier chain
    and the caller must own it (or be admin) per the module's ownership rule; the response
    returns the claim token ONCE plus a fragment-carrier path (``/login#claim=<token>``)
    and an expiry.

    Accepted oracle (deliberate, not an oversight): an unresolvable key answers 400 and a
    valid-but-not-yours key answers 403, so an authenticated caller can tell a live key
    from garbage. This adds NO capability the ``/api/auth/me`` carve-out does not already
    grant a caller holding a candidate key. The uniform-404 no-oracle rule governs the
    unauthenticated EXCHANGE surface, never this authed creation."""
    caller = await resolve_caller()
    try:
        return await _create_claim_link(
            api_key=api_key,
            caller_id=caller.caller_id,
            caller_is_admin=caller.is_admin,
            caller_owner_claim=caller.owner_claim,
            ttl_seconds=ttl_seconds,
        )
    except ClaimLinkError as exc:
        # The store raises 400 (unresolvable key / bad ttl) or 403 (not the caller's key);
        # map each to the operation error the adapter renders at that status.
        if exc.status == 403:
            raise ForbiddenError(exc.message) from exc
        raise BadRequestError(exc.message) from exc


# -- Mint capabilities + role templates --------------------------------------


@operation(summary="Report key-mint capabilities", tags=["access-control"])
async def get_capabilities() -> dict[str, Any]:
    """Whether any configured identity provider can MINT keys, per provider. Lets the
    Studio disable mint UI with a clear message on a validator-only deployment instead
    of surfacing a raw error at mint time."""
    capabilities = management.provider_capabilities()
    providers = [{"name": name, "mintable": mintable} for name, mintable in capabilities]
    return {"mintable": any(m for _, m in capabilities), "providers": providers}


@operation(summary="List roles", tags=["access-control"], errors=[ForbiddenError])
async def list_roles() -> list[dict[str, Any]]:
    """The seeded/operator-authored roles as full ``RoleDefinition``-shaped bodies
    (``{name, description, scopes, condition, condition_id, condition_kwargs, base_tier,
    allow_all, grants}``) — the users-admin role picker and the Studio Roles page read
    this. A store-less deployment (no versioned store configured) has no roles — the seed
    step is skipped at boot — so the read is skipped and the list is empty.

    Admin-only: a listing exposes every role's raw base-tier jq condition, so a non-admin
    caller is denied 403. This op-level check is defense in depth behind the ``secret``
    route action-class that already fences the route."""
    caller = await resolve_caller()
    require_admin(caller)
    from tai42_skeleton.versioning import versioned_store_configured

    if not versioned_store_configured():
        return []
    return await role_store().list_roles()


# -- Caller capability projection --------------------------------------------


@operation(
    summary="The caller's capability projection",
    tags=["access-control"],
    caller_context=True,
    response_model=ProjectionResult,
)
async def get_me(
    user_id: str | None, effective_scopes: list[str] | None, claims: dict[str, Any] | None
) -> ProjectionResult:
    """The authenticated caller's derived capability projection — the concrete routes,
    dynamic patterns, sub-MCP mounts, tools, and agents it can reach right now (derived,
    never stored).

    ``user_id``/``effective_scopes``/``claims`` are the caller's OWN identity, derived at
    the HTTP edge from the authenticated request — never caller-supplied. This is
    ``caller_context=True`` (tier-1, never projectable): as an MCP tool a caller would
    supply those identity params itself and read ANY principal's projection, so the HTTP
    route (``/api/auth/me``, whose extractor derives them from ``request.user``) is the
    only surface. With the gate OFF there is no identity to project (the edge passes
    ``user_id=None``), so a synthetic TOTAL projection is returned; otherwise the
    projection is built through the REAL enforcer so it can never advertise a door the
    gate would deny. Any infrastructure error propagates per the projection's failure
    doctrine."""
    if user_id is None:
        return synthetic_full_projection()
    return await build_projection(user_id, effective_scopes or [], claims or {})


# -- Policy condition validation --------------------------------------------


@operation(
    summary="Validate a jq policy condition",
    tags=["access-control"],
    errors=[BadRequestError],
    request_model=ConditionValidation,
)
async def validate_condition(
    condition: str | None,
    condition_id: str | None,
    condition_kwargs: dict[str, Any] | None,
    sample_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail-closed guard: compile — and optionally sample-evaluate — a jq policy
    condition WITHOUT persisting it.

    A syntactically broken condition raises at enforcement and DENIES the key (a
    lock-out), so authoring flows validate here before saving. The condition is
    rendered exactly as enforcement renders it, compiled with ``get_compiled_jq``, and —
    when a ``sample_context`` is supplied — evaluated against a ``JqAuthContext``-shaped
    sample. This ONLY compiles/evaluates; it never writes any store. Returns ``{"ok":
    true, "result": <bool|null>}`` (``result`` is ``null`` when no sample was evaluated).
    An AUTHOR error is a loud ``BadRequestError`` (400); a server-side fault (an
    unconfigured resource manager, a redis/storage outage rendering a stored
    ``condition_id``) is NOT an author error and propagates as a loud 500."""
    if condition and condition_id:
        raise BadRequestError("provide either 'condition' or 'condition_id', not both")

    # Mirror enforcement's own "was a condition configured?" test exactly
    # (``policy.condition is not None or policy.condition_id is not None``): a
    # PRESENT-but-empty ``condition``/``condition_id`` (e.g. ``""``) is configured and
    # denies at enforcement, so it must reach the render-empty lock-out branch below —
    # a truthiness test would wrongly treat ``""`` as "nothing configured" and pass it.
    configured = condition is not None or condition_id is not None
    try:
        rendered = await tai42_app.storage.resource_manager.render_by_id_or_content(
            content=condition, template_id=condition_id, kwargs=condition_kwargs
        )
        result: Any = None
        if rendered:
            # Compile-validate the expression (the compile half of this endpoint):
            # a broken expression raises here and lands in the verbatim-400 handler.
            get_compiled_jq(rendered)
            if sample_context is not None:
                # Enforcement allows ONLY when the jq emits exactly ``True`` (a truthy
                # non-``True`` value denies), so coerce to that same boolean here — the
                # sample result then honestly mirrors the allow/deny enforcement would
                # reach and stays a clean ``bool | null``. The evaluation runs off-loop
                # under a wall-clock budget (JQ_TIMEOUT_SECONDS).
                result = (await run_jq_first(rendered, JqAuthContext(**sample_context).model_dump())) is True
        elif configured:
            # A configured condition that renders to an EMPTY string denies at
            # enforcement (fail-closed), so reporting ``ok`` would tell the author a
            # lock-out condition is safe — the exact footgun this guard exists to
            # catch. Surface it as a loud validation failure instead.
            raise BadRequestError(
                "condition renders empty, which denies at enforcement and would lock the key out; "
                "author a condition that renders to a non-empty jq expression"
            )
    except (ValueError, ValidationError, TemplateError, TemplateNotFoundError) as exc:
        # AUTHOR errors only — the jq compile/eval ``ValueError`` (the jq lib's error
        # type), the pydantic ``ValidationError`` from a malformed ``sample_context``,
        # a jinja ``TemplateError`` from a broken inline condition, and
        # ``TemplateNotFoundError`` for a missing ``condition_id``. Their message is
        # the actionable feedback the author needs, surfaced verbatim as a 400. Any
        # other exception (an unconfigured resource manager ``RuntimeError``, a
        # redis/storage outage) is a server fault and propagates as a loud 500.
        raise BadRequestError(str(exc)) from exc
    return {"ok": True, "result": result}


# -- Policy version history + rollback --------------------------------------


@operation(
    summary="List a user's policy version history", tags=["access-control"], errors=[ForbiddenError, NotFoundError]
)
async def list_policy_versions(user_id: str) -> list[dict[str, Any]]:
    """The user's append-only policy version history from the durable PG store, each
    row flagged ``is_current`` against the active pointer. Secret-adjacent (a version
    body carries the raw condition) and admin-only: a non-admin caller is denied 403 so
    it can never read another user's policy history. 404 when the user has no policy
    history."""
    caller = await resolve_caller()
    require_admin(caller)
    # A store-less deployment (no versioned store configured) keeps no policy version
    # history, so short-circuit an empty list rather than let the store read raw-500 on
    # an absent Postgres backend.
    from tai42_skeleton.versioning import versioned_store_configured

    if not versioned_store_configured():
        return []
    try:
        versions = await ac_policy_store().list_versions(user_id)
    except DocumentNotFoundError as exc:
        raise NotFoundError(f"no policy history for user: {user_id!r}") from exc
    return [v.model_dump() for v in versions]


@operation(
    summary="Roll a policy back to a version",
    tags=["access-control"],
    destructive=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError],
    request_model=PolicyRollback,
)
async def rollback_policy(user_id: str, version: int) -> dict[str, Any]:
    """Re-point the enforced policy to a prior version. Store-first: the target version
    body is read from the history, written to the enforced store (the authority) FIRST;
    on that success the cache-invalidation key is bumped immediately so enforcement
    follows, then the durable history pointer is advanced. Admin-only: a non-admin caller
    is denied 403 so it can never roll back another user's (or its own) enforced policy.
    404 if the version is absent or the user has no live key."""
    caller = await resolve_caller()
    require_admin(caller)

    # A store-less deployment (no versioned store configured) keeps no policy version
    # history, so no version can exist to roll back to — return the same clean 404 an
    # absent version yields rather than let the store read raw-500 on an absent Postgres
    # backend.
    from tai42_skeleton.versioning import versioned_store_configured

    if not versioned_store_configured():
        raise NotFoundError(f"user {user_id!r} has no policy version {version}")

    store = ac_policy_store()
    try:
        target = await store.get_version(user_id, version)
    except DocumentVersionNotFoundError as exc:
        raise NotFoundError(f"user {user_id!r} has no policy version {version}") from exc

    # Store-first, then the cache bump, then the history pointer.
    try:
        restored = await management.restore_policy_body(user_id, target.body)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    if not restored:
        raise NotFoundError(f"user not found: {user_id!r}")
    await management.bump_policy_version()
    await store.rollback(user_id, version)
    return {"user_id": user_id, "active_version": version}
