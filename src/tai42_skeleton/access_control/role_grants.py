"""The LIVE role→grant-map resolution + the shared per-tag enforcement decision.

Editing a role affects EVERY holder LIVE: a user's enforced policy carries a separate
role-name POINTER (``policy_data[ROLE_POINTER_KEY]``, never ``condition_id`` — that
would collide with the admin discriminator), read here to fetch the role's CURRENT
grant map. The lookup rides a VERSION-KEYED cache busted by ``bump_policy_version`` on
any role edit, so a live edit lands on the next request with no uncached hot-path store
read.

:func:`role_level_decision_for_route` is the ONE shared per-tag decision the request gate
(:mod:`~tai42_skeleton.access_control.backend`), the capability projection
(:mod:`~tai42_skeleton.access_control.projection`) and the tool edge
(:mod:`~tai42_skeleton.authz.check`) all consume, so the projection can never advertise a
door the gate would deny and an operation is fenced identically as route or as tool. The
edges differ only in how the route reaches it: a real request path resolves through
:func:`role_level_decision`, while the tool edge pins the operation's own registered route
first — its path is synthesized from caller-supplied arguments, so an unresolvable path
there is a denial, not an ungated route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from async_lru import alru_cache
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy, RoleDefinition
from tai42_contract.versioning.errors import DocumentNotFoundError
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.access_control.role_gate import (
    DenialCause,
    RoleGrants,
    grant_map_admits,
    resolve_route_meta,
)
from tai42_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai42_skeleton.access_control.user import is_admin_policy

if TYPE_CHECKING:
    from tai42_skeleton.app.route_registry import RouteMetadata


async def _raw_resolve_grants(role_name: str, version: int) -> RoleGrants:
    # ``version`` participates only in the cache key (a role edit bumps it, forcing a
    # fresh read of the CURRENT grant map); the fetch itself is version-independent.
    # A missing/deleted role raises ``DocumentNotFoundError`` — a fail-closed deny at
    # the caller, never a silent empty grant that would read as "no access" yet not
    # distinguish a genuinely deleted role.
    from tai42_skeleton.access_control.roles import role_store

    body = await role_store().get_active_body(role_name)
    role = RoleDefinition(**body)
    return dict(role.grants)


_grants_cache = None


def _get_grants_cache(settings: AccessControlSettings):
    """The memoized version-keyed grant cache, mirroring ``PolicyEnforcer``'s policy
    cache (same size/ttl bound). Version participates in the key, so a role edit's
    version bump yields a fresh slot — a cross-worker miss that re-reads the CURRENT
    grant map once and re-caches."""
    global _grants_cache
    if _grants_cache is None:
        _grants_cache = alru_cache(maxsize=settings.cache_size, ttl=settings.cache_ttl_seconds)(_raw_resolve_grants)
    return _grants_cache


@register_settings_reset
def reset_role_grants_cache() -> None:
    """Drop the memoized grant cache so a fresh settings object (or a test) rebuilds it,
    mirroring the sibling ``@register_settings_reset`` caches."""
    global _grants_cache
    _grants_cache = None


async def resolve_role_grants(role_name: str, version: int) -> RoleGrants:
    """The role's CURRENT grant map (version-keyed cache). Raises
    ``DocumentNotFoundError`` when the role does not exist (fail-closed deny)."""
    settings = access_control_settings()
    return await _get_grants_cache(settings)(role_name, version)


async def role_level_decision_for_route(
    policy: AccessPolicy,
    owner_policy: AccessPolicy | None,
    meta: RouteMetadata,
    method: str,
    version: int,
) -> tuple[bool, DenialCause | None]:
    """The shared per-tag LEVEL decision over an ALREADY-RESOLVED registered route — the
    term intersected with the base-tier jq and the owner second-pass at the enforcement
    site.

    * A ``fenced``/``secret`` route is decided FIRST and against the CALLER's OWN policy:
      fence-exemption is a PRINCIPAL property and an owned key is never the admin
      principal, so an admin-OWNED key is still a hard-fence DENY and an owner cannot
      delegate fence access by minting a broad owned key. The fence is never grantable.
    * On a grantable route the GOVERNING role is the OWNER's for an owned key (keys
      inherit the owner's role grant map — no per-key grant), else the caller's own
      policy. An ``allow_all``/admin governing role skips the pass entirely (admin is
      everything, un-lockable).
    * Otherwise the route is grantable: the governing role's per-tag level must satisfy
      the method's derived action. A governing role with NO pointer is not level-governed
      (its reach is the jq base); a pointer naming a MISSING/deleted role is a
      fail-closed DENY.

    Returns ``(allowed, cause)``; ``cause`` names the internal denial reason on a deny.
    """
    if meta.action in ("fenced", "secret"):
        # Keyed on the CALLER's OWN admin verdict, never the owner's: fence-exemption
        # cannot be inherited through the owner channel that carries a grant.
        caller_owner = policy.policy_data.get(OWNER_USER_ID_CLAIM)
        if is_admin_policy(policy, caller_owner):
            return True, None
        return False, DenialCause.HARD_FENCE

    governing = owner_policy if owner_policy is not None else policy
    governing_owner = governing.policy_data.get(OWNER_USER_ID_CLAIM)
    if is_admin_policy(governing, governing_owner):
        return True, None

    from tai42_skeleton.access_control.roles import ROLE_POINTER_KEY

    role_name = governing.policy_data.get(ROLE_POINTER_KEY)
    if role_name is None:
        return True, None

    try:
        grants = await resolve_role_grants(role_name, version)
    except DocumentNotFoundError:
        # The pointer names a role that no longer exists — deny fail-closed (LIVE:
        # deleting a role denies its holders on their next request).
        return False, DenialCause.LEVEL_MISS
    return grant_map_admits(meta, method, grants)


async def role_level_decision(
    policy: AccessPolicy,
    owner_policy: AccessPolicy | None,
    path: str,
    method: str | None,
    version: int,
) -> tuple[bool, DenialCause | None]:
    """:func:`role_level_decision_for_route` for a caller holding a REAL request target,
    resolving the route here.

    Two shapes reach the resolution with no gated route behind them and are not acted on
    (the scope layer + jq base govern them): a method-less scope (websocket/MCP — every
    fenced/secret route is an HTTP route with concrete methods) and a path registered
    nowhere (SPA shell, probe, unmapped).

    Neither is an allow-by-omission: the path is the one the caller REQUESTED, and the
    boot audit enforces that every registered ``fenced``/``secret`` route resolves back to
    itself here or the process refuses to start. A grantable route carries no such audit.
    A path SYNTHESIZED from caller-supplied arguments has neither guarantee, so the tool
    edge pins its route and calls :func:`role_level_decision_for_route` directly.
    """
    if method is None:
        return True, None
    meta = resolve_route_meta(path, method)
    if meta is None:
        return True, None
    return await role_level_decision_for_route(policy, owner_policy, meta, method, version)
