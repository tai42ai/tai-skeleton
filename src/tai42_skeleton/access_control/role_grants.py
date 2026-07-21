"""The LIVE role→grant-map resolution + the shared per-tag enforcement decision.

Editing a role affects EVERY holder LIVE: a user's enforced policy carries a separate
role-name POINTER (``policy_data[ROLE_POINTER_KEY]``, never ``condition_id`` — that
would collide with the admin discriminator), read here to fetch the role's CURRENT
grant map. The lookup rides a VERSION-KEYED cache busted by ``bump_policy_version`` on
any role edit, so a live edit lands on the next request with no uncached hot-path store
read.

:func:`role_level_decision` is the ONE shared per-tag decision the request gate
(:mod:`~tai42_skeleton.access_control.backend`) AND the capability projection
(:mod:`~tai42_skeleton.access_control.projection`) both consume — never two divergent
copies — so the projection can never advertise a door the gate would deny.
"""

from __future__ import annotations

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


async def role_level_decision(
    policy: AccessPolicy,
    owner_policy: AccessPolicy | None,
    path: str,
    method: str | None,
    version: int,
) -> tuple[bool, DenialCause | None]:
    """The shared per-tag LEVEL decision for one request — the term intersected with the
    base-tier jq and the owner second-pass at the enforcement site.

    * The GOVERNING role is the OWNER's for an owned key (keys inherit the owner's role
      grant map — no per-key grant), else the caller's own policy.
    * An ``allow_all``/admin governing role skips the pass entirely (admin is
      everything, un-lockable).
    * A path that resolves to no registered gated route is not acted on (the scope layer
      + jq base govern the SPA shell / operational / unmapped paths).
    * A ``fenced``/``secret`` route is a hard-fence DENY for every non-admin, regardless
      of any level (never grantable).
    * Otherwise the route is grantable: the governing role's per-tag level must satisfy
      the method's derived action. A governing role with NO pointer is not level-governed
      (its reach is the jq base) — the fence above still applies; a pointer naming a
      MISSING/deleted role is a fail-closed DENY.

    Returns ``(allowed, cause)``; ``cause`` names the internal denial reason on a deny.
    """
    governing = owner_policy if owner_policy is not None else policy
    governing_owner = governing.policy_data.get(OWNER_USER_ID_CLAIM)
    if is_admin_policy(governing, governing_owner):
        return True, None

    if method is None:
        # A missing method means this is NOT an HTTP-route request — a websocket/MCP scope
        # legitimately carries no method. The per-tag fence governs HTTP routes only, and
        # every fenced/secret route IS an HTTP route with concrete methods, so it cannot
        # apply to a method-less scope; the scope layer + jq base govern here. This is a
        # deliberate non-HTTP branch, not an allow-by-omission: a real registered route
        # always arrives WITH its method, so no fence for a registered route is ever
        # skipped by this path.
        return True, None

    meta = resolve_route_meta(path, method)
    if meta is None:
        return True, None

    if meta.action in ("fenced", "secret"):
        return False, DenialCause.HARD_FENCE

    from tai42_skeleton.access_control.roles import ROLE_POINTER_KEY

    role_name = governing.policy_data.get(ROLE_POINTER_KEY)
    if role_name is None:
        return True, None

    assert method is not None  # resolve_route_meta returned a route, so the method matched
    try:
        grants = await resolve_role_grants(role_name, version)
    except DocumentNotFoundError:
        # The pointer names a role that no longer exists — deny fail-closed (LIVE:
        # deleting a role denies its holders on their next request).
        return False, DenialCause.LEVEL_MISS
    return grant_map_admits(meta, method, grants)
