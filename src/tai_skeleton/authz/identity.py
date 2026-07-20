"""The caller-identity model for the tool-edge authorization check.

The identity is RESOLVED AT EXECUTION from the request-user context the access
control middleware binds (:func:`get_current_user_id`). The tool-edge
``AuthzMiddleware`` fires only on the MCP-serving transport — an EXTERNAL surface
— so a call reaching it with no resolvable identity (while access control is
enabled) is denied fail-closed. Internal dispatch (agents resolving tools,
schedulers, backend workers) never passes through that middleware; the internal
principal exists for the rare direct programmatic check that must be allowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tai_contract.access_control.context import get_current_user_id

from tai_skeleton.access_control.request_scopes import (
    get_request_effective_scopes,
    get_request_identity_claims,
)


@dataclass(frozen=True)
class CallerIdentity:
    """Who is dispatching an operation.

    ``user_id`` is the authenticated caller (``None`` = no resolvable identity).
    ``is_internal`` marks the platform-internal principal, which is always
    allowed (behavior preservation for agents/schedulers/workers).

    ``effective_scopes`` is the HTTP auth backend's already-decided scope set for
    this caller (owner-attenuated for an owned/delegated key), carried from the
    guard middleware so the tool-edge check enforces the SAME scopes the HTTP edge
    did rather than re-deriving an owned key's unattenuated policy scopes. ``None``
    means no attenuation decision was carried — the check then falls back to the
    caller's own policy scopes, which for a non-owned key ARE the effective scopes.
    An authenticated request always carries it (the guard binds it with the id).

    ``claims`` is the caller's verified token claims, carried from the same guard
    binding, so the tool-edge check builds its jq context with the SAME
    ``.identity.*`` the HTTP backend uses AND reads the owner reference (under
    ``OWNER_USER_ID_CLAIM``) that drives the owner second-pass enforce. ``None``
    means no caller was bound — the check then reads an empty identity, matching a
    request that carried no claims.
    """

    user_id: str | None = None
    is_internal: bool = False
    effective_scopes: tuple[str, ...] | None = None
    claims: Mapping[str, Any] | None = None


# The platform-internal principal — always allowed.
INTERNAL_PRINCIPAL = CallerIdentity(is_internal=True)


def resolve_caller_identity() -> CallerIdentity:
    """The external caller's identity at the current dispatch, from the request-user
    context and the effective-scopes context the access control middleware bound.

    ``user_id`` is ``None`` when no caller is bound — an unauthenticated external
    dispatch, which the authorization check denies while access control is enabled.
    ``effective_scopes`` is the owner-attenuated scope set the HTTP edge decided and
    ``claims`` the caller's verified token claims, both bound as a set with the
    caller id on every authenticated request.
    """
    return CallerIdentity(
        user_id=get_current_user_id(),
        effective_scopes=get_request_effective_scopes(),
        claims=get_request_identity_claims(),
    )
