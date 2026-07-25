"""Request-scoped caller facts the HTTP edge already decided, for the tool edge.

The HTTP auth backend is the single decision authority for a caller. For an
owned/delegated key it computes the owner-attenuated ``effective_scopes`` (the
key's own scopes ∩ the owner's CURRENT scopes) and stamps them onto the access
token, and it holds the caller's verified token CLAIMS (the ``.identity.*`` a jq
condition reads, and the owner reference under ``OWNER_USER_ID_CLAIM`` that drives
the owner second-pass enforce).
:class:`~tai42_skeleton.access_control.middleware.ResourceGuardMiddleware` binds
those already-decided facts here — alongside the caller id
(``set_request_user_id``) and in the SAME place — so the tool-edge authorization
(:func:`tai42_skeleton.authz.check.check`) reaches the identical decision the HTTP
edge did, consuming its results rather than re-deriving them (which would let the
MCP surface out-permit the HTTP surface).

Each value is ``None`` when no caller is bound — an anonymous request, or code
running outside a bound request. An authenticated request always binds the caller
id, the effective scopes, and the identity claims TOGETHER (the guard binds them
as a set), so a bound caller id always carries its bound facts; the ``None`` case
is the no-caller / internal-principal / direct-construction path, never a real
external tool dispatch with a resolvable identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any

__all__ = [
    "get_request_effective_scopes",
    "get_request_identity_claims",
    "reset_request_effective_scopes",
    "reset_request_identity_claims",
    "set_request_effective_scopes",
    "set_request_identity_claims",
]

_current_effective_scopes: ContextVar[tuple[str, ...] | None] = ContextVar(
    "tai42_current_effective_scopes", default=None
)
_current_identity_claims: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "tai42_current_identity_claims", default=None
)


def get_request_effective_scopes() -> tuple[str, ...] | None:
    """The current caller's owner-attenuated effective scopes, or ``None`` when no
    caller is bound (an anonymous request, or code outside a bound request)."""
    return _current_effective_scopes.get()


def set_request_effective_scopes(scopes: tuple[str, ...] | None) -> Token[tuple[str, ...] | None]:
    """Bind ``scopes`` as the current caller's effective scopes and return the reset
    token. The guard middleware calls this once per authenticated request, paired
    with :func:`~tai42_contract.access_control.context.set_request_user_id`; pass the
    returned token to :func:`reset_request_effective_scopes` to restore the previous
    value."""
    return _current_effective_scopes.set(scopes)


def reset_request_effective_scopes(token: Token[tuple[str, ...] | None]) -> None:
    """Restore the effective scopes to the value captured in ``token`` by the
    matching :func:`set_request_effective_scopes` call."""
    _current_effective_scopes.reset(token)


def get_request_identity_claims() -> Mapping[str, Any] | None:
    """The current caller's verified token claims (the ``.identity.*`` a policy
    condition reads, and the owner reference the owner second-pass enforce needs),
    or ``None`` when no caller is bound."""
    return _current_identity_claims.get()


def set_request_identity_claims(claims: Mapping[str, Any] | None) -> Token[Mapping[str, Any] | None]:
    """Bind ``claims`` as the current caller's verified token claims and return the
    reset token. The guard middleware calls this once per authenticated request,
    paired with :func:`~tai42_contract.access_control.context.set_request_user_id`;
    pass the returned token to :func:`reset_request_identity_claims` to restore the
    previous value."""
    return _current_identity_claims.set(claims)


def reset_request_identity_claims(token: Token[Mapping[str, Any] | None]) -> None:
    """Restore the identity claims to the value captured in ``token`` by the
    matching :func:`set_request_identity_claims` call."""
    _current_identity_claims.reset(token)
