from collections.abc import Mapping
from typing import Any

from fastmcp.server.auth import AccessToken
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from tai42_contract.access_control import OWNER_USER_ID_CLAIM, get_current_user_id
from tai42_contract.access_control.models import AccessPolicy

from tai42_skeleton.access_control.request_scopes import get_request_identity_claims


def is_admin_policy(policy: AccessPolicy, owner_claim: str | None) -> bool:
    """Whether ``policy`` is the ADMIN discriminator: a condition-free ``"*"`` policy
    that is not itself an owned key — the single spelling of "admin" every consumer
    shares (the key-management ownership rules and the capability projection).

    Admin iff the policy grants ``"*"`` with NO jq condition (inline or stored) AND
    carries no owner claim. Role-holders carry ``["*"]`` scopes plus a jq condition, so
    a scopes-only test would classify every editor/viewer as admin; a condition-bearing
    caller is never admin; and the owner-claim conjunct denies admin to an owned key (an
    editor-minted condition-free ``["*"]`` key would otherwise read as admin from its raw
    stored policy — a you-plus escalation). ``owner_claim`` is the owner drawn from the
    caller's STORED ``policy.policy_data`` (the management dual-home), NEVER a request
    claim, so the classification is byte-identical wherever it is used."""
    return "*" in policy.scopes and policy.condition is None and policy.condition_id is None and owner_claim is None


class TaiUser(AuthenticatedUser):
    """The authenticated principal placed in the request scope on a fully
    successful authenticate + policy pass.

    Subclassing the mcp-SDK ``AuthenticatedUser`` is what makes the SDK's
    bearer-auth route gate (``RequireAuthMiddleware``, which admits a request only
    when ``isinstance(scope["user"], AuthenticatedUser)``) and its
    ``AuthContextMiddleware`` (which powers ``get_access_token()`` inside tools)
    recognize this authenticated principal on the main ``/mcp`` and ``/sse``
    endpoints and admit the request.

    ``fastmcp``'s ``AccessToken`` subclasses the mcp-SDK ``AccessToken`` that
    ``AuthenticatedUser.__init__`` expects, so the token the backend already holds
    is passed straight through. ``.token`` is retained for
    ``ResourceGuardMiddleware`` (which reads ``user.token.client_id``)."""

    def __init__(self, token: AccessToken, is_admin: bool = False):
        super().__init__(token)
        self.token = token
        # Whether this principal is the ADMIN discriminator (a condition-free ``"*"``
        # policy that is not an owned key — see :func:`is_admin_policy`). Computed
        # server-side by the auth backend from the resolved policy and stamped here, so
        # it cannot be forged by a provider claim. The resource guard reads it to admit a
        # super-admin to a route with no configured row: a root identity is never gated
        # by a missing route mapping (it can map the route anyway), so blocking it is a
        # footgun, not security — while every non-admin identity still fails closed.
        self.is_admin = is_admin

    @property
    def identity(self) -> str:
        return self.token.client_id


def _acting_principal() -> tuple[str | None, Mapping[str, Any] | None]:
    """``(own id, claims)`` of the principal ACTING at the current dispatch.

    A bound execution identity takes PRECEDENCE, never a fallback: a fire dispatched as a
    Starlette ``BackgroundTask`` runs inside the triggering request's contextvar context,
    so the request-scope vars are still that caller's and must not be consulted while a
    fire is bound. Outside a fire the pair is the request-scope caller; both halves are
    ``None`` when none is bound.

    The same rule :func:`~tai42_skeleton.operations._authority.resolve_caller` applies, so
    isolation and the pass-role gate never key on different principals.

    Imported at call time: ``authz`` reaches this module back through
    ``access_control.backend``."""
    from tai42_skeleton.authz.execution_identity import get_execution_identity

    identity = get_execution_identity()
    if identity is not None:
        return identity.user_id, identity.claims
    return get_current_user_id(), get_request_identity_claims()


def restricted_identity() -> str | None:
    """The identity a RESTRICTED caller is isolated to — its OWN id — or ``None`` when
    the caller is unrestricted (admin, editor/viewer role-holder, ownerless machine
    key) and for the unauthenticated / gate-off cases where no caller is bound.

    A caller is restricted iff its claims carry ``OWNER_USER_ID_CLAIM`` — an owned key
    acting on behalf of its owner. Being owned is what CONFINES the caller, but the
    identity it is confined to is its OWN id (its ``user_id`` / token ``client_id``),
    NOT its owner's: each owned key is its own island. A restricted caller sees and
    touches ONLY the tool runs, interactions, and notifications belonging to (addressed
    to) its OWN key identity — never its owner's, never a sibling owned key's. This
    helper is the one definition of "restricted" the whole codebase shares.

    Both the deciding claims and the confining id come from :func:`_acting_principal`, so
    a fire is isolated to the key it is authorized as rather than to whoever triggered it,
    and no Starlette ``Request`` is needed — the flat-argument operation doors can enforce
    isolation without one. With the gate off no claims are bound, so the result is
    ``None``."""
    own, claims = _acting_principal()
    if claims is None or claims.get(OWNER_USER_ID_CLAIM) is None:
        return None
    if own is None:
        # An owner-claim-bearing principal always has a bound own id. A miss is a broken
        # invariant, not a state to isolate to nothing: confining to None would open the
        # full view to a restricted caller.
        raise RuntimeError("owner-claim-bearing caller has no bound own id; identity invariant broken")
    return own


class CrossIdentityAudienceError(Exception):
    """A RESTRICTED caller tried to address an ``audience`` other than its own
    identity — the cross-identity inject/exfil attempt :func:`clamp_write_audience`
    rejects.

    It is an AUTHORIZATION denial, NOT input validation: a write door (``notify_user``)
    maps it to the same ``403``/``ForbiddenError`` the read-side answer door raises for
    the symmetric cross-identity read denial, so both boundary violations surface as
    403 — distinct from the blank-audience ``ValueError`` a door validates as a 400.
    Kept as an access-control domain exception (not the operations-layer
    ``ForbiddenError``) so this foundational module stays free of an upward operations
    dependency; the door owns the mapping."""


def clamp_write_audience(audience: str | None) -> str | None:
    """The WRITE-side dual of the isolation read clamps: scope the ``audience`` a
    write door (``ask_user`` / ``notify_user``) may address to the caller's own slice.

    A RESTRICTED caller (:func:`restricted_identity` returns a non-None id — an owned
    key confined to its OWN slice) may address ONLY its own identity, so its writes
    land exclusively in its own isolation slice — the write-side guarantee the read
    clamps assume. ``audience is None`` is scoped to SELF (the owned key addresses its
    own slice), ``audience == own id`` passes unchanged, and ANY OTHER identity is a
    loud :class:`CrossIdentityAudienceError` (a cross-identity inject/exfil attempt
    through another identity's slice) — an AUTHORIZATION denial the write doors map to
    a ``403``, mirroring the read-side answer door, NOT the blank-audience
    ``ValueError``/400. An UNRESTRICTED caller (admin / system / ownerless execution
    key / no bound principal at all) is returned unchanged — it may address any
    identity, or broadcast with ``audience is None``.

    Returns the audience the door must persist. A door runs its own blank-audience
    validation first; this clamp is in addition to it."""
    own = restricted_identity()
    if own is None:
        return audience
    if audience is None:
        return own
    if audience != own:
        raise CrossIdentityAudienceError("a restricted caller may address only its own identity")
    return audience


def request_identity() -> tuple[str | None, str | None]:
    """``(user_id, restricted)`` for the current caller, resolved once so a door never
    re-derives the pair.

    ``user_id`` is the acting principal's own id (:func:`_acting_principal`), so a fire's
    writes are attributed to the KEY rather than to whoever triggered it. The second
    element is the ISOLATION identity (:func:`restricted_identity`) — the caller's OWN id
    when restricted, else ``None`` (unrestricted → full view). ``restricted is not None``
    is the restricted test. Both are ``None`` when no principal is bound; a gate-off FIRE
    still binds its key, so only the isolation half is ``None`` there. When restricted,
    the two are the SAME id."""
    return _acting_principal()[0], restricted_identity()
