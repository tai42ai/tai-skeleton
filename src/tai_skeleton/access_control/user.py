from fastmcp.server.auth import AccessToken
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from tai_contract.access_control import OWNER_USER_ID_CLAIM, get_current_user_id
from tai_contract.access_control.models import AccessPolicy

from tai_skeleton.access_control.request_scopes import get_request_identity_claims


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

    def __init__(self, token: AccessToken):
        super().__init__(token)
        self.token = token

    @property
    def identity(self) -> str:
        return self.token.client_id


def restricted_identity() -> str | None:
    """The identity a RESTRICTED caller is isolated to — its OWN id — or ``None`` when
    the caller is unrestricted (admin, editor/viewer role-holder, ownerless machine
    key) and for the unauthenticated / gate-off cases where no caller is bound.

    A caller is restricted iff its verified token claims carry ``OWNER_USER_ID_CLAIM``
    — an owned key acting on behalf of its owner. Being owned is what CONFINES the
    caller, but the identity it is confined to is its OWN id (its ``user_id`` /
    token ``client_id``), NOT its owner's: each owned key is its own island. A
    restricted caller sees and touches ONLY the tool runs, interactions, and
    notifications belonging to (addressed to) its OWN key identity — never its
    owner's, never a sibling owned key's. This helper is the one definition of
    "restricted" the whole codebase shares.

    It reads the request-scoped claims the access-control guard bound
    (:func:`~tai_skeleton.access_control.request_scopes.get_request_identity_claims`)
    to DECIDE restriction, and the bound caller id (:func:`get_current_user_id`) as
    the confinement VALUE — NOT a Starlette ``Request`` — so the flat-argument
    operation doors can enforce isolation without a request object. With the gate off
    no claims are bound, so the result is ``None`` — there is no identity to restrict."""
    claims = get_request_identity_claims()
    if claims is None or claims.get(OWNER_USER_ID_CLAIM) is None:
        return None
    own = get_current_user_id()
    if own is None:
        # A caller whose claims carry an owner is always an authenticated caller, so
        # its own id is always bound. A missing own id here is a broken invariant, not
        # a state to isolate to nothing — raise loudly rather than silently confine to
        # None (which would open the full view to a restricted caller).
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
    ``ValueError``/400. An UNRESTRICTED caller (admin / system / no bound request
    identity) is returned unchanged — it may address any identity, or broadcast with
    ``audience is None``.

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

    ``user_id`` is the authenticated caller's own id (:func:`get_current_user_id`);
    the second element is the ISOLATION identity (:func:`restricted_identity`) — the
    caller's OWN id when it is restricted (an owned key confined to its own slice),
    else ``None`` (unrestricted → full view). ``restricted is not None`` is the
    restricted test — the isolated surfaces stamp and read a restricted caller's slice
    under it, and a run/notification submitted by an unrestricted caller is owned by
    ``user_id``. Both are ``None`` when no caller is bound (an anonymous request, or
    the gate off); when restricted, the two are the SAME id."""
    return get_current_user_id(), restricted_identity()
