import logging

from starlette.authentication import AuthCredentials, UnauthenticatedUser
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id

from tai42_skeleton.access_control.request_scopes import (
    reset_request_effective_scopes,
    reset_request_identity_claims,
    set_request_effective_scopes,
    set_request_identity_claims,
)
from tai42_skeleton.access_control.verifier import AccessControlVerifier

logger = logging.getLogger(__name__)

# Websocket close code for a denied connection (RFC 6455 "policy violation").
_WS_POLICY_VIOLATION = 1008

# Appended to every server-side deny log line (never to the client response):
# access control is on by default, so a denied local run needs to learn about
# the kill switch from the server log, not from the (deliberately generic) body.
_DISABLE_HINT = "set ACCESS_CONTROL_ENABLE=false to disable access control for local development"


class ResourceGuardMiddleware:
    """The route-authorization guard, resolving three prefix/path families:

    * RESERVED (``reserved_public_pin_prefixes``) — never public. Resolved in the
      verifier BEFORE this middleware: it drops the public marker for a reserved-prefix
      path, so the control plane stays authenticated regardless of the route table.
      Reserved affects only public-pin writes and that resolution drop.
    * ALWAYS-PUBLIC (``always_public_path_prefixes``) — public regardless of the table.
      Also resolved in the verifier before this middleware sees a decision (it answers
      the public resource id unconditionally), surfacing here as CASE B.
    * AUTHENTICATED-ALWAYS-ALLOWED (``authenticated_always_allowed_paths``) — allowed for
      ANY authenticated identity regardless of the table. This is the one family THIS
      middleware decides: an exact-path member is checked BEFORE table resolution, so it
      is reachable even with no route row; an unauthenticated caller is denied 401. The
      backend's jq enforcement already ran upstream, so a role condition still gates it.
    """

    def __init__(
        self,
        app: ASGIApp,
        verifier: AccessControlVerifier,
        public_resource_id: str,
        authenticated_always_allowed_paths: tuple[str, ...] = (),
    ):
        self.app = app
        self.verifier = verifier
        self.public_id = public_resource_id
        # Stored as a frozenset for O(1) exact-path membership on every request.
        self.authenticated_always_allowed_paths = frozenset(authenticated_always_allowed_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # 1. Skip non-HTTP scopes (lifespan, etc.); http + websocket are guarded.
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # 2. Resolve Route -> Resource IDs
        # We build an HTTPConnection (works for both http and websocket scopes)
        # purely for URL/path helpers; we don't pass it to the app.
        conn = HTTPConnection(scope)
        path_to_check = conn.url.path

        # Prepare User/Auth info from Scope (Populated by AuthenticationMiddleware).
        user = scope.get("user", UnauthenticatedUser())
        auth = scope.get("auth", AuthCredentials())

        # AUTHENTICATED-ALWAYS-ALLOWED carve-out: an EXACT-path member is reachable by
        # ANY authenticated identity regardless of the route table. Checked BEFORE table
        # resolution (CASE A below) so the path is reachable even with no route row; the
        # backend's jq enforcement already ran upstream, so a role condition still gates
        # it. An unauthenticated caller is denied 401 exactly like a protected route.
        # The membership test uses the RAW request path (no trailing-slash normalization)
        # so it admits exactly the path shape the companion role jq fence admits — an
        # exact-match string — and a ``/api/auth/me/`` variant is denied on both sides
        # rather than admitted here but 403'd by the fence.
        if path_to_check in self.authenticated_always_allowed_paths:
            if not user.is_authenticated:
                logger.warning(
                    "access_control: denied %s — unauthenticated request to an authenticated-always-allowed route; %s",
                    path_to_check,
                    _DISABLE_HINT,
                )
                await self._deny(scope, receive, send, 401, "Authentication required")
                return
            await self._run_app_with_context(scope, receive, send, user)
            return

        # A resolve failure (e.g. a backend outage in the verifier's fetchers)
        # must fail closed as a clean deny, not leak out as a raw 500. The verifier
        # raises rather than caching a degraded result; we log it here and deny.
        try:
            resource_ids = await self.verifier.resolve_resource_ids(path_to_check)
        except Exception:
            logger.exception(
                "access_control: route resolution failed for %s — denying; %s",
                path_to_check,
                _DISABLE_HINT,
            )
            await self._deny(scope, receive, send, 403, "Forbidden: Route not configured")
            return

        # CASE A: Unknown Route (403)
        if not resource_ids:
            logger.warning(
                "access_control: denied %s — no resource is configured for this route; %s",
                path_to_check,
                _DISABLE_HINT,
            )
            await self._deny(scope, receive, send, 403, "Forbidden: Route not configured")
            return

        # CASE B: Public Route — deny wins: a path is public only when public
        # is the ONLY resolved resource id. If it also matched a protected
        # route, an over-broad public pattern must not open that route.
        is_public = set(resource_ids) == {self.public_id}

        if is_public:
            # Public route logic: Allow, but set context if user exists
            await self._run_app_with_context(scope, receive, send, user)
            return

        # CASE C: Protected Route - Check Auth
        if not user.is_authenticated:
            logger.warning(
                "access_control: denied %s — unauthenticated request to a protected route; %s",
                path_to_check,
                _DISABLE_HINT,
            )
            await self._deny(scope, receive, send, 401, "Authentication required")
            return

        # CASE D: Check Permissions — deny wins, mirroring the public decision
        # above. The caller must be authorized for EVERY protected resource this
        # path resolved to (the public id, if also matched, carries no scope
        # requirement); a wildcard scope satisfies all. Requiring all — not any —
        # stops a broad tier's scope from opening a route that a more-specific
        # tier restricted to a stronger scope.
        user_scopes = auth.scopes
        protected_ids = [rid for rid in resource_ids if rid != self.public_id]
        has_permission = "*" in user_scopes or all(rid in user_scopes for rid in protected_ids)

        if not has_permission:
            # Log which scopes were required server-side, but never echo the
            # resource names back to the caller — that discloses the protected
            # resource layout to an authenticated-but-unauthorized user.
            logger.warning(
                "access_control: user %s denied; required all of %s for %s; %s",
                user.token.client_id,
                protected_ids,
                path_to_check,
                _DISABLE_HINT,
            )
            await self._deny(scope, receive, send, 403, "Forbidden")
            return

        # Success: Run App with Context
        await self._run_app_with_context(scope, receive, send, user)

    @staticmethod
    async def _deny(scope: Scope, receive: Receive, send: Send, status_code: int, message: str):
        """Emit a deny shaped for the scope type.

        A websocket scope cannot receive an ``http.response.start`` (it would be a
        malformed close); it must be closed with a websocket close frame instead.
        http scopes get the JSON error body.
        """
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
            return
        response = JSONResponse({"error": message}, status_code=status_code)
        await response(scope, receive, send)

    async def _run_app_with_context(self, scope, receive, send, user):
        """
        Runs the app within the context variable scope.
        Pure ASGI implementation avoids 'call_next' overhead.
        """
        context_token = None
        scopes_token = None
        claims_token = None
        if user.is_authenticated:
            user_id = user.token.client_id
            context_token = set_request_user_id(user_id)
            # Carry the auth backend's already-decided effective scopes (owner-attenuated
            # for an owned key) so the tool-edge authorization consumes the SAME scope set
            # this edge enforces, never an owned key's unattenuated policy scopes. Bound
            # as a set with the caller id, so a bound caller always carries them.
            scopes_token = set_request_effective_scopes(tuple(user.token.scopes))
            # Carry the caller's verified token claims too, so the tool-edge check reads
            # the SAME identity the backend does: the ``.identity.*`` a policy condition
            # references, and the owner reference that drives the owner second-pass enforce.
            claims_token = set_request_identity_claims(user.token.claims)

        try:
            await self.app(scope, receive, send)
        finally:
            if context_token:
                reset_request_user_id(context_token)
            if scopes_token is not None:
                reset_request_effective_scopes(scopes_token)
            if claims_token is not None:
                reset_request_identity_claims(claims_token)
