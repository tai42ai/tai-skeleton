import logging
import time

from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, UnauthenticatedUser
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import JqAuthContext

# The auth gate renders a policy's jq condition through the live template
# manager via this interface.
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control.path_canon import MalformedPathError, canonicalize_path
from tai42_skeleton.access_control.policy import PolicyEnforcer, policy_is_empty
from tai42_skeleton.access_control.role_gate import DenialCause
from tai42_skeleton.access_control.role_grants import role_level_decision
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.user import TaiUser, is_admin_policy
from tai42_skeleton.access_control.verifier import AccessControlVerifier, is_always_public_prefix

logger = logging.getLogger(__name__)


def extract_credential_candidates(conn) -> list[str]:
    """Every presented credential candidate in the backend's priority order:
    ``Authorization`` Bearer (or a raw ``Authorization`` value with no scheme) first,
    then ``X-Api-Key``. Returns the FULL list, not the first match — the logout
    dispatcher fans out over all of them so a stale value in one header cannot hide a
    live session in the other. Any non-bearer scheme (Basic, Digest, …) is never a
    candidate."""
    candidates: list[str] = []

    auth_header = conn.headers.get("Authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if not token:
            # No scheme at all — the whole header is a raw credential.
            candidates.append(scheme)
        elif scheme.lower() == "bearer":
            token = token.strip()
            if token:
                candidates.append(token)

    api_key = conn.headers.get("X-Api-Key")
    if api_key:
        candidates.append(api_key)

    return candidates


def effective_scopes(key_scopes: list[str], owner_scopes: list[str]) -> list[str]:
    """The scopes an owned key actually carries: its own scopes ∩ the owner's CURRENT
    scopes, with ``"*"`` behaving as "everything" on BOTH sides (``"*" ∩ X = X``).

    Three explicit cases: a ``"*"`` owner caps nothing (the key keeps its scopes); a
    ``"*"`` KEY under a scoped owner collapses to the owner's scopes (a plain membership
    filter would wrongly yield ``[]`` here); otherwise a plain intersection preserving
    the key's order."""
    if "*" in owner_scopes:
        return list(key_scopes)
    if "*" in key_scopes:
        return list(owner_scopes)
    owner_set = set(owner_scopes)
    return [scope for scope in key_scopes if scope in owner_set]


class AuthorizationError(AuthenticationError):
    """An already-authenticated caller is denied access — either the policy
    condition rejected them or the policy decision could not be completed.

    It subclasses ``AuthenticationError`` so Starlette's ``AuthenticationMiddleware``
    still routes it through ``on_error``, but the distinct type lets the error
    handler render it as 403 (authenticated but forbidden) rather than 401.

    ``cause`` is the INTERNAL denial cause (a ``DenialCause``) for debugging/logging —
    it never reaches the client body (the external response stays a generic 403 that
    leaks nothing), so a fenced-route denial and a per-tag level-miss are internally
    distinguishable while the wire response is unchanged.
    """

    def __init__(self, *args: object, cause: DenialCause | None = None) -> None:
        super().__init__(*args)
        self.cause = cause


class AccessControlAuthBackend(AuthenticationBackend):
    def __init__(self, verifier: AccessControlVerifier, settings: AccessControlSettings):
        self.verifier = verifier
        self.settings = settings
        self.enforcer = PolicyEnforcer(settings)

    async def _get_access_token(self, conn):
        candidates = extract_credential_candidates(conn)

        # Case A: No credentials provided at all -> Return None (Handled as Unauthenticated by caller)
        if not candidates:
            return None

        # Case B: Credentials provided -> Try to verify them in order
        for token in candidates:
            try:
                access_token = await self.verifier.verify_token(token)
                if access_token:
                    return access_token
            except Exception:
                # Fail-closed: a verification error never grants access. Treat
                # this candidate as invalid and try the next; if every candidate
                # is exhausted the loop falls through to the raise below (deny),
                # so an error here can only ever lead to denial, never to allow.
                logger.exception("access_control: token verification errored; treating candidate as invalid")
                continue

        # Case C: Credentials were provided, but NONE were valid -> deny loudly.
        raise AuthenticationError("Invalid API key")

    def _is_always_public_path(self, path: str) -> bool:
        """Whether ``path`` is the pre-auth login surface, asked of the ONE definition of
        that family over the SAME canonical form the resource guard resolves on.

        A malformed path has no canonical form, so it is NOT this surface: it falls
        through to the credential path and is denied downstream, never admitted
        unauthenticated on a shape the guard itself refuses to reason about."""
        try:
            canonical = canonicalize_path(path)
        except MalformedPathError:
            logger.warning(
                "access_control: request path %r is malformed (NUL/control/backslash or residual "
                "percent-escape) — not admitting it as the pre-auth login surface",
                path,
            )
            return False
        return is_always_public_prefix(canonical, self.settings)

    async def authenticate(self, conn):
        # 0. Public login surface: ignore any presented credential outright. Identity is
        # never needed here, and this middleware runs BEFORE the resource guard's public
        # short-circuit — so verifying a stale ``tai-sess-``/``sk-`` token would 401 the
        # recovery path this namespace exists for. No verification, no provider I/O.
        if self._is_always_public_path(conn.url.path):
            return AuthCredentials(["unauthenticated"]), UnauthenticatedUser()

        # 1. Resolve Identity
        # This will either return a token, return None (guest), or raise AuthenticationError (bad token)
        access_token = await self._get_access_token(conn)

        if not access_token:
            return AuthCredentials(["unauthenticated"]), UnauthenticatedUser()

        user_id = access_token.client_id

        # 2 & 3. Fetch Policy (Cached) and Live Context (Fresh)
        # A backend error here (redis down, etc.) must fail closed as a clean
        # deny, not leak out as a raw 500: wrap it into AuthenticationError so the
        # AuthenticationMiddleware's on_error handler renders a 401/403.
        try:
            policy, dynamic_context = await self.enforcer.get_auth_data(user_id)

            # A credential that verifies against a principal the policy store no longer
            # knows (the residue mid-revoke). Denied here so "authenticated" can never
            # mean "has no policy" — a door gated on nothing but an authenticated
            # principal would otherwise admit a revoked key.
            if policy_is_empty(policy):
                logger.warning("access_control: denied principal %s — no policy", user_id)
                raise AuthorizationError("Access Denied")

            # Disabled principal (direct): a disabled account user's own session/key is
            # denied here — defense in depth beside the owned-key owner-disable check.
            if policy.policy_data.get("disabled") is True:
                logger.warning("access_control: denied disabled principal %s", user_id)
                raise AuthorizationError("Access Denied")

            # Owned-key attenuation: a credential whose claims carry an owner is
            # capped by the owner's CURRENT policy at REQUEST time, so the cap holds over
            # time rather than freezing at mint. A missing/empty/disabled owner denies.
            owner = access_token.claims.get(OWNER_USER_ID_CLAIM)
            owner_policy = None
            resolved_scopes = policy.scopes
            if owner is not None:
                owner_policy = await self.enforcer.get_policy(owner)
                if owner_policy.policy_data.get("disabled") is True:
                    logger.warning("access_control: denied owned key %s — owner %s is disabled", user_id, owner)
                    raise AuthorizationError("Access Denied")
                if policy_is_empty(owner_policy):
                    logger.warning("access_control: denied owned key %s — owner %s has no policy", user_id, owner)
                    raise AuthorizationError("Access Denied")
                resolved_scopes = effective_scopes(policy.scopes, owner_policy.scopes)
        except AuthorizationError:
            raise
        except Exception as e:
            # Log the underlying failure (redis host:port, etc.) server-side, but
            # deny the caller with a generic message that leaks no internal detail.
            logger.exception("access_control: policy/context fetch failed for user %s", user_id)
            raise AuthorizationError("Access Denied") from e

        # 4. Build Unified Context (with the effective, owner-attenuated scopes)
        context = JqAuthContext(
            sub=user_id,
            scopes=resolved_scopes,
            identity=access_token.claims,
            policy=policy.policy_data,
            context=dynamic_context,
            request={"method": conn.scope.get("method"), "path": conn.url.path},
            system={"time": time.time()},
        )

        # 5. Enforce Policy — the KEY's own condition, then (for an owned key) the
        # OWNER's condition as a SEPARATE enforce pass over a context built from the
        # OWNER's policy_data + scopes, so an owner condition referencing ``.policy.*``
        # reads the owner's policy, never the key's mint-time policy_data. Two sequential
        # enforce calls are semantically AND — no jq string concatenation (splicing is
        # injection-shaped).
        try:
            condition = await tai42_app.storage.resource_manager.render_by_id_or_content(
                content=policy.condition,
                template_id=policy.condition_id,
                kwargs=policy.condition_kwargs,
            )
            # Whether a condition was configured is known from the policy, not from
            # the rendered string: a configured condition that renders empty must
            # still be enforced (deny), so pass the configured flag through and let
            # ``enforce`` fail closed rather than mistaking empty for "no condition".
            condition_configured = policy.condition is not None or policy.condition_id is not None
            await self.enforcer.enforce(context.model_dump(), condition, condition_configured=condition_configured)

            if owner_policy is not None and (
                owner_policy.condition is not None or owner_policy.condition_id is not None
            ):
                owner_context = JqAuthContext(
                    sub=user_id,
                    scopes=owner_policy.scopes,
                    identity=access_token.claims,
                    policy=owner_policy.policy_data,
                    context=dynamic_context,
                    request={"method": conn.scope.get("method"), "path": conn.url.path},
                    system={"time": time.time()},
                )
                owner_condition = await tai42_app.storage.resource_manager.render_by_id_or_content(
                    content=owner_policy.condition,
                    template_id=owner_policy.condition_id,
                    kwargs=owner_policy.condition_kwargs,
                )
                await self.enforcer.enforce(owner_context.model_dump(), owner_condition, condition_configured=True)
        except Exception as e:
            # Log the underlying failure (jq internals, render errors) server-side,
            # but deny the caller with a generic message that leaks no internal detail.
            logger.exception("access_control: policy enforcement failed for user %s", user_id)
            raise AuthorizationError("Access Denied") from e

        # 5b. The per-tag LEVEL pass — Layer 2 of the (resource-x-action) model,
        # INTERSECTED with the base-tier jq above (fail-closed AND). Skipped for an admin
        # governing role; a fenced/secret route is admin-only; a grantable route needs
        # the governing role's per-tag level (the OWNER's role for an owned key — keys
        # inherit the owner). A resolution/infra fault fails closed as a clean deny.
        method = conn.scope.get("method")
        path = conn.url.path
        try:
            version = await self.enforcer.current_policy_version()
            allowed, cause = await role_level_decision(policy, owner_policy, path, method, version)
        except Exception as e:
            logger.exception("access_control: per-tag level resolution failed for user %s", user_id)
            raise AuthorizationError("Access Denied") from e
        if not allowed:
            logger.warning(
                "access_control: per-tag level denied user %s on %s %s (%s)",
                user_id,
                method,
                path,
                cause.value if cause else "deny",
            )
            raise AuthorizationError("Access Denied", cause=cause)

        # 6. Finalize with the effective scopes, stamping the admin discriminator so the
        # resource guard can admit a super-admin to a not-yet-configured route. The owner
        # is read from the STORED ``policy.policy_data`` (the management dual-home), NOT
        # the request token claim — the contract ``is_admin_policy`` and its other callers
        # (the projection, key management) all share, so the guard's admin verdict is
        # byte-identical to theirs and an owned condition-free ``["*"]`` key fails CLOSED.
        access_token.scopes = resolved_scopes
        stored_owner = policy.policy_data.get(OWNER_USER_ID_CLAIM)
        return AuthCredentials(scopes=resolved_scopes), TaiUser(
            access_token, is_admin=is_admin_policy(policy, stored_owner)
        )
