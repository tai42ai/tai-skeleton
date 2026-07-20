"""The single tool-edge authorization entry point.

``check`` consumes the SAME ``access_control/`` primitives the HTTP middleware
uses — the route→resource verifier and the policy/jq-fence enforcer — but keys
them on the concrete path SYNTHESIZED from the operation's route template + the
call's path arguments. The fences stay ``{"method", "path"}``-shaped, so the tool
edge and the route edge reach the same decision for the same operation by
construction. It raises :class:`PermissionDenied` on a denial and returns on an
allow; it never grants on an error (fail-closed).
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from starlette.authentication import AuthenticationError
from tai_contract.access_control import OWNER_USER_ID_CLAIM
from tai_contract.access_control.models import JqAuthContext
from tai_contract.app import tai_app

from tai_skeleton.access_control.policy import PolicyEnforcer
from tai_skeleton.access_control.settings import access_control_settings
from tai_skeleton.access_control.verifier import AccessControlVerifier
from tai_skeleton.authz.identity import CallerIdentity
from tai_skeleton.operations.errors import PermissionDenied

if TYPE_CHECKING:
    from tai_skeleton.access_control.settings import AccessControlSettings
    from tai_skeleton.operations.registry import OperationMetadata

logger = logging.getLogger(__name__)

_PATH_PARAM = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


def synthesize_path(op: OperationMetadata, call_arguments: dict[str, object]) -> str:
    """The concrete resource path for ``op``, substituting the call's path args
    into the route template — exactly the shape the HTTP edge's decoded
    ``scope["path"]`` carries (ASGI paths are already percent-decoded, so the raw
    argument value is substituted, path-converter segments included)."""
    if op.route_template is None:
        raise ValueError(f"operation {op.name!r} has no route template; it was never registered as a route")

    def _sub(match: re.Match[str]) -> str:
        param = match.group(1)
        if param not in call_arguments:
            raise PermissionDenied(f"access denied: missing path argument {param!r} for {op.name!r}")
        return str(call_arguments[param])

    return _PATH_PARAM.sub(_sub, op.route_template)


async def check(
    caller_identity: CallerIdentity,
    operation_metadata: OperationMetadata,
    call_arguments: dict[str, object],
    *,
    settings: AccessControlSettings | None = None,
) -> None:
    """Authorize ``caller_identity`` to dispatch ``operation_metadata``.

    Returns on allow; raises :class:`PermissionDenied` on deny. With access
    control disabled everything is allowed (matching the HTTP edge, where no
    middleware runs). The internal principal is allowed; an external caller with
    no resolvable identity is denied fail-closed.
    """
    ac_settings = settings if settings is not None else access_control_settings()
    if not ac_settings.enable:
        return
    if caller_identity.is_internal:
        return
    user_id = caller_identity.user_id
    if user_id is None:
        raise PermissionDenied("access denied: no caller identity for an external tool dispatch")

    path = synthesize_path(operation_metadata, call_arguments)
    method = operation_metadata.http_method or "POST"

    # 1. Route -> resource ids (ResourceGuardMiddleware's resolution primitive).
    verifier = AccessControlVerifier(ac_settings)
    try:
        resource_ids = await verifier.resolve_resource_ids(path)
    except Exception as exc:
        logger.warning("authz: route resolution failed for %s — denying", path, exc_info=True)
        raise PermissionDenied("access denied") from exc

    if not resource_ids:
        raise PermissionDenied(f"access denied: no resource configured for {method} {path}")

    public = ac_settings.public_resource_id
    # A path public only when public is the ONLY resolved id (deny wins, exactly
    # as the middleware decides).
    if set(resource_ids) == {public}:
        return

    # 2. Policy + live context + scopes (the enforcer primitive).
    enforcer = PolicyEnforcer(ac_settings)
    try:
        policy, context = await enforcer.get_auth_data(user_id)
    except Exception as exc:
        logger.warning("authz: policy/context fetch failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc

    if policy.policy_data.get("disabled") is True:
        raise PermissionDenied("access denied: principal is disabled")

    # The caller's verified token claims — the SAME identity the HTTP backend reads
    # (backend.py: ``identity=access_token.claims``). Present for a real external
    # dispatch (the guard binds it with the id); an empty identity only on the
    # internal/direct-construction path, matching a request that carried no claims.
    claims: dict[str, Any] = dict(caller_identity.claims) if caller_identity.claims is not None else {}

    # Owned/delegated key: the owner reference is read from the SAME claim the HTTP
    # backend reads (backend.py: ``access_token.claims.get(OWNER_USER_ID_CLAIM)``),
    # so the tool edge runs the owner second-pass enforce for exactly the keys the
    # HTTP edge does. Fetch the owner's CURRENT policy (the enforcer primitive) and,
    # as the backend does, deny a disabled or policy-less owner fail-closed.
    owner = claims.get(OWNER_USER_ID_CLAIM)
    owner_policy = None
    if owner is not None:
        try:
            owner_policy = await enforcer.get_policy(owner)
        except Exception as exc:
            logger.warning("authz: owner policy fetch failed for %s — denying", owner, exc_info=True)
            raise PermissionDenied("access denied") from exc
        if owner_policy.policy_data.get("disabled") is True:
            raise PermissionDenied("access denied: owner is disabled")
        owner_empty = not owner_policy.scopes and owner_policy.condition is None and owner_policy.condition_id is None
        if owner_empty:
            raise PermissionDenied("access denied: owner has no policy")

    # The scope set is the HTTP auth backend's already-decided EFFECTIVE scopes when
    # carried (owner-attenuated for an owned/delegated key: key scopes ∩ owner scopes),
    # so the tool edge enforces exactly what the HTTP edge did. The single decision
    # authority stays the auth backend — this consumes its result, never re-derives the
    # attenuation. It falls back to the caller's own policy scopes only when no attenuation
    # decision was carried (the internal/direct-construction path), where for a non-owned
    # key the policy scopes ARE the effective scopes.
    scopes = list(caller_identity.effective_scopes) if caller_identity.effective_scopes is not None else policy.scopes
    protected_ids = [rid for rid in resource_ids if rid != public]
    has_permission = "*" in scopes or all(rid in scopes for rid in protected_ids)
    if not has_permission:
        raise PermissionDenied("access denied: insufficient scope")

    # 3. The jq policy fences, over the SYNTHESIZED path (the same enforce primitive
    # the backend runs, keyed on {"method", "path"}): the KEY's own condition, then —
    # for an owned key — the OWNER's condition as a SEPARATE enforce pass over a
    # context built from the OWNER's policy_data + scopes (so an owner condition
    # referencing ``.policy.*`` reads the owner's policy), the KEY's live context, and
    # the KEY's claims. Two sequential enforce calls are semantically AND — no jq
    # string concatenation. This mirrors backend.py's key-then-owner enforce exactly.
    jq_context = JqAuthContext(
        sub=user_id,
        scopes=scopes,
        identity=claims,
        policy=policy.policy_data,
        context=context,
        request={"method": method, "path": path},
        system={"time": time.time()},
    )
    condition_configured = policy.condition is not None or policy.condition_id is not None
    try:
        condition = await tai_app.storage.resource_manager.render_by_id_or_content(
            content=policy.condition,
            template_id=policy.condition_id,
            kwargs=policy.condition_kwargs,
        )
        await enforcer.enforce(jq_context.model_dump(), condition, condition_configured=condition_configured)

        if owner_policy is not None and (owner_policy.condition is not None or owner_policy.condition_id is not None):
            owner_context = JqAuthContext(
                sub=user_id,
                scopes=owner_policy.scopes,
                identity=claims,
                policy=owner_policy.policy_data,
                context=context,
                request={"method": method, "path": path},
                system={"time": time.time()},
            )
            owner_condition = await tai_app.storage.resource_manager.render_by_id_or_content(
                content=owner_policy.condition,
                template_id=owner_policy.condition_id,
                kwargs=owner_policy.condition_kwargs,
            )
            await enforcer.enforce(owner_context.model_dump(), owner_condition, condition_configured=True)
    except AuthenticationError as exc:
        raise PermissionDenied("access denied: policy condition rejected") from exc
    except Exception as exc:
        logger.warning("authz: policy enforcement failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc
