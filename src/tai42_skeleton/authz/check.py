"""The single tool-edge authorization entry point.

``check`` applies the HTTP edge's terms — route→resource verifier, policy/jq fences,
per-tag LEVEL decision — to the path SYNTHESIZED from the operation's route template
plus the call's path arguments, in the same fail-closed conjunction. Raises
:class:`PermissionDenied` on a deny, returns on an allow, never grants on an error.

Path arguments are caller-supplied, so the synthesized path is pinned twice before any
layer reads it: substitution refuses a value that does not fill the segment(s) its
parameter declares, and the result must resolve back to the operation's OWN registered
route.

Unlike ``ResourceGuardMiddleware`` CASE A, an operation with no configured resource row
is denied for EVERY caller, super-admins included: the tool edge is never easier than
the route.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from jinja2 import TemplateError
from starlette.authentication import AuthenticationError
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy, JqAuthContext
from tai42_contract.app import tai42_app
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.access_control.backend import effective_scopes
from tai42_skeleton.access_control.path_canon import MalformedPathError, canonicalize_path
from tai42_skeleton.access_control.policy import PolicyEnforcer, policy_is_empty
from tai42_skeleton.access_control.role_gate import resolve_route_meta
from tai42_skeleton.access_control.role_grants import role_level_decision_for_route
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.access_control.verifier import AccessControlVerifier, is_always_public_prefix
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.authz.token_free import TokenFreeConditionError, assert_token_free_evaluable
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.template import TemplateNotFoundError

if TYPE_CHECKING:
    from tai42_skeleton.access_control.settings import AccessControlSettings
    from tai42_skeleton.app.route_registry import RouteMetadata
    from tai42_skeleton.operations.registry import OperationMetadata

logger = logging.getLogger(__name__)

# Route-template parameter: name, plus declared converter (``{id}`` → None).
_PATH_PARAM = re.compile(r"\{([^}:]+)(?::([^}]+))?\}")

# The only converter whose value may span several path segments; any other names exactly one.
_MULTI_SEGMENT_CONVERTER = "path"

# Segments a path argument may never contribute: they collapse or re-parent the path.
_UNSAFE_SEGMENTS = frozenset({"", ".", ".."})


def synthesize_path(op: OperationMetadata, call_arguments: dict[str, object]) -> str:
    """The concrete resource path for ``op``, substituting the call's path args into the
    route template — the shape the HTTP edge's already-decoded ``scope["path"]`` carries.

    A value may fill only the segment(s) its parameter declares (``{name}`` exactly one,
    ``{name:path}`` one or more) and may contribute no empty or ``.``/``..`` segment;
    anything else raises :class:`PermissionDenied`.
    """
    if op.route_template is None:
        raise ValueError(f"operation {op.name!r} has no route template; it was never registered as a route")

    def _sub(match: re.Match[str]) -> str:
        param = match.group(1)
        if param not in call_arguments:
            raise PermissionDenied(f"access denied: missing path argument {param!r} for {op.name!r}")
        value = str(call_arguments[param])
        if match.group(2) != _MULTI_SEGMENT_CONVERTER and "/" in value:
            raise PermissionDenied(
                f"access denied: path argument {param!r} for {op.name!r} spans more than one path segment"
            )
        if any(segment in _UNSAFE_SEGMENTS for segment in value.split("/")):
            raise PermissionDenied(f"access denied: path argument {param!r} for {op.name!r} is not a path segment")
        return value

    return _PATH_PARAM.sub(_sub, op.route_template)


def _own_route(op: OperationMetadata, path: str, method: str) -> RouteMetadata:
    """The registered route ``op`` dispatches as, asserted to be the one ``path`` resolves
    to; denies if ``path`` resolves to no route or a different one.

    Resolved ONCE here and reused by every term below: re-resolving the caller-influenced
    path per term could let terms disagree on which route is authorized, and an
    unresolvable path reads as "not gated" to the per-tag decision, dropping the fence.
    """
    template = op.route_template
    assert template is not None  # synthesize_path already refused a template-less operation
    try:
        canonical = canonicalize_path(path)
    except MalformedPathError as exc:
        raise PermissionDenied(f"access denied: {method} {path} is not a well-formed path for {op.name!r}") from exc
    meta = resolve_route_meta(canonical, method)
    if meta is None or canonicalize_path(meta.path) != canonicalize_path(template):
        raise PermissionDenied(
            f"access denied: {method} {path} does not resolve to the route {op.name!r} is registered at"
        )
    return meta


def _assert_execution_condition_evaluable(condition: str, *, principal: str, template_id: str | None) -> None:
    """Deny unless a RENDERED policy condition is evaluable under the reduced claim set a
    background execution carries.

    Must be asserted on the rendered text about to be enforced, not only at bind time: a
    template edit changes that text with no write to the bound record.
    """
    try:
        assert_token_free_evaluable(condition)
    except TokenFreeConditionError as exc:
        logger.warning(
            "authz: background execution denied — the policy condition enforced for %s (template %r) is not "
            "token-free-evaluable: %s",
            principal,
            template_id,
            exc,
        )
        raise PermissionDenied(f"access denied: policy condition for {principal!r} is not evaluable at a fire") from exc


async def _render_condition(policy: AccessPolicy, *, principal: str) -> str:
    """``policy``'s condition as the text ``enforce`` will evaluate.

    A render failure is a typed refusal naming the principal — never read as "no
    condition" and never flattened into the generic catch-all, which would drop the very
    fences this decision applies. The render error's own text is logged, not answered: it
    can quote template content.
    """
    try:
        return await tai42_app.storage.resource_manager.render_by_id_or_content(
            content=policy.condition,
            template_id=policy.condition_id,
            kwargs=policy.condition_kwargs,
        )
    except (ValueError, TemplateError, TemplateNotFoundError) as exc:
        logger.warning(
            "authz: denied — the policy condition of %s (template %r) does not render: %s",
            principal,
            policy.condition_id,
            exc,
        )
        raise PermissionDenied(f"access denied: the policy condition of {principal!r} does not render") from exc


_verifier: tuple[AccessControlSettings, AccessControlVerifier] | None = None


def _tool_edge_verifier(settings: AccessControlSettings) -> AccessControlVerifier:
    """The ONE verifier this edge resolves routes through, memoized per settings object.

    Its route/pattern caches are keyed on the policy version each decision reads live, so
    reuse costs nothing against revocation; a fresh instance per dispatch would make them
    dead. Memo is keyed on the settings OBJECT — a different settings object gets its own
    verifier. Only the route→resource map is shared; policy/context/grant reads stay
    per-decision.
    """
    global _verifier
    if _verifier is None or _verifier[0] is not settings:
        _verifier = (settings, AccessControlVerifier(settings))
    return _verifier[1]


@register_settings_reset
def reset_tool_edge_verifier() -> None:
    """Drop the memoized tool-edge verifier so a settings reload rebuilds it."""
    global _verifier
    _verifier = None


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

    The HTTP edge's terms, all of them, ANDed fail-closed: route→resource resolution, the
    caller's policy (and an owned key's owner's), the scope test, the jq fences, and the
    per-tag LEVEL pass. The pre-auth login surface short-circuits ahead of all of them but
    never ahead of the route pin. The store's policy version is read ONCE and threaded
    through every versioned read, so no layer answers from a pre-bump cache slot while
    another answers from a post-bump one.

    Everything deciding the CALLER's authority is read live per decision, so a revocation
    lands on a fire's very next dispatch.
    """
    ac_settings = settings if settings is not None else access_control_settings()
    if not ac_settings.enable:
        return
    if caller_identity.is_internal:
        return
    user_id = caller_identity.user_id
    if user_id is None:
        raise PermissionDenied("access denied: no caller identity for an external tool dispatch")

    # A background fire rather than a request; keys several terms of the shared tail.
    is_execution_fire = get_execution_identity() is not None

    # The one target every layer of the tail keys on, pinned to the operation's OWN
    # registered route before ANY layer — the always-public short-circuit included — reads
    # it. Method defaults to POST for a route that declares none.
    path = synthesize_path(operation_metadata, call_arguments)
    method = operation_metadata.http_method or "POST"
    route = _own_route(operation_metadata, path, method)

    await _authorize_pinned_route(
        caller_identity,
        ac_settings,
        user_id=user_id,
        path=path,
        method=method,
        route=route,
        is_execution_fire=is_execution_fire,
    )


async def _authorize_pinned_route(
    caller_identity: CallerIdentity,
    ac_settings: AccessControlSettings,
    *,
    user_id: str,
    path: str,
    method: str,
    route: RouteMetadata,
    is_execution_fire: bool,
) -> None:
    """The post-pin authorization tail, over a target already resolved to ``path``,
    ``method`` and the registered ``route`` it dispatches as.

    The ONE spelling of the HTTP edge's decision downstream of the route pin, shared by
    :func:`check` and
    :func:`~tai42_skeleton.authz.execution.authorize_execution_agent_run` so neither can
    drift onto a narrower one. ``is_execution_fire`` is the caller's to decide; it keys
    the deleted-principal refusal, the fingerprint re-assert, the live effective-scope
    derivation and the token-free-evaluable re-assert. Returns on an allow; raises
    :class:`PermissionDenied` on a deny, and never grants on a read fault.
    """
    # The store's policy VERSION, read ONCE and threaded through EVERY versioned read
    # below, so no layer serves a pre-bump cached copy while another serves a post-bump
    # one. A read fault denies fail-closed.
    enforcer = PolicyEnforcer(ac_settings)
    try:
        version = await enforcer.current_policy_version()
    except Exception as exc:
        logger.warning("authz: policy version read failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc

    # 1. Route -> resource ids, through the edge's one memoized verifier.
    verifier = _tool_edge_verifier(ac_settings)
    try:
        resource_ids = await verifier.resolve_resource_ids(path, policy_version=version)
    except Exception as exc:
        logger.warning("authz: route resolution failed for %s — denying", path, exc_info=True)
        raise PermissionDenied("access denied") from exc

    if not resource_ids:
        raise PermissionDenied(f"access denied: no resource configured for {method} {path}")

    # The pre-auth login surface is public regardless of the policy layers and
    # short-circuits ahead of every one of them, as it does at the HTTP edge; running them
    # would make it HARDER to reach as a tool than as its route.
    if is_always_public_prefix(canonicalize_path(path), ac_settings):
        return

    public = ac_settings.public_resource_id
    # Public only when public is the ONLY resolved id (deny wins), and publicness relaxes
    # the SCOPE test alone — the policy, jq and LEVEL passes still run.
    is_public = set(resource_ids) == {public}

    # 2. Policy + live context + scopes, pinned to ``version``.
    try:
        policy = await enforcer.get_policy_at(user_id, version)
        context = await enforcer.get_live_context(user_id)
    except Exception as exc:
        logger.warning("authz: policy/context fetch failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc

    if policy.policy_data.get("disabled") is True:
        raise PermissionDenied("access denied: principal is disabled")

    # A fire's identity is built once at fire-open, so only the store can say the key still
    # EXISTS; an empty policy is what a deleted key reads as, denying its next dispatch. A
    # real request cannot reach here deleted — its credential fails to verify.
    if is_execution_fire and policy_is_empty(policy):
        raise PermissionDenied("access denied: principal has no policy")

    # The fire's bound per-mint fingerprint is re-asserted against the LIVE policy every
    # dispatch, so a within-fire revoke+remint of the same ``user_id`` is denied here
    # rather than authorized against the reminted key's grants.
    if is_execution_fire:
        bound_fingerprint = caller_identity.execution_key_fingerprint
        if bound_fingerprint is None:
            # An invariant breach: a gate-on execution identity always carries one. Refuse
            # loudly rather than leave the fire's dispatches unbound to a key identity.
            raise PermissionDenied("access denied: bound execution identity carries no key fingerprint")
        # Imported at call time: the execution module imports this one.
        from tai42_skeleton.authz.execution import assert_policy_matches_fingerprint

        assert_policy_matches_fingerprint(policy, user_id, bound_fingerprint=bound_fingerprint)

    # The caller's verified token claims; empty only on the internal/direct-construction
    # path, matching a request that carried no claims.
    claims: dict[str, Any] = dict(caller_identity.claims) if caller_identity.claims is not None else {}

    # Owned/delegated key: read from the same claim the HTTP backend reads, so the owner
    # second-pass enforce runs for exactly the same keys. A disabled or policy-less owner
    # denies fail-closed.
    owner = claims.get(OWNER_USER_ID_CLAIM)
    owner_policy = None
    if owner is not None:
        try:
            owner_policy = await enforcer.get_policy_at(owner, version)
        except Exception as exc:
            logger.warning("authz: owner policy fetch failed for %s — denying", owner, exc_info=True)
            raise PermissionDenied("access denied") from exc
        if owner_policy.policy_data.get("disabled") is True:
            raise PermissionDenied("access denied: owner is disabled")
        if policy_is_empty(owner_policy):
            raise PermissionDenied("access denied: owner has no policy")

    # The scope set. On the request path this CONSUMES the auth backend's already-decided
    # effective scopes (owner-attenuated), never re-deriving the attenuation; it falls back
    # to the caller's own policy scopes only when none was carried. A background fire
    # carries no attenuation decision, so the set is derived here from the policies just
    # read live — narrowing a running key's scopes denies its very next dispatch.
    if is_execution_fire:
        scopes = effective_scopes(policy.scopes, owner_policy.scopes) if owner_policy is not None else policy.scopes
    elif caller_identity.effective_scopes is not None:
        scopes = list(caller_identity.effective_scopes)
    else:
        scopes = policy.scopes
    if not is_public:
        protected_ids = [rid for rid in resource_ids if rid != public]
        has_permission = "*" in scopes or all(rid in scopes for rid in protected_ids)
        if not has_permission:
            raise PermissionDenied("access denied: insufficient scope")

    # 3. The jq policy fences over the synthesized path, keyed on {"method", "path"}: the
    # key's condition, then — for an owned key — the owner's as a SEPARATE enforce pass
    # over a context built from the OWNER's policy_data + scopes. Two sequential enforce
    # calls are semantically AND; never concatenate the jq strings.
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
    # A fire presents no token, so its ``.identity`` carries only the stored owner claim;
    # each rendered condition is re-asserted token-free-evaluable before being enforced.
    # An ordinary request carries full claims and skips this.
    try:
        condition = await _render_condition(policy, principal=user_id)
        if is_execution_fire and condition:
            _assert_execution_condition_evaluable(condition, principal=user_id, template_id=policy.condition_id)
        await enforcer.enforce(jq_context.model_dump(), condition, condition_configured=condition_configured)

        if (
            owner is not None
            and owner_policy is not None
            and (owner_policy.condition is not None or owner_policy.condition_id is not None)
        ):
            owner_context = JqAuthContext(
                sub=user_id,
                scopes=owner_policy.scopes,
                identity=claims,
                policy=owner_policy.policy_data,
                context=context,
                request={"method": method, "path": path},
                system={"time": time.time()},
            )
            owner_condition = await _render_condition(owner_policy, principal=owner)
            if is_execution_fire and owner_condition:
                _assert_execution_condition_evaluable(
                    owner_condition, principal=owner, template_id=owner_policy.condition_id
                )
            await enforcer.enforce(owner_context.model_dump(), owner_condition, condition_configured=True)
    except AuthenticationError as exc:
        raise PermissionDenied("access denied: policy condition rejected") from exc
    except PermissionDenied:
        # The render and token-free-evaluable refusals are already final decisions; re-raise
        # so the catch-all below cannot flatten them into a generic denial.
        raise
    except Exception as exc:
        logger.warning("authz: policy enforcement failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc

    # 4. The per-tag LEVEL pass over the pinned route — never re-resolved from the
    # caller-influenced path — with the policies already read above and keyed on the SAME
    # version, so the grant cache answers from their generation. It fences a
    # fenced/secret operation to an admin. An infra fault fails closed.
    try:
        allowed, cause = await role_level_decision_for_route(policy, owner_policy, route, method, version)
    except Exception as exc:
        logger.warning("authz: per-tag level resolution failed for %s — denying", user_id, exc_info=True)
        raise PermissionDenied("access denied") from exc

    if not allowed:
        logger.warning(
            "authz: per-tag level denied %s on %s %s (%s)",
            user_id,
            method,
            path,
            cause.value if cause is not None else "deny",
        )
        raise PermissionDenied(f"access denied: {method} {path} is not permitted for {user_id!r}")
