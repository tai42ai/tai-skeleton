"""Authorization for a fire that runs under a bound EXECUTION KEY.

Everything here is tokenless — a background fire presents no credential, so every fact is
read LIVE from the policy store at the moment of the fire, and any authority reduction on
the key or its owner lands on the very next fire with nothing to invalidate.

* :func:`build_execution_identity` / :func:`assert_key_carries_authority` — the tokenless
  analogue of the HTTP auth backend's policy stage, with the owner taken from the key's
  stored ``policy_data`` rather than a token claim.
* :func:`bind_execution_identity` — the ONE set/``finally``-reset of the contextvar.
* :func:`assert_execution_key_evaluable` — the key-level token-free-evaluable rule every
  surface storing a record that names an execution key runs before the write;
  :class:`ExecutionKeyScan` runs it over a batch, reading each distinct key once.
* :func:`authorize_execution_tool_call` — the per-call decision at the dispatch seam.

Granularity mirrors the MCP edge: an OPERATION tool takes the full per-call decision
(:func:`~tai42_skeleton.authz.check.check`, whose LEVEL pass fences a ``fenced``/``secret``
operation to an admin), a capability tool takes only the liveness question.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from jinja2 import TemplateError
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control.path_canon import canonicalize_path
from tai42_skeleton.access_control.policy import PolicyEnforcer, policy_is_empty
from tai42_skeleton.access_control.role_gate import resolve_route_meta
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.authz.check import _authorize_pinned_route, check
from tai42_skeleton.authz.execution_identity import reset_execution_identity, set_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.authz.resolver import resolve_dispatch
from tai42_skeleton.authz.token_free import TokenFreeConditionError, assert_token_free_evaluable
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.template import TemplateNotFoundError

if TYPE_CHECKING:
    from tai42_skeleton.access_control.settings import AccessControlSettings


async def build_execution_identity(execution_key: str, *, bound_fingerprint: str) -> CallerIdentity:
    """The identity a fire bound to ``execution_key`` runs as, built from that key's
    CURRENT stored grants.

    ``bound_fingerprint`` is the record's captured per-mint key identity; the live key must
    still carry it, and it is carried onto the built identity so a mid-turn re-read asserts
    the same equality. ``claims`` carries the owner reference alone — omitted entirely for
    an ownerless key, so the owner second-pass is skipped as for a real ownerless
    credential. No ``effective_scopes`` is carried: the check re-derives the owner
    attenuation live at every dispatch, which is what makes a mid-fire de-scope land.

    Raises :class:`PermissionDenied` when the key cannot carry authority at all (no stored
    policy, disabled, disabled/policy-less owner, fingerprint mismatch) — refusing here,
    rather than returning an authority-less identity, is what stops a capability-tool fire
    from running under a key that no longer exists. With access control disabled the
    identity carries the key alone and every decision against it allows.
    """
    settings = access_control_settings()
    if not settings.enable:
        return CallerIdentity(user_id=execution_key)

    owner = await assert_key_carries_authority(
        PolicyEnforcer(settings), execution_key, bound_fingerprint=bound_fingerprint
    )
    claims = {} if owner is None else {OWNER_USER_ID_CLAIM: owner}
    return CallerIdentity(user_id=execution_key, claims=claims, execution_key_fingerprint=bound_fingerprint)


class ExecutionKeyAuthorityError(PermissionDenied):
    """A key cannot carry authority AT ALL — the refusal
    :func:`assert_key_carries_authority` raises.

    ``principal`` is whose defect it is (the key or its owner); ``defect`` is the phrase
    naming it ("is disabled") and carries NO principal id, so a door answering an untrusted
    caller can compose a refusal naming only the party that caller may read.
    """

    def __init__(self, message: str, *, principal: str, defect: str) -> None:
        super().__init__(message)
        self.principal = principal
        self.defect = defect


def _authority_refusal(execution_key: str, principal: str, defect: str) -> ExecutionKeyAuthorityError:
    subject = (
        f"execution key {execution_key!r}"
        if principal == execution_key
        else f"owner {principal!r} of execution key {execution_key!r}"
    )
    return ExecutionKeyAuthorityError(f"access denied: {subject} {defect}", principal=principal, defect=defect)


def assert_policy_matches_fingerprint(policy: AccessPolicy, execution_key: str, *, bound_fingerprint: str) -> None:
    """Assert the LIVE ``policy`` of ``execution_key`` still carries the exact per-mint
    fingerprint the binding anchored to, raising :class:`ExecutionKeyAuthorityError`
    otherwise.

    A ``user_id`` is reusable across a revoke+remint; the fingerprint is not, so the
    reminted key never inherits the old record's authority. An absent live fingerprint
    fails the same equality (``bound_fingerprint`` is always non-empty) and is refused.
    The ONE spelling of this equality — both seam branches route through it.
    """
    if policy.policy_data.get(KEY_FINGERPRINT_CLAIM) != bound_fingerprint:
        raise _authority_refusal(execution_key, execution_key, "no longer matches the bound key identity")


async def assert_key_carries_authority(
    enforcer: PolicyEnforcer, execution_key: str, *, bound_fingerprint: str
) -> str | None:
    """Assert that ``execution_key`` can carry authority AT ALL — and is still the SAME
    minted key the binding named — and answer its owner reference (``None`` if unowned).

    Raises :class:`ExecutionKeyAuthorityError` for a key with no stored policy, a disabled
    key, an owner that is disabled or has no policy, or a live fingerprint that is not
    ``bound_fingerprint``. The ONE spelling of that refusal set: write doors run it so no
    record names a key every fire would refuse, and a capability-tool dispatch runs it to
    re-assert liveness the per-call decision never re-reads.

    Key and owner grants are read against ONE already-read store version, so both answer
    from the same cache generation.
    """
    version = await enforcer.current_policy_version()
    policy = await enforcer.get_policy_at(execution_key, version)
    if policy_is_empty(policy):
        raise _authority_refusal(execution_key, execution_key, "has no policy")
    if policy.policy_data.get("disabled") is True:
        raise _authority_refusal(execution_key, execution_key, "is disabled")
    assert_policy_matches_fingerprint(policy, execution_key, bound_fingerprint=bound_fingerprint)

    owner = policy.policy_data.get(OWNER_USER_ID_CLAIM)
    if owner is None:
        return None

    owner_policy = await enforcer.get_policy_at(owner, version)
    if owner_policy.policy_data.get("disabled") is True:
        raise _authority_refusal(execution_key, owner, "is disabled")
    if policy_is_empty(owner_policy):
        raise _authority_refusal(execution_key, owner, "has no policy")

    return owner


@asynccontextmanager
async def bind_execution_identity(execution_key: str, *, bound_fingerprint: str) -> AsyncIterator[CallerIdentity]:
    """Run the ``async with`` body AS ``execution_key``, binding the identity built from
    that key's current grants into the ``execution_identity`` contextvar and releasing it
    in a ``finally`` so the binding never outlives the dispatch.

    ``bound_fingerprint`` is the firing record's captured per-mint key identity; the build
    is refused unless the live key still carries it.

    Contextvar semantics matter here: opening this block INSIDE each concurrently-fired
    task gives every task its own key with no cross-talk, and a task DETACHED inside the
    body (the tool-run supervisor) inherits the identity for its own lifetime, past the
    release — which is what keeps an async submit's later inner dispatch bound.

    Raises :class:`~tai42_skeleton.operations.errors.PermissionDenied`, body never entered,
    when the key cannot carry authority at all.
    """
    identity = await build_execution_identity(execution_key, bound_fingerprint=bound_fingerprint)
    token = set_execution_identity(identity)
    try:
        yield identity
    finally:
        reset_execution_identity(token)


class ExecutionConditionError(TokenFreeConditionError):
    """A NAMED principal's stored policy condition cannot be evaluated by a tokenless
    background execution — the one refusal type :func:`assert_execution_key_evaluable`
    and :class:`ExecutionKeyScan` raise.

    The message quotes a bounded excerpt of the RAW jq condition, which is store-secret;
    ``principal`` is a separate field so a door answering an untrusted caller can log the
    diagnostic and answer with the principal alone.
    """

    def __init__(self, message: str, *, principal: str) -> None:
        super().__init__(message)
        self.principal = principal


async def _assert_condition_evaluable(policy: AccessPolicy, *, principal: str) -> None:
    """Assert that ``policy``'s condition, RENDERED, can be evaluated by a tokenless
    background execution.

    Rendered with the identical render enforcement runs, since that text — not the stored
    template reference — is what a fire evaluates. A render failure is a loud refusal,
    NEVER read as "no condition": it would hide the very identity references being scanned
    for. Only author-fixable failures become :class:`ExecutionConditionError`; an
    infrastructure fault propagates as itself.
    """
    if policy.condition is None and policy.condition_id is None:
        # Nothing configured, so nothing to scan. A PRESENT-but-empty condition is NOT this
        # case — it is configured and still goes through the render.
        return
    try:
        condition = await tai42_app.storage.resource_manager.render_by_id_or_content(
            content=policy.condition,
            template_id=policy.condition_id,
            kwargs=policy.condition_kwargs,
        )
    except (ValueError, TemplateError, TemplateNotFoundError) as exc:
        raise ExecutionConditionError(
            f"the policy condition of {principal!r} does not render ({exc}), so it cannot be shown evaluable "
            "for a background execution; repair the condition before binding this execution key",
            principal=principal,
        ) from exc
    if not condition:
        return
    try:
        assert_token_free_evaluable(condition)
    except TokenFreeConditionError as exc:
        raise ExecutionConditionError(
            f"the policy condition of {principal!r} is unusable at a fire: {exc}", principal=principal
        ) from exc


async def assert_execution_key_evaluable(enforcer: PolicyEnforcer, execution_key: str) -> None:
    """Assert that a record naming ``execution_key`` can actually fire under it.

    A fire presents no credential, so its jq context carries only the identity claim
    readable from the store. The key's condition — and, when the key is owned, its
    OWNER's condition, which is enforced as a second pass — must not depend on any
    other one. Raises :class:`ExecutionConditionError` naming the offending reference —
    and the principal it belongs to — otherwise; each write surface maps that to its own
    typed refusal.

    Both conditions are read through the CALLER's ``enforcer`` against ONE already-read
    store version, so the pair answers from a single cache generation.

    Early rejection only — the authoritative assertion runs at the fire, on the text
    rendered then. Callers must already have established that access control is enabled;
    this does not check.
    """
    version = await enforcer.current_policy_version()
    policy = await enforcer.get_policy_at(execution_key, version)
    await _assert_condition_evaluable(policy, principal=execution_key)

    owner = policy.policy_data.get(OWNER_USER_ID_CLAIM)
    if owner is not None:
        owner_policy = await enforcer.get_policy_at(owner, version)
        await _assert_condition_evaluable(owner_policy, principal=owner)


class ExecutionKeyScan:
    """The RECORD-level halves of the execution-key bind gate across ONE batch of records,
    reading each distinct execution key exactly once.

    The same two key questions the single-record bind door asks — can carry a fire at all,
    and is token-free-evaluable. The door's third question (pass-role) is not asked: the
    restore route is admin-fenced. Batching gives every record naming a key ONE verdict,
    rather than two store states either side of a mid-batch policy edit.

    Unlike the functions it batches, this carries the access-control gate itself and
    asserts nothing when access control is disabled.
    """

    def __init__(self) -> None:
        self._enforcer = PolicyEnforcer(access_control_settings())
        self._verdict: dict[tuple[str, str], ExecutionKeyAuthorityError | ExecutionConditionError | None] = {}

    async def assert_usable(self, execution_key: str, *, bound_fingerprint: str) -> None:
        """Assert that a record may name ``execution_key`` under ``bound_fingerprint``,
        taking the verdict this batch already reached for that pair when it has one.

        Raises :class:`ExecutionKeyAuthorityError` for a key no fire could run under (or a
        fingerprint mismatch) and :class:`ExecutionConditionError` for one a tokenless fire
        could not evaluate. A refusal is cached AS a refusal, so every record naming an
        unusable key is still refused on its own terms.

        Keyed by the (key, bound_fingerprint) PAIR: the fingerprint equality is a property
        of the RECORD, so two records naming one key under different bound fingerprints
        reach different verdicts.

        Returns having asserted nothing when access control is disabled.
        """
        if not self._enforcer.settings.enable:
            return
        key = (execution_key, bound_fingerprint)
        if key in self._verdict:
            recorded = self._verdict[key]
            if isinstance(recorded, ExecutionConditionError):
                raise ExecutionConditionError(str(recorded), principal=recorded.principal) from recorded
            if recorded is not None:
                raise ExecutionKeyAuthorityError(
                    str(recorded), principal=recorded.principal, defect=recorded.defect
                ) from recorded
            return
        try:
            # Authority first: a key with no policy row has no condition, so the evaluable
            # question would pass vacuously and admit a record every fire then dies on.
            await assert_key_carries_authority(self._enforcer, execution_key, bound_fingerprint=bound_fingerprint)
            await assert_execution_key_evaluable(self._enforcer, execution_key)
        except (ExecutionKeyAuthorityError, ExecutionConditionError) as exc:
            self._verdict[key] = exc
            raise
        self._verdict[key] = None


async def authorize_execution_tool_call(
    identity: CallerIdentity,
    tool_name: str,
    call_arguments: dict[str, object],
    *,
    tool_registry: Any | None,
    preset_manager: Any | None,
) -> None:
    """Authorize ``identity`` to dispatch ``tool_name`` with ``call_arguments``.

    Returns on an allow; raises :class:`PermissionDenied` on a deny.

    ``tool_name`` is resolved through presets and extension branches to the operation it
    ultimately runs and the arguments it receives. An operation gets the full edge decision
    (:func:`check`), on policies read live. A NON-operation tool (connector, remote-MCP
    tool, toolbox) takes no per-call decision — its reach is bounded by exposure — but is
    still refused once the KEY stops carrying authority, re-read live here so a mid-turn
    revocation stops that turn's next dispatch.

    An unclassifiable dispatch raises
    :class:`~tai42_skeleton.authz.resolver.OperationSurfaceUnsettledError` (retriable
    ``reloading``) rather than being waved through as a capability, which would run a
    projected operation with no decision made about it.
    """
    settings = access_control_settings()
    if not settings.enable:
        return

    resolved = resolve_dispatch(tool_name, call_arguments, tool_registry=tool_registry, preset_manager=preset_manager)
    if resolved is None:
        # A capability tool takes no scope/jq/LEVEL decision, but the identity was built
        # once at fire-open and a turn outlives that moment, so the liveness question is
        # still asked. The identity's own terms are re-spelled here: no check() to defer to.
        if identity.is_internal:
            return
        if identity.user_id is None:
            raise PermissionDenied("access denied: no caller identity for an external tool dispatch")
        if identity.execution_key_fingerprint is None:
            # An invariant breach: a gate-on execution identity always carries one. Refuse
            # rather than re-read with no anchor, which fails open on a key carrying none.
            raise PermissionDenied("access denied: bound execution identity carries no key fingerprint")
        await assert_key_carries_authority(
            PolicyEnforcer(settings), identity.user_id, bound_fingerprint=identity.execution_key_fingerprint
        )
        return

    await check(identity, resolved.operation, resolved.call_arguments, settings=settings)


async def authorize_execution_agent_run(
    identity: CallerIdentity, agent_name: str, *, settings: AccessControlSettings | None = None
) -> None:
    """Authorize ``identity`` to run the agent ``agent_name`` through the base run door
    ``POST /api/agents/{agent_name}/runs``.

    The run door is a ``custom_route``, so it has no ``OperationMetadata`` and
    :func:`~tai42_skeleton.authz.check.check` cannot decide it. This names the concrete
    path directly and runs the SAME shared tail
    (:func:`~tai42_skeleton.authz.check._authorize_pinned_route`), so the door is no easier
    reached this way than over HTTP.

    Always a fire, so the tail's fire-mode guards all run. Returns on an allow; raises
    :class:`PermissionDenied` on a deny. Access control disabled allows, as does the
    internal principal; an identity with no user id denies fail-closed. A path resolving
    to no registered route is a fail-closed deny — the run door IS registered, so a miss
    means a torn surface, not an ungated one.
    """
    ac_settings = settings if settings is not None else access_control_settings()
    if not ac_settings.enable:
        return
    if identity.is_internal:
        return
    user_id = identity.user_id
    if user_id is None:
        raise PermissionDenied("access denied: no caller identity for an agent run")

    path = f"/api/agents/{agent_name}/runs"
    method = "POST"
    route = resolve_route_meta(canonicalize_path(path), method)
    if route is None:
        raise PermissionDenied(f"access denied: {method} {path} does not resolve to a registered route")

    await _authorize_pinned_route(
        identity,
        ac_settings,
        user_id=user_id,
        path=path,
        method=method,
        route=route,
        is_execution_fire=True,
    )
