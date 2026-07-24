"""Caller resolution and the operations-layer authority rules keyed on it: may THIS caller
do that to THAT principal?

Shared by the operation leaves, so it lives here rather than in one of them: a leaf is
popped from ``sys.modules`` and re-imported on reload, and a rule held by one leaf and
imported by another would leave the two holding different module objects.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM, get_current_user_id
from tai42_contract.access_control.models import AccessPolicy

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.policy import PolicyEnforcer, policy_is_empty
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.access_control.user import is_admin_policy
from tai42_skeleton.authz.execution import (
    ExecutionConditionError,
    ExecutionKeyAuthorityError,
    assert_execution_key_evaluable,
    assert_key_carries_authority,
)
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.operations.errors import BadRequestError, ForbiddenError, NotFoundError, OperationFailed

logger = logging.getLogger(__name__)


# Fingerprint bound with access control OFF; never equals a real (uuid4-hex) mint, so such a
# record fails closed once the gate is ON.
GATE_OFF_EXECUTION_FINGERPRINT = "unanchored:access-control-disabled"


def owner_of(policy_data: Mapping[str, Any] | None) -> str | None:
    """The management/listing OWNER a principal's ``policy_data`` declares, or ``None``
    for a top-level (unowned) key or an absent/empty mapping.

    The one place the operations layer spells the ownership home, so every surface keying
    on it moves together if that home ever does."""
    return (policy_data or {}).get(OWNER_USER_ID_CLAIM)


@dataclass(frozen=True)
class Caller:
    """The acting principal of the current dispatch, resolved for the ownership rules."""

    caller_id: str | None
    policy: AccessPolicy
    is_admin: bool
    owner_claim: str | None


async def resolve_caller() -> Caller:
    """Resolve the ACTING principal of this dispatch and classify it for the ownership
    rules.

    Inside a background fire the acting principal is the bound EXECUTION identity, which
    takes precedence: the request-scope caller id is then unset or still the principal of
    whichever request triggered the fire, which may be unrelated to the record written.
    Both cases run the SAME classification, so the gate can never key on one principal
    while the dispatch runs as another.

    ``owner_claim`` comes from the principal's OWN stored policy_data, never the
    request-scope claims var. ``is_admin`` iff the caller holds a condition-free ``"*"``
    policy AND is not itself an owned key — role-holders carry ``["*"]`` plus a jq
    condition, and an editor-minted condition-free owned key would otherwise read as admin.

    Gate OFF ⇒ admin (nothing to classify; the surfaces are already open). Gate ON with NO
    principal bound is an invariant breach: RAISE the typed 500 rather than escalate
    silently, logging what was missing and answering generically — the typed error's text
    reaches the caller."""
    settings = access_control_settings()
    if not settings.enable:
        return Caller(caller_id=None, policy=AccessPolicy(scopes=["*"]), is_admin=True, owner_claim=None)

    execution_identity = get_execution_identity()
    caller_id = execution_identity.user_id if execution_identity is not None else get_current_user_id()
    if caller_id is None:
        logger.error(
            "access_control: no acting principal is bound — neither an execution identity (which every "
            "background fire binds) nor the request-scoped caller user id (which the guard middleware binds "
            "on every authed request); refusing to resolve an acting principal"
        )
        raise OperationFailed("access_control: internal authority-resolution failure")

    policy = await PolicyEnforcer(settings).get_policy(caller_id)
    owner_claim = owner_of(policy.policy_data)
    return Caller(
        caller_id=caller_id,
        policy=policy,
        is_admin=is_admin_policy(policy, owner_claim),
        owner_claim=owner_claim,
    )


async def require_owned_by_caller(caller: Caller, user_id: str) -> None:
    """A non-admin caller may act only on a key whose stored management/listing owner
    (``policy_data[OWNER_USER_ID_CLAIM]``) is the caller. Raises ``NotFoundError`` for an
    unknown key and ``ForbiddenError`` for someone else's key."""
    body = await management.get_policy_body(user_id)
    if body is None:
        raise NotFoundError(f"user not found: {user_id!r}")
    if owner_of(body.get("policy_data")) != caller.caller_id:
        raise ForbiddenError("you may only act on API keys you own")


def require_admin(caller: Caller) -> None:
    """Admin-only gate for the control-plane surfaces (policy/role version history,
    rollback, role management): a version body carries the raw jq condition, and a
    rollback can restore a more-privileged definition. Raises ``ForbiddenError`` for a
    non-admin. Allows with access control off, where every caller resolves as admin."""
    if not caller.is_admin:
        raise ForbiddenError("this operation is restricted to administrators")


def _unevaluable_key_refusal(
    caller: Caller, execution_key: str, policy: AccessPolicy, exc: ExecutionConditionError
) -> BadRequestError:
    """The 400 the bind door owes a caller whose execution key carries a policy condition a
    tokenless fire cannot evaluate — carrying the scan's DIAGNOSTIC only when it is the
    caller's to read.

    The diagnostic quotes a RAW jq condition, which is store-secret, and the refusal may be
    about the key OR its owner. On a door reachable with ``hooks`` write alone, a non-admin
    would otherwise read its owner's id and condition excerpt out of a 400; that caller
    gets the principal and the rule, with the diagnostic logged server-side.
    """
    caller_may_read = caller.is_admin or (
        exc.principal == execution_key and owner_of(policy.policy_data) == caller.caller_id
    )
    if caller_may_read:
        return BadRequestError(str(exc))
    logger.warning(
        "access_control: %r may not bind execution key %r — the stored policy condition of %r is not "
        "token-free-evaluable: %s",
        caller.caller_id,
        execution_key,
        exc.principal,
        exc,
    )
    return BadRequestError(
        f"the policy condition of {exc.principal!r} is not evaluable by a background execution; it must be "
        f"repaired before {execution_key!r} can be bound"
    )


def _unfireable_key_refusal(caller: Caller, execution_key: str, exc: ExecutionKeyAuthorityError) -> BadRequestError:
    """The 400 the bind door owes a caller whose execution key cannot carry a fire at all
    — naming the failing PRINCIPAL only when that principal is the caller's to name.

    The refusal may be about the key or its OWNER. A caller that cleared pass-role by
    binding its OWN identity is not that owner and would otherwise read the owner's id out
    of a 400 on a door reachable with ``hooks`` write alone; it is told the defect and
    which side carries it, with the id logged server-side."""
    if caller.is_admin or exc.principal in (execution_key, caller.caller_id):
        subject = (
            f"execution key {execution_key!r}"
            if exc.principal == execution_key
            else f"owner {exc.principal!r} of execution key {execution_key!r}"
        )
        return BadRequestError(f"{subject} {exc.defect}, so no background execution can run under {execution_key!r}")
    logger.warning(
        "access_control: %r may not bind execution key %r — its owner %r %s",
        caller.caller_id,
        execution_key,
        exc.principal,
        exc.defect,
    )
    return BadRequestError(
        f"the owner of execution key {execution_key!r} {exc.defect}, so no background execution can run under it"
    )


async def assert_execution_key_bindable(caller: Caller, execution_key: str) -> str:
    """Assert that ``caller`` may bind ``execution_key`` as the identity a background record
    fires as, and return the key's server-derived per-mint fingerprint for the record to
    store; gate off returns :data:`GATE_OFF_EXECUTION_FINGERPRINT`.

    A non-admin's refusal is a UNIFORM 403 across absent-key and not-yours, so this door is
    no probe for api-key ids; every check after the pass-role test runs only for a caller
    that cleared it.
    """
    settings = access_control_settings()
    if not settings.enable:
        return GATE_OFF_EXECUTION_FINGERPRINT

    # One enforcer serves both halves, so the key's policy row the pass-role test reads
    # is the one the token-free scan asserts against rather than a second read of it.
    enforcer = PolicyEnforcer(settings)
    policy = await enforcer.get_policy(execution_key)
    exists = not policy_is_empty(policy)
    if caller.is_admin:
        if not exists:
            raise NotFoundError(f"user not found: {execution_key!r}")
    elif not exists or (execution_key != caller.caller_id and owner_of(policy.policy_data) != caller.caller_id):
        raise ForbiddenError("you may only bind your own identity or an execution key you own")

    # An existing key carrying no anchor is corrupt (every mint stamps one): refuse loudly
    # rather than bind a record no fire could resolve.
    fingerprint = policy.policy_data.get(KEY_FINGERPRINT_CLAIM)
    if not isinstance(fingerprint, str) or not fingerprint:
        logger.error(
            "access_control: execution key %r has a policy but no %s claim; refusing to bind an unanchored record",
            execution_key,
            KEY_FINGERPRINT_CLAIM,
        )
        raise OperationFailed("access_control: internal key-fingerprint resolution failure")

    try:
        await assert_key_carries_authority(enforcer, execution_key, bound_fingerprint=fingerprint)
    except ExecutionKeyAuthorityError as exc:
        raise _unfireable_key_refusal(caller, execution_key, exc) from exc

    try:
        await assert_execution_key_evaluable(enforcer, execution_key)
    except ExecutionConditionError as exc:
        raise _unevaluable_key_refusal(caller, execution_key, policy, exc) from exc

    return fingerprint
