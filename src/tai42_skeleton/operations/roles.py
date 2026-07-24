"""The admin-only role-management operations: create / edit / delete / version history
/ rollback over the versioned role store.

Every mutation validates the grant map BEFORE persisting (fail-closed), guards the
reserved permanent ``admin`` role (block-downgrade + block-delete), rejects deleting a
role still assigned to any principal, bumps the policy version so the LIVE grant caches
miss, and appends a who/before→after audit record. These routes live under ``/api/auth``
(the control-plane gate admin-gates them); the op-level ``require_admin`` is defense in
depth. With access control off it, like every other rule, allows: there is no principal
to classify and the surface is already reachable by anyone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, Field
from tai42_contract.access_control.models import RoleDefinition
from tai42_contract.versioning.errors import DocumentExistsError, DocumentNotFoundError, DocumentVersionNotFoundError
from tai42_kit.utils.data import get_compiled_jq

from tai42_skeleton.access_control.role_audit import role_audit
from tai42_skeleton.access_control.roles import (
    EDITOR_JQ,
    RESERVED_ADMIN_ROLE,
    ROLE_POINTER_KEY,
    VIEWER_JQ,
    grantable_feature_tags,
    role_store,
)
from tai42_skeleton.access_control.store import access_control_store
from tai42_skeleton.operations._authority import require_admin, resolve_caller
from tai42_skeleton.operations.decorator import operation
from tai42_skeleton.operations.errors import BadRequestError, ConflictError, ForbiddenError, NotFoundError

# The base-tier jq a NEW role inherits from its ``base_tier`` — resolved SERVER-SIDE from
# the seeded constants, NEVER accepted as raw jq from the operator. ``admin`` is reserved
# (its base is ``None``/``allow_all``, not operator-authored).
_BASE_TIER_CONDITIONS = {"editor": EDITOR_JQ, "viewer": VIEWER_JQ}

_GrantLevel = Literal["none", "read", "write"]
_GrantMap = dict[str, _GrantLevel]


class RoleCreate(BaseModel):
    """The create-role request body: the grant map + base tier the operator authors; the
    base-tier jq is resolved server-side (no raw jq surface)."""

    name: str
    description: str = ""
    base_tier: str
    grants: dict[str, _GrantLevel] = Field(default_factory=dict)


class RoleUpdate(BaseModel):
    """The edit-role request body: only the per-tag grant map + description are editable;
    the base-tier jq / base_tier are seed-fixed and rejected on any change attempt. Both
    fields are omit-means-keep — an absent ``grants`` (``None``) preserves the stored grant
    map (it is never silently wiped), and an absent ``description`` preserves the stored
    description."""

    grants: dict[str, _GrantLevel] | None = None
    description: str | None = None


class RoleRollback(BaseModel):
    version: int


def _validate_grants(grants: Mapping[str, str]) -> None:
    """Reject a grant on a tag that is not a GRANTABLE feature group — a nonexistent tag
    or one whose routes are ALL fenced/secret (admin-only, never opened by a level).
    A loud 400 so a typo'd or un-openable tag can never be persisted as dead access."""
    grantable = grantable_feature_tags()
    unknown = sorted(tag for tag in grants if tag not in grantable)
    if unknown:
        raise BadRequestError(
            f"grant tag(s) are not grantable feature groups (nonexistent, or all their routes are "
            f"admin-only fenced/secret): {unknown}"
        )


def _resolved_create(name: str, description: str, base_tier: str, grants: Mapping[str, str]) -> RoleDefinition:
    if name == RESERVED_ADMIN_ROLE:
        raise ForbiddenError("the 'admin' role is reserved and permanent; it cannot be created or replaced")
    if base_tier not in _BASE_TIER_CONDITIONS:
        raise BadRequestError(
            f"base_tier must be one of {sorted(_BASE_TIER_CONDITIONS)} (admin is reserved); got {base_tier!r}"
        )
    _validate_grants(grants)
    condition = _BASE_TIER_CONDITIONS[base_tier]
    # Belt-and-suspenders lock-out guard: the resolved base jq compiles cleanly (a
    # non-compiling base would deny every request the role governs). The grant LEVELS are
    # already the validated ``none``/``read``/``write`` Literal (the request model / the
    # RoleDefinition validator enforce it), so the cast only re-narrows the value type.
    get_compiled_jq(condition)
    try:
        return RoleDefinition(
            name=name,
            description=description,
            base_tier=base_tier,
            scopes=["*"],
            grants=cast("_GrantMap", dict(grants)),
            condition=condition,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@operation(
    summary="Create a role",
    tags=["access-control"],
    authority_changing=True,
    errors=[BadRequestError, ForbiddenError, ConflictError],
    request_model=RoleCreate,
)
async def create_role(name: str, description: str, base_tier: str, grants: dict[str, str]) -> dict[str, Any]:
    """Create an operator-authored role. Admin-only; validates the grant map + base tier
    before persist; 409 on a name collision. Bumps the policy version and audits."""
    caller = await resolve_caller()
    require_admin(caller)
    role = _resolved_create(name, description, base_tier, grants)
    body = role.model_dump()
    # Atomic unit of work: the role create and its audit append commit together, so a
    # live role can never exist without its audit record. The Redis version bump follows
    # strictly AFTER the commit.
    try:
        async with _versioned_store().transaction() as tx:
            await role_store().create(name, body, tx=tx)
            await role_audit().record(name, "create", caller.caller_id, None, body, tx=tx)
    except DocumentExistsError as exc:
        raise ConflictError(f"role already exists: {name!r}") from exc
    await _bump()
    return body


@operation(
    summary="Edit a role's grant map",
    tags=["access-control"],
    authority_changing=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError],
    request_model=RoleUpdate,
)
async def update_role(name: str, grants: dict[str, str] | None, description: str | None) -> dict[str, Any]:
    """Edit a role's per-tag grant map + description (the base-tier jq is seed-fixed, not
    editable here). Admin-only; guards the reserved ``admin`` role and the block-downgrade
    of any allow_all role. Both inputs are omit-means-keep — an absent ``grants``/
    ``description`` preserves the stored value, so a description-only edit never wipes the
    grant map. Validates a supplied grant map before persist; LIVE — the edit changes every
    holder's reach on their next request via the policy-version bump. Audits."""
    caller = await resolve_caller()
    require_admin(caller)
    if name == RESERVED_ADMIN_ROLE:
        raise ForbiddenError("the 'admin' role is reserved and permanent; it cannot be edited (block-downgrade)")
    if grants is not None:
        _validate_grants(grants)
    # The locking read, the compute, the edit, and the audit append all ride ONE
    # transaction on ONE connection: the ``before`` read row-locks the active role so two
    # concurrent edits serialize (no lost update, and the audit ``before`` is the truly
    # current body), and no role edit holds a second pooled connection. The version bump
    # follows the commit.
    async with _versioned_store().transaction() as tx:
        try:
            before = await role_store().get_active_body(name, tx=tx, for_update=True)
        except DocumentNotFoundError as exc:
            raise NotFoundError(f"unknown role: {name!r}") from exc
        existing = RoleDefinition(**before)
        if existing.allow_all:
            raise ForbiddenError("an allow_all role cannot be narrowed (block-downgrade)")
        updated = existing.model_copy(
            update={
                "grants": existing.grants if grants is None else dict(grants),
                "description": existing.description if description is None else description,
            }
        )
        body = updated.model_dump()
        await role_store().update(name, body, tx=tx)
        await role_audit().record(name, "edit", caller.caller_id, before, body, tx=tx)
    await _bump()
    return body


@operation(
    summary="Delete a role",
    tags=["access-control"],
    authority_changing=True,
    destructive=True,
    errors=[ForbiddenError, NotFoundError, ConflictError],
)
async def delete_role(name: str) -> dict[str, Any]:
    """Delete a role. Admin-only; the reserved ``admin`` role is undeletable; a role still
    assigned to any principal (its LIVE pointer held by any policy) is rejected loudly so
    a holder can never be orphaned. Bumps the version and audits."""
    caller = await resolve_caller()
    require_admin(caller)
    if name == RESERVED_ADMIN_ROLE:
        raise ForbiddenError("the 'admin' role is reserved and permanent; it cannot be deleted")
    # The assigned-role guard reads a DIFFERENT store (``access_control_store``) that cannot
    # ride the versioned-store transaction, so it runs BEFORE the transaction opens: inside
    # it would pin a SECOND pooled connection for the whole tx (a pool-exhaustion deadlock
    # risk under concurrent deletes if the two stores share a pool), and it gains no
    # consistency from the tx anyway — the role-row lock does not lock policy rows.
    assigned = await access_control_store().count_policies_with_role(name, pointer_key=ROLE_POINTER_KEY)
    if assigned > 0:
        raise ConflictError(f"role {name!r} is assigned to {assigned} principal(s); reassign them before deleting")
    # The locking read, the delete, and the audit append all ride ONE transaction on ONE
    # connection: the ``before`` read row-locks the active role so the audit records the
    # truly current body and the delete cannot race a concurrent edit, and the mutation
    # holds no second pooled connection. The version bump follows the commit.
    async with _versioned_store().transaction() as tx:
        try:
            before = await role_store().get_active_body(name, tx=tx, for_update=True)
        except DocumentNotFoundError as exc:
            raise NotFoundError(f"unknown role: {name!r}") from exc
        await role_store().delete(name, tx=tx)
        await role_audit().record(name, "delete", caller.caller_id, before, None, tx=tx)
    await _bump()
    return {"name": name, "deleted": True}


@operation(
    summary="A role's version history + audit trail",
    tags=["access-control"],
    errors=[ForbiddenError, NotFoundError],
)
async def list_role_versions(name: str) -> dict[str, Any]:
    """The role's append-only version history plus its who/when/before→after audit trail.
    Admin-only. A store-less deployment keeps no history, so the read is an empty pair."""
    caller = await resolve_caller()
    require_admin(caller)
    from tai42_skeleton.versioning import versioned_store_configured

    if not versioned_store_configured():
        return {"versions": [], "audit": []}
    try:
        versions = await role_store().list_versions(name)
    except DocumentNotFoundError as exc:
        raise NotFoundError(f"unknown role: {name!r}") from exc
    audit = await role_audit().list_events(name)
    return {"versions": [v.model_dump() for v in versions], "audit": [a.model_dump() for a in audit]}


@operation(
    summary="Roll a role back to a version",
    tags=["access-control"],
    authority_changing=True,
    destructive=True,
    errors=[BadRequestError, ForbiddenError, NotFoundError],
    request_model=RoleRollback,
)
async def rollback_role(name: str, version: int) -> dict[str, Any]:
    """Re-point a role's active version to a prior one (LIVE — holders follow on their
    next request). Admin-only; the reserved ``admin`` role has no editable history.
    Bumps the version and audits."""
    caller = await resolve_caller()
    require_admin(caller)
    if name == RESERVED_ADMIN_ROLE:
        raise ForbiddenError("the 'admin' role is reserved and permanent; it has no editable history")
    # The locking read, the re-point, the target-version read, and the audit append all
    # ride ONE transaction on ONE connection. ``before`` is the active body captured (and
    # row-locked) BEFORE the re-point so the audit records the truly current body and the
    # re-point cannot race a concurrent edit; ``after`` is the target version's (immutable)
    # body. Neither read opens a second pooled connection; the bump follows the commit.
    async with _versioned_store().transaction() as tx:
        try:
            before = await role_store().get_active_body(name, tx=tx, for_update=True)
        except DocumentNotFoundError as exc:
            raise NotFoundError(f"unknown role: {name!r}") from exc
        try:
            await role_store().rollback(name, version, tx=tx)
        except DocumentVersionNotFoundError as exc:
            raise NotFoundError(f"role {name!r} has no version {version}") from exc
        after = (await role_store().get_version(name, version, tx=tx)).body
        await role_audit().record(name, "rollback", caller.caller_id, before, after, tx=tx)
    await _bump()
    return after


async def _bump() -> None:
    from tai42_skeleton.access_control import management

    await management.bump_policy_version()


def _versioned_store():
    """The active versioned store, resolved lazily so it follows the same construction
    point ``role_store``/``role_audit`` build over (a single transaction spans both)."""
    from tai42_skeleton.versioning import versioned_store

    return versioned_store()
