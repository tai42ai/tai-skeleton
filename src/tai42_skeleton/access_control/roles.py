"""Roles: the versioned store view, the seeded defaults, the LIVE apply helper, and the
application's ``AccountsAdminServices`` implementation.

A role is an operator-authored, versioned permission set under two layers. Layer 1 is a
KEPT base-tier jq security ceiling carried on ``condition`` (owner-scoping, the
``/api/auth`` control-plane gate, the viewer read-only ceiling); Layer 2 is the editable
per-tag ACCESS LEVEL map ``grants`` (feature-group tag → ``none``/``read``/``write``).
Enforcement INTERSECTS the two with the route action-class fence, fail-closed.

Roles are LIVE: a user's enforced policy carries a role-name POINTER
(``policy_data[ROLE_POINTER_KEY]``, never ``condition_id``), and enforcement resolves
the role's CURRENT grant map at request time — an edit to a role changes every holder's
reach on their next request. Roles are stored under the generic
:class:`~tai42_contract.versioning.VersionedStore` as ``kind="role"`` — a view mirroring
:class:`~tai42_skeleton.access_control.policy_store.AcPolicyStore` and
:class:`~tai42_skeleton.presets.store.PresetStoreView`.

**The seeded roles carry ``"*"`` scopes and differ by their base-tier jq + grant map:**
routes are operator-mapped rows, so a seeded scope re-mapping of the route table would
break every existing key on deployments that already scoped their routes. The base-tier
jq needs no route-table surgery and works on any deployment; the enforcement engine
carries ``.request.method``/``.request.path``.

- ``admin``: unconditional ``["*"]``, ``allow_all`` — full control including
  access-control administration. Its per-tag pass is SKIPPED at enforcement; it is
  reserved, permanent, and un-lockable.
- ``editor``: ``["*"]`` under ``EDITOR_JQ`` (the ``/api/auth`` control-plane gate with
  self-service carve-outs), with ``write`` on every grantable feature tag — everything
  EXCEPT the admin-only fence and the access-control admin area, self-service surfaces
  carved back in (own API keys, tokens payload, mint capabilities, ``/api/auth/logout``,
  own-password change, the read-only scopes listing, ``/api/auth/me``, one-time
  claim-link creation).
- ``viewer``: ``["*"]`` under ``VIEWER_JQ`` (the viewer read-only ceiling), with ``read``
  on every grantable feature tag — read-only plus login/logout and own-key management.

No ``/api/login`` clause exists in either base-tier string: always-public paths
short-circuit to the public resource id before any jq evaluates, so a login-namespace
carve-out would be dead text.

The ``EDITOR_JQ``/``VIEWER_JQ`` carve-in admits the whole ``/api/auth/api-keys`` subtree
for own-key CRUD, but the policy-administration routes beneath it
(``/api/auth/api-keys/{user_id}/policy/versions`` and ``.../policy/rollback``) are
enforced ADMIN-ONLY at the route level: a non-admin editor/viewer is denied there
regardless of this jq, so it can never read another user's policy history nor roll an
enforced policy back to a prior version.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import RoleDefinition
from tai42_contract.versioning import VersionedStore, VersionedStoreTransaction
from tai42_contract.versioning.errors import DocumentExistsError, DocumentNotFoundError
from tai42_contract.versioning.models import DocumentRecord, DocumentVersion

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.policy_store import ac_policy_store
from tai42_skeleton.access_control.store import access_control_store

_KIND = "role"

# The reserved, permanent role name: undeletable, unrenamable, non-downgradable — its jq
# base stays ``None`` and it is ``allow_all`` (the per-tag pass skipped), so an admin can
# never be locked out of the control plane from either direction.
RESERVED_ADMIN_ROLE = "admin"

# The policy_data key holding a user's LIVE role pointer — the role NAME whose CURRENT
# grant map governs the user, resolved per request. It is orthogonal metadata read ONLY
# for the grant lookup: it is NEVER routed through ``condition_id`` (which would collide
# with the ``is_admin_policy`` discriminator), and an ``allow_all``/admin policy carries
# no pointer so that discriminator holds byte-for-byte.
ROLE_POINTER_KEY = "role"

# editor = everything except the access-control admin area, with the self-service
# surfaces carved back in (own keys / tokens-payload / capabilities / me /
# one-time claim-link creation / GET scopes / logout / own-password). This is the
# base-tier control-plane ceiling; the admin-only mutation fence is the route
# action-class (enforced in code), not composed here.
_EDITOR_AUTH_CARVE = (
    '((.request.path | startswith("/api/auth")) | not) '
    'or (.request.path | startswith("/api/auth/api-keys")) '
    'or (.request.path == "/api/auth/tokens-payload") '
    'or (.request.path == "/api/auth/capabilities") '
    'or (.request.path == "/api/auth/me") '
    'or (.request.path == "/api/auth/claim-links") '
    'or ((.request.path == "/api/auth/scopes") and (.request.method == "GET")) '
    'or (.request.path == "/api/auth/logout") '
    'or (.request.path == "/api/auth/users/me/password")'
)
EDITOR_JQ = f"({_EDITOR_AUTH_CARVE})"

# viewer = read-only methods OR logout OR own-key surface OR own-password OR one-time
# claim-link creation, so state-changing calls are confined to the self-service
# surfaces; the second conjunct is the editor control-plane ceiling. This is the
# base-tier viewer read ceiling; the admin-only mutation fence is the route
# action-class (enforced in code), not composed here.
_VIEWER_AUTH_CARVE = (
    '(((.request.path | startswith("/api/auth/api-keys")) '
    'or (.request.path == "/api/auth/logout") '
    'or (.request.path == "/api/auth/users/me/password") '
    'or (.request.path == "/api/auth/claim-links") '
    'or (.request.method | IN("GET","HEAD","OPTIONS"))) '
    'and (((.request.path | startswith("/api/auth")) | not) '
    'or (.request.path | startswith("/api/auth/api-keys")) '
    'or (.request.path == "/api/auth/tokens-payload") '
    'or (.request.path == "/api/auth/capabilities") '
    'or (.request.path == "/api/auth/me") '
    'or (.request.path == "/api/auth/claim-links") '
    'or ((.request.path == "/api/auth/scopes") and (.request.method == "GET")) '
    'or (.request.path == "/api/auth/logout") '
    'or (.request.path == "/api/auth/users/me/password")))'
)
VIEWER_JQ = f"({_VIEWER_AUTH_CARVE})"


def grantable_feature_tags() -> set[str]:
    """Every feature-group tag that carries at least one GRANTABLE (``read``/``write``,
    non-fenced) gated route — the tags a per-tag level can open. A tag whose routes are
    ALL fenced/secret is admin-only and never appears here (a level can never open it)."""
    from tai42_skeleton.app.route_registry import load_all_routes

    tags: set[str] = set()
    for meta in load_all_routes():
        if meta.authed and meta.action in ("read", "write"):
            tags.update(meta.tags)
    return tags


def _seeded_roles() -> list[dict[str, Any]]:
    """The default role bodies (``RoleDefinition`` dumps). ``editor``/``viewer`` derive
    their grant maps from the live registry so a new grantable feature area joins the
    default reach automatically; the bulk-secret reads are ``action=secret`` (fenced)
    so they never appear in any grant map."""
    grantable = sorted(grantable_feature_tags())
    return [
        RoleDefinition(
            name="admin",
            description="Full control, including access-control management.",
            allow_all=True,
            grants={},
            condition=None,
        ).model_dump(),
        RoleDefinition(
            name="editor",
            description="Everything except access-control administration; may manage own API keys.",
            base_tier="editor",
            grants=dict.fromkeys(grantable, "write"),
            condition=EDITOR_JQ,
        ).model_dump(),
        RoleDefinition(
            name="viewer",
            description="Read-only, plus login/logout and own-key management.",
            base_tier="viewer",
            grants=dict.fromkeys(grantable, "read"),
            condition=VIEWER_JQ,
        ).model_dump(),
    ]


class RoleStoreView:
    """Typed role view delegating to a generic :class:`VersionedStore` under
    ``kind="role"``. The body is a :class:`RoleDefinition` dump."""

    def __init__(self, store: VersionedStore) -> None:
        self._store = store

    async def seed(self, name: str, body: dict[str, Any]) -> bool:
        """Create the role only if it does not exist (idempotent create-only). Returns
        ``True`` when a new role was created, ``False`` when one already existed and was
        left untouched (an operator edit survives a re-seed)."""
        try:
            await self._store.create(_KIND, name, body)
            return True
        except DocumentExistsError:
            return False

    async def create(self, name: str, body: dict[str, Any], tx: VersionedStoreTransaction | None = None) -> None:
        """Create a brand-new role. Raises ``DocumentExistsError`` on a name collision
        (the caller maps it to a loud 409). Runs within ``tx`` when one is supplied."""
        await self._store.create(_KIND, name, body, tx=tx)

    async def update(self, name: str, body: dict[str, Any], tx: VersionedStoreTransaction | None = None) -> None:
        """Persist an edit as a NEW version (versioned history + rollback come free).
        Raises ``DocumentNotFoundError`` when the role does not exist (loud 404). Runs
        within ``tx`` when one is supplied."""
        await self._store.save_version(_KIND, name, body, tx=tx)

    async def delete(self, name: str, tx: VersionedStoreTransaction | None = None) -> None:
        """Hard-delete the active role and its version rows. Raises
        ``DocumentNotFoundError`` when the role does not exist. Runs within ``tx`` when
        one is supplied."""
        await self._store.delete(_KIND, name, tx=tx)

    async def rename(self, name: str, new_name: str) -> DocumentRecord:
        """Re-key the role, moving its whole history untouched."""
        return await self._store.rename(_KIND, name, new_name)

    async def list_versions(self, name: str) -> list[DocumentVersion]:
        return await self._store.list_versions(_KIND, name)

    async def get_version(
        self, name: str, version: int, tx: VersionedStoreTransaction | None = None
    ) -> DocumentVersion:
        """The immutable body of one version. Runs within ``tx`` when one is supplied, so a
        rollback's before/after reads ride the transaction's connection."""
        return await self._store.get_version(_KIND, name, version, tx=tx)

    async def rollback(self, name: str, version: int, tx: VersionedStoreTransaction | None = None) -> DocumentRecord:
        return await self._store.rollback(_KIND, name, version, tx=tx)

    async def get_active_body(
        self, name: str, *, tx: VersionedStoreTransaction | None = None, for_update: bool = False
    ) -> dict[str, Any]:
        """The active body of role ``name``. Raises ``DocumentNotFoundError`` when the role
        does not exist. Passed a ``tx`` the read rides that transaction's connection (no
        second pooled connection while the transaction is open); with ``for_update=True`` it
        row-locks the active role so a read-modify-write serializes against a concurrent
        edit — the lock needs the transaction, so ``for_update`` requires ``tx``."""
        return await self._store.get_active_body(_KIND, name, tx=tx, for_update=for_update)

    async def list_roles(self) -> list[dict[str, Any]]:
        """Every role's active body as a full ``RoleDefinition``-shaped dict
        (``{name, description, scopes, condition, condition_id, condition_kwargs,
        base_tier, allow_all, grants}``) — the listing shape the roles route returns."""
        records = await self._store.list(_KIND)
        roles: list[dict[str, Any]] = []
        for record in records:
            body = await self._store.get_active_body(_KIND, record.name)
            body = {**body, "name": record.name}
            roles.append(body)
        return roles


def role_store() -> RoleStoreView:
    """Build the active role view over the generic versioned store."""
    from tai42_skeleton.versioning import versioned_store

    return RoleStoreView(versioned_store())


async def seed_default_roles() -> None:
    """Seed the default admin/editor/viewer roles, idempotent create-only: an
    operator-edited role is never overwritten by a re-seed."""
    store = role_store()
    for body in _seeded_roles():
        await store.seed(body["name"], body)


async def apply_role(user_id: str, role_name: str) -> None:
    """Assign role ``role_name`` to ``user_id``'s ENFORCED policy (LIVE semantics).

    Writes the ``scopes`` + condition dimension (``condition``/``condition_id``/
    ``condition_kwargs``, normalized together so a re-assignment never strands a prior
    role's ``condition_id``/``condition_kwargs``) AND the role-name POINTER merged into
    the user's existing ``policy_data`` — preserving the disabled marker and key
    ownership. It does NOT freeze a grant-map COPY: enforcement resolves the role's
    CURRENT grants through the pointer, so a later role edit retro-applies.

    An ``allow_all``/admin role carries NO pointer (its per-tag pass is skipped, and the
    absent pointer keeps the ``is_admin_policy`` discriminator intact) — a stale pointer
    from a prior non-admin role is dropped on the admin assignment.

    CREATE-OR-UPDATE: ``update_policy_fields`` is UPDATE-only and returns ``None`` when
    the user has no policy row (the bootstrap owner and every admin-created/invited user
    reach here with no row). On that sentinel this falls through to ``create_policy``, so
    the user is never left on the empty ``AccessPolicy()`` default. Raises ``KeyError``
    on an unknown role (loud)."""
    try:
        body = await role_store().get_active_body(role_name)
    except DocumentNotFoundError as exc:
        raise KeyError(f"unknown role: {role_name!r}") from exc

    role = RoleDefinition(**body)
    scopes = list(role.scopes)
    condition = role.condition
    # Normalize the rest of the condition dimension so re-assignment leaves nothing stale.
    condition_id = None
    condition_kwargs: dict[str, Any] = {}

    # Fail-closed guard on the admin discriminator: a non-allow_all role MUST carry a
    # base-tier condition. This guards on the condition actually WRITTEN below —
    # ``condition_id`` is normalized to ``None`` here regardless of the role body, so a
    # role carrying ``condition=None`` (whatever its stored ``condition_id``) would assign
    # a condition-free ["*"] policy (condition/condition_id both None) that
    # ``is_admin_policy`` reads as FULL ADMIN — the role pointer is orthogonal metadata the
    # discriminator ignores. Only the reserved allow_all admin role is legitimately
    # condition-free; refuse to mint an admin-shaped policy from any other role rather than
    # silently escalate it.
    if not role.allow_all and role.condition is None:
        raise ValueError(
            f"role {role_name!r} is not allow_all yet carries no base-tier condition; assigning it would "
            "produce a condition-free ['*'] policy the admin discriminator misreads as full admin — refusing"
        )

    store = access_control_store()
    existing = await store.get_policy_body(user_id)
    policy_data = dict((existing or {}).get("policy_data") or {})
    if role.allow_all:
        policy_data.pop(ROLE_POINTER_KEY, None)
    else:
        policy_data[ROLE_POINTER_KEY] = role_name

    committed = await store.update_policy_fields(
        user_id,
        {
            "scopes": scopes,
            "condition": condition,
            "condition_id": condition_id,
            "condition_kwargs": condition_kwargs,
            "policy_data": policy_data,
        },
    )
    if committed is None:
        # No policy row yet — upsert a real policy so the user (including the first
        # admin owner) is never left on the empty AccessPolicy() default.
        committed = await store.create_policy(user_id, scopes, policy_data, condition, condition_id, condition_kwargs)

    await management.bump_policy_version()
    await ac_policy_store().write(user_id, committed)


class SkeletonAccountsAdminServices:
    """The application's implementation of the ``AccountsAdminServices`` Protocol.

    Injected onto ``settings.admin`` at ``AuthAdapter`` construction so every
    accounts-provider factory reaches it as ``settings.admin`` (never by importing this
    module). Every method mutates application-owned policy state and bumps the policy
    version so enforcement follows immediately."""

    async def apply_role(self, user_id: str, role: str) -> None:
        await apply_role(user_id, role)

    async def remove_policy(self, user_id: str) -> None:
        """Delete ``user_id``'s enforced policy and revoke every key it owned.

        Owned keys are walked from the management/listing home (``policy_data``'s
        ``OWNER_USER_ID_CLAIM``). The user is expected to EXIST: a delete of a missing
        policy row is an invariant breach here (only ``apply_role`` legitimately upserts
        a missing user), so it raises rather than proceeding silently."""
        store = access_control_store()

        # Revoke keys this user owns first (before its own policy is gone), reading the
        # owner claim from the management/listing home. The enumeration is empty on a
        # validator-only deployment (no key-minting provider, so no api-keys to own).
        for entry in await management.get_all_existing_tokens_payload():
            owner = (entry.get("policy_data") or {}).get(OWNER_USER_ID_CLAIM)
            if owner == user_id:
                await management.revoke_api_key(entry["user_id"])

        if not await store.delete_policy(user_id):
            raise KeyError(f"cannot remove policy for unknown user: {user_id!r}")
        await management.bump_policy_version()

    async def set_user_disabled(self, user_id: str, disabled: bool) -> None:
        """Set/clear the disabled marker on ``user_id``'s enforced policy.

        The user is expected to EXIST: a missing policy row is an invariant breach and
        raises (only ``apply_role`` upserts a missing user)."""
        store = access_control_store()
        body = await store.get_policy_body(user_id)
        if body is None:
            raise KeyError(f"cannot set disabled marker for unknown user: {user_id!r}")

        policy_data = dict(body.get("policy_data") or {})
        if disabled:
            policy_data["disabled"] = True
        else:
            policy_data.pop("disabled", None)

        committed = await store.update_policy_fields(user_id, {"policy_data": policy_data})
        if committed is None:
            raise KeyError(f"cannot set disabled marker for unknown user: {user_id!r}")
        await management.bump_policy_version()
        await ac_policy_store().write(user_id, committed)
