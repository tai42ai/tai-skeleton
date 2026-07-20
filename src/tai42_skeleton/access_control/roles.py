"""Role templates: the versioned store view, the seeded defaults, the apply helper,
and the application's ``AccountsAdminServices`` implementation.

A role is a versioned template with COPY semantics: applying a role copies the
template's ``scopes`` + ``condition`` dimension into the user's ENFORCED policy;
later template edits do NOT retro-apply (re-assignment is explicit). Roles are stored
under the generic :class:`~tai42_contract.versioning.VersionedStore` as ``kind="role"``
— a third view mirroring :class:`~tai42_skeleton.access_control.policy_store.AcPolicyStore`
and :class:`~tai42_skeleton.presets.store.PresetStoreView`.

**The seeded templates carry ``"*"`` scopes and differ only by a jq condition:**
routes are operator-mapped rows, so a seeded scope re-mapping of the route table would
break every existing key on deployments that already scoped their routes. Conditions
need no route-table surgery and work on any deployment. The enforcement engine already
carries ``.request.method``/``.request.path``.

- ``admin``: unconditional ``["*"]`` — full control including access-control admin.
- ``editor``: ``["*"]`` gated by ``EDITOR_JQ`` — everything EXCEPT the access-control
  admin area, with the self-service surfaces carved back in: own API keys (the
  route-level ownership rules scope these to OWN keys, and the admin discriminator
  classifies a condition-bearing role-holder as non-admin, so those rules genuinely
  fire), the tokens payload (``GET /api/auth/tokens-payload``) and mint capabilities
  (``GET /api/auth/capabilities``), ``/api/auth/logout``, own-password change
  (``PUT /api/auth/users/me/password``), the read-only scopes listing
  (``GET /api/auth/scopes``), the caller's own capability projection (``GET /api/auth/me``),
  and one-time claim-link creation (``POST /api/auth/claim-links``, whose route-level
  ownership rule confines it to keys the caller owns / its own key).
- ``viewer``: ``["*"]`` gated by ``VIEWER_JQ`` — editor's fence AND (read-only methods
  OR logout OR the own-key surface OR own-password OR one-time claim-link creation), so a
  viewer can log out, change their own password, mint/revoke own keys, and mint a
  one-time claim link for a key it owns (attenuated to read-only power at request time by
  the backend — attenuation, not trust).

No ``/api/login`` clause exists in either string: always-public paths short-circuit to
the public resource id before any jq evaluates, so a login-namespace carve-out would be
dead text.

**Maintenance rule:** with all three roles on ``["*"]`` scopes, non-admin authorization
rides ENTIRELY on these seeded jq strings — every future admin-only route prefix MUST
be reflected here. The templates are versioned and operator-editable; the unit tests
run these strings through the REAL enforcer so a broken revision fails loudly.

The ``EDITOR_JQ``/``VIEWER_JQ`` carve-in admits the whole ``/api/auth/api-keys`` subtree
for own-key CRUD, but the policy-administration routes beneath it
(``/api/auth/api-keys/{user_id}/policy/versions`` and ``.../policy/rollback``) are
enforced ADMIN-ONLY at the route level: a non-admin editor/viewer is denied there
regardless of this jq, so it can never read another user's policy history (which leaks
raw jq conditions) nor roll an enforced policy back to a prior version.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.versioning import VersionedStore
from tai42_contract.versioning.errors import DocumentExistsError, DocumentNotFoundError

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.policy_store import ac_policy_store
from tai42_skeleton.access_control.store import access_control_store

_KIND = "role"

# ADMIN-ONLY MUTATION FENCE — full-execution / soft-restart / manifest-authority
# doors only an admin may reach; editor AND viewer are denied here regardless of method.
# ``run_tool`` is the god-tool (runs any registered tool with real side effects);
# ``reload_tool`` / ``remove_tool`` / ``reload_config`` soft-restart or rewrite the live
# registry across the fleet; ``manifest/replace`` persists a whole-manifest replacement
# and reloads it fleet-wide (it governs ``api_tools`` + module loading);
# ``fleet/reload-config`` soft-restarts the worker fleet — a recovery/ops door only, since
# every pipeline mutation already self-propagates on the bus, so a manual fleet reload is
# reserved for reconverging stranded workers and belongs to the admin;
# ``mcp-status/reload-failed`` and the
# per-server ``.../deregister`` re-probe or detach MCP servers fleet-wide. Membership
# rule: every admin-only mutating route MUST be reflected here (the same rule the
# maintenance note below states for admin-only route prefixes). The single-server
# ``reload`` (``/api/mcp-status/{title}/reload``) is deliberately NOT fenced — it stays
# reachable to editors/viewers, unchanged. The fleet census ``GET /api/fleet/workers`` is
# a read and stays unfenced. The unit tests run the composed strings through the REAL
# enforcer so a broken revision fails loudly.
_ADMIN_ONLY_MUTATIONS = (
    "/api/run-tool",
    "/api/tools/reload",
    "/api/tools/remove",
    "/api/config/reload",
    "/api/fleet/reload-config",
    "/api/manifest/replace",
    "/api/mcp-status/reload-failed",
)
# Admin-only mutating routes with a variable path segment cannot match by literal
# ``IN``; the per-server detach (``/api/mcp-status/{title}/deregister``) is fenced by
# matching the concrete path's fixed prefix + suffix (its sibling ``.../reload`` ends in
# ``/reload``, so it is not caught — preserving its non-membership).
_ADMIN_ONLY_MUTATION_SHAPES = (("/api/mcp-status/", "/deregister"),)
_MUTATION_LITERAL = "(.request.path | IN(" + ", ".join(f'"{path}"' for path in _ADMIN_ONLY_MUTATIONS) + "))"
_MUTATION_SHAPED = " or ".join(
    f'((.request.path | startswith("{pre}")) and (.request.path | endswith("{suf}")))'
    for pre, suf in _ADMIN_ONLY_MUTATION_SHAPES
)
_ADMIN_MUTATION_FENCE = f"(({_MUTATION_LITERAL} or {_MUTATION_SHAPED}) | not)"

# editor = everything except the access-control admin area, with the self-service
# surfaces carved back in (own keys / tokens-payload / capabilities / me /
# one-time claim-link creation / GET scopes / logout / own-password) — AND never the
# admin-only mutation fence.
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
EDITOR_JQ = f"({_EDITOR_AUTH_CARVE}) and {_ADMIN_MUTATION_FENCE}"

# viewer = editor's fence AND (read-only methods OR logout OR own-key surface OR
# own-password OR one-time claim-link creation), so state-changing calls are confined to
# the self-service surfaces — AND never the admin-only mutation fence.
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
VIEWER_JQ = f"({_VIEWER_AUTH_CARVE}) and {_ADMIN_MUTATION_FENCE}"


def _seeded_roles() -> list[dict[str, Any]]:
    """The default role bodies. ``policy_data`` stays in the body schema for
    forward use but is NOT applied to users (see :func:`apply_role`)."""
    return [
        {
            "name": "admin",
            "scopes": ["*"],
            "condition": None,
            "policy_data": {},
            "description": "Full control, including access-control management.",
        },
        {
            "name": "editor",
            "scopes": ["*"],
            "condition": EDITOR_JQ,
            "policy_data": {},
            "description": "Everything except access-control administration; may manage own API keys.",
        },
        {
            "name": "viewer",
            "scopes": ["*"],
            "condition": VIEWER_JQ,
            "policy_data": {},
            "description": "Read-only, plus login/logout and own-key management.",
        },
    ]


class RoleStoreView:
    """Typed role-template view delegating to a generic :class:`VersionedStore` under
    ``kind="role"``. The body is ``{scopes, condition, policy_data, description}``."""

    def __init__(self, store: VersionedStore) -> None:
        self._store = store

    async def seed(self, name: str, body: dict[str, Any]) -> bool:
        """Create the role only if it does not exist (idempotent create-only).
        Returns ``True`` when a new template was created, ``False`` when one already
        existed and was left untouched (an operator edit survives a re-seed)."""
        try:
            await self._store.create(_KIND, name, body)
            return True
        except DocumentExistsError:
            return False

    async def get_active_body(self, name: str) -> dict[str, Any]:
        """The active body of role ``name``. Raises ``DocumentNotFoundError`` when the
        role does not exist."""
        return await self._store.get_active_body(_KIND, name)

    async def list_roles(self) -> list[dict[str, Any]]:
        """Every role's active body as ``{name, scopes, condition, description}`` — the
        listing shape the roles route returns."""
        records = await self._store.list(_KIND)
        roles: list[dict[str, Any]] = []
        for record in records:
            body = await self._store.get_active_body(_KIND, record.name)
            roles.append(
                {
                    "name": record.name,
                    "scopes": list(body.get("scopes") or []),
                    "condition": body.get("condition"),
                    "description": body.get("description"),
                }
            )
        return roles


def role_store() -> RoleStoreView:
    """Build the active role view over the generic versioned store."""
    from tai42_skeleton.versioning import versioned_store

    return RoleStoreView(versioned_store())


async def seed_default_roles() -> None:
    """Seed the default admin/editor/viewer templates, idempotent create-only:
    an operator-edited template is never overwritten by a re-seed."""
    store = role_store()
    for body in _seeded_roles():
        await store.seed(body["name"], body)


async def apply_role(user_id: str, role_name: str) -> None:
    """Copy role ``role_name`` into ``user_id``'s ENFORCED policy (COPY semantics).

    Writes ONLY the ``scopes`` + condition dimension (``condition`` +
    ``condition_id`` + ``condition_kwargs``), normalizing the whole condition dimension
    together so a re-assignment never leaves a stale ``condition_id``/``condition_kwargs``
    from a prior role behind. It NEVER writes ``policy_data``: on the update path
    ``update_policy_fields`` wholesale-replaces ``policy_data`` when the key is present,
    so copying the template's ``policy_data`` would clobber the user's own — most
    critically the disabled marker (disable-then-change-role would revive the killed
    keys, a fail-OPEN on exactly the credentials disable kills). ``policy_data``
    (disabled marker, key ownership) is user-lifecycle state role application never
    touches; the template body's ``policy_data`` field is reserved for forward use.

    CREATE-OR-UPDATE: ``update_policy_fields`` is UPDATE-only and returns ``None`` when
    the user has no policy row (the bootstrap owner and every admin-created/invited user
    reach here with no row). On that sentinel this falls through to ``create_policy`` —
    an upsert — never leaving the user on the empty ``AccessPolicy()`` default (a broken
    bootstrap / silent-degrade). Raises ``KeyError`` on an unknown role (loud)."""
    try:
        body = await role_store().get_active_body(role_name)
    except DocumentNotFoundError as exc:
        raise KeyError(f"unknown role: {role_name!r}") from exc

    scopes = list(body.get("scopes") or [])
    condition = body.get("condition")
    # Normalize the rest of the condition dimension so re-assignment leaves nothing stale.
    condition_id = None
    condition_kwargs: dict[str, Any] = {}

    store = access_control_store()
    committed = await store.update_policy_fields(
        user_id,
        {"scopes": scopes, "condition": condition, "condition_id": condition_id, "condition_kwargs": condition_kwargs},
    )
    if committed is None:
        # No policy row yet — upsert a real policy so the user (including the first
        # admin owner) is never left on the empty AccessPolicy() default.
        committed = await store.create_policy(user_id, scopes, None, condition, condition_id, condition_kwargs)

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
