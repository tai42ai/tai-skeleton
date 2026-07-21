"""Role templates: the seeded jq through the REAL enforcer, seed idempotence,
``apply_role`` copy/upsert semantics, and the injected ``AccountsAdminServices``.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import registry
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity, IdentityProvider
from tai42_contract.accounts import AccountsAdminServices
from tai42_kit.settings import reset_all_settings
from tai42_kit.utils.data import run_jq_first

import tai42_skeleton.versioning as versioning_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.roles import (
    EDITOR_JQ,
    VIEWER_JQ,
    SkeletonAccountsAdminServices,
    apply_role,
    role_store,
    seed_default_roles,
)
from tai42_skeleton.access_control.store import access_control_store

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx
from .test_policy_store import _MemStore


class _SpyProvider(ApiKeyIdentityProvider):
    def __init__(self) -> None:
        self.identities: dict[str, str] = {}

    async def validate_token(self, token: str):  # pragma: no cover - unused
        return None

    async def provision(self, user_id: str, description: str, *, owner_user_id: str | None = None) -> str:
        self.identities[user_id] = description
        return f"sk-{user_id}"

    async def revoke(self, user_id: str) -> bool:
        return self.identities.pop(user_id, None) is not None

    async def update_description(self, user_id: str, description: str) -> bool:  # pragma: no cover - unused
        return user_id in self.identities

    async def list_identities(self) -> list[tuple[str, str]]:
        return list(self.identities.items())


@pytest.fixture
def mem() -> _MemStore:
    return _MemStore()


@pytest.fixture(autouse=True)
def _wire_versioned_store(monkeypatch, mem: _MemStore) -> None:
    # role_store() and ac_policy_store() both build over versioned_store().
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)


@pytest.fixture
def provider() -> _SpyProvider:
    spy = _SpyProvider()
    registry._REGISTRY["redis"] = lambda _settings: spy
    return spy


@pytest.fixture
def redis_mgmt(monkeypatch) -> FakeRedis:
    fake = FakeRedis(strings={})
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(fake))
    return fake


# -- seeded jq through the REAL enforcer -------------------------------------


async def _allows(jq: str, path: str, method: str) -> bool:
    ctx = {"request": {"path": path, "method": method}}
    return (await run_jq_first(jq, ctx)) is True


# The base-tier jq carries ONLY the control-plane ceiling (the /api/auth gate + its
# self-service carve-outs + the viewer read-only ceiling): the admin-only mutation fence
# is the route action-class and is enforced in code, so it is NOT in these strings. These
# matrices pin the base-tier ceiling; the fence + per-tag level are pinned separately
# through the shared per-tag decision below.
@pytest.mark.parametrize(
    ("path", "method", "allowed"),
    [
        ("/api/tools/run", "POST", True),  # outside /api/auth → allowed by the base tier
        ("/api/config/env", "GET", True),  # a non-/api/auth read — base tier allows (fence is code)
        ("/api/fleet/workers", "GET", True),
        ("/api/mcp-status/failed", "GET", True),
        ("/api/mcp-status/gh/reload", "POST", True),  # a non-/api/auth write — base tier allows
        ("/api/auth/api-keys", "POST", True),  # own keys carve-out
        ("/api/auth/api-keys/x", "DELETE", True),  # own keys carve-out
        ("/api/auth/claim-links", "POST", True),  # one-time claim-link creation carve-in
        ("/api/auth/tokens-payload", "GET", True),
        ("/api/auth/capabilities", "GET", True),
        ("/api/auth/me", "GET", True),  # capability projection carve-in
        ("/api/auth/scopes", "GET", True),  # read-only scopes listing
        ("/api/auth/scopes", "POST", False),  # scope ADMINISTRATION stays admin-only
        ("/api/auth/public-routes", "POST", False),  # admin area gated
        ("/api/auth/roles", "GET", False),  # the roles management surface is admin-only
        ("/api/auth/logout", "POST", True),
        ("/api/auth/users/me/password", "PUT", True),
    ],
)
async def test_editor_jq_matrix(path, method, allowed):
    assert await _allows(EDITOR_JQ, path, method) is allowed


@pytest.mark.parametrize(
    ("path", "method", "allowed"),
    [
        ("/api/tools/run", "GET", True),  # read-only outside /api/auth
        ("/api/tools/run", "POST", False),  # state-changing outside self-service → read ceiling denies
        ("/api/config/env", "GET", True),  # a non-/api/auth read — base tier allows (fence is code)
        ("/api/fleet/workers", "GET", True),
        ("/api/mcp-status/failed", "GET", True),
        ("/api/mcp-status/gh/reload", "GET", True),  # a read is allowed by the viewer read ceiling
        ("/api/auth/api-keys", "POST", True),  # own keys carve-out
        # A POST claim-link fails conjunct 1's read-only test, so it passes ONLY because
        # the leg is in BOTH conjuncts — the both-conjuncts pin.
        ("/api/auth/claim-links", "POST", True),
        ("/api/auth/logout", "POST", True),
        ("/api/auth/users/me/password", "PUT", True),
        ("/api/auth/me", "GET", True),  # capability projection carve-in (read-only)
        ("/api/auth/me", "POST", False),  # only the read-only leg admits /me for a viewer
        ("/api/auth/scopes", "GET", True),
        ("/api/auth/scopes", "POST", False),
        ("/api/auth/public-routes", "GET", False),  # admin area gated even for reads
    ],
)
async def test_viewer_jq_matrix(path, method, allowed):
    assert await _allows(VIEWER_JQ, path, method) is allowed


# -- the admin-only fence is the route action-class, enforced in code --------

# These privileged mutations carry ``action=fenced``, so the per-tag decision denies
# them to every non-admin (editor + viewer) regardless of any granted level, while an
# admin (allow_all) governing policy skips the pass and reaches them. This is exactly the
# pre-M20 admin-only mutation set (marketplace/backup/run-tool/tool+config+fleet reload/
# manifest-replace/failed-MCP re-probe/per-server deregister/env write).
_PRIVILEGED_MUTATIONS = [
    ("/api/marketplace/install", "POST"),
    ("/api/marketplace/uninstall", "POST"),
    ("/api/marketplace/update", "POST"),
    ("/api/backup/import", "POST"),
    ("/api/backup/export", "POST"),
    ("/api/run-tool", "POST"),
    ("/api/tools/reload", "POST"),
    ("/api/tools/remove", "POST"),
    ("/api/config/reload", "POST"),
    ("/api/fleet/reload-config", "POST"),
    ("/api/manifest/replace", "POST"),
    ("/api/mcp-status/reload-failed", "POST"),
    ("/api/mcp-status/gh/deregister", "POST"),
    ("/api/config/env", "POST"),  # write_env — rewrites arbitrary env + fleet reload
]


async def _level_allows(role_name: str, path: str, method: str) -> tuple[bool, object]:
    from tai42_contract.access_control.models import AccessPolicy

    from tai42_skeleton.access_control.role_grants import role_level_decision

    # A non-admin policy carries its base-tier condition, so ``is_admin_policy`` is False
    # (a scopes-only ["*"] with no condition would read as admin and skip the pass).
    base = {"editor": EDITOR_JQ, "viewer": VIEWER_JQ}[role_name]
    policy = AccessPolicy(scopes=["*"], condition=base, policy_data={"role": role_name})
    return await role_level_decision(policy, None, path, method, 0)


@pytest.mark.parametrize(("path", "method"), _PRIVILEGED_MUTATIONS)
async def test_fenced_routes_denied_to_every_non_admin(mem: _MemStore, path, method):
    from tai42_skeleton.access_control.role_gate import DenialCause

    await seed_default_roles()
    for role in ("editor", "viewer"):
        allowed, cause = await _level_allows(role, path, method)
        assert allowed is False
        assert cause is DenialCause.HARD_FENCE  # the fence, not a level-miss


async def test_admin_skips_the_per_tag_pass_and_reaches_fenced(mem: _MemStore, pg: FakeAccessControlPg, redis_mgmt):
    from tai42_contract.access_control.models import AccessPolicy

    from tai42_skeleton.access_control.role_grants import role_level_decision

    await seed_default_roles()
    # An admin governing policy (allow_all → condition None, no role pointer) skips the
    # per-tag pass, so a fenced route is reached.
    admin = AccessPolicy(scopes=["*"], policy_data={})
    allowed, cause = await role_level_decision(admin, None, "/api/marketplace/install", "POST", 0)
    assert allowed is True
    assert cause is None


async def test_access_control_admin_secret_reads_are_not_grantable(mem: _MemStore):
    from tai42_skeleton.access_control.role_gate import DenialCause

    await seed_default_roles()
    # The bulk-secret reads stay action=secret: denied to editor/viewer (hard fence), and no
    # granted level can open them. Two planes: the access-control-admin reads (raw jq + version
    # audit) and the two config bulk reads (settings==env==one admin-owned store).
    for path in (
        "/api/auth/roles",
        "/api/auth/roles/x/versions",
        "/api/auth/api-keys/u/policy/versions",
        "/api/config/env",
        "/api/config/settings-schema",
    ):
        for role in ("editor", "viewer"):
            allowed, cause = await _level_allows(role, path, "GET")
            assert allowed is False
            assert cause is DenialCause.HARD_FENCE


async def test_hooks_listing_is_grantable_to_editor_and_viewer(mem: _MemStore):
    from tai42_skeleton.access_control import role_grants as role_grants_module
    from tai42_skeleton.access_control.role_gate import reset_route_index

    # The hooks listing is action=read (editor/viewer-readable), so a seeded editor AND viewer
    # reach it. (The config env + settings-schema reads are action=secret — admin-only bulk
    # reads pinned in the secret-read test above.)
    role_grants_module.reset_role_grants_cache()
    reset_route_index()
    await seed_default_roles()
    for role in ("editor", "viewer"):
        allowed, _cause = await _level_allows(role, "/api/hooks", "GET")
        assert allowed is True, f"GET /api/hooks should be a grantable read for {role}"


async def test_config_env_read_and_write_both_admin_only(mem: _MemStore):
    from tai42_skeleton.access_control import role_grants as role_grants_module
    from tai42_skeleton.access_control.role_gate import DenialCause, reset_route_index

    role_grants_module.reset_role_grants_cache()
    reset_route_index()
    await seed_default_roles()
    # Both routes on /api/config/env are admin-only for editor AND viewer: POST (write_env) is
    # action=fenced, GET (read_env) is action=secret — settings==env==one admin-owned store. They
    # are distinct routes on the same path, both a HARD_FENCE deny to a non-admin.
    for role in ("editor", "viewer"):
        for method in ("POST", "GET"):
            allowed, cause = await _level_allows(role, "/api/config/env", method)
            assert allowed is False, f"{method} /api/config/env must be admin-only for {role}"
            assert cause is DenialCause.HARD_FENCE


# -- the un-fenced capability-supply-chain / arbitrary-execution / hooks doors ------

# These write doors were editor/viewer-reachable before M20: the async tool-execution submit,
# the MCP-config rewrite, the sub-MCP mount/unmount, a run-any-tool schedule, the fleet MCP
# re-bind, the manifest-section persist, and the four hooks routes. Each is action=write, so a
# seeded editor (write on every grantable tag) reaches them.
_GRANTABLE_WRITE_DOORS = [
    ("/api/tool-runs", "POST"),
    ("/api/mcp-config", "POST"),
    ("/api/sub-mcp", "POST"),
    ("/api/sub-mcp/weather", "DELETE"),
    ("/api/schedules", "POST"),
    ("/api/mcp-status/gh/reload", "POST"),
    ("/api/tools/shout/extensions", "POST"),
    ("/api/hooks", "POST"),
    ("/api/hooks/orders-hook", "DELETE"),
    ("/api/hooks/topics/orders/verifier", "PUT"),
    ("/api/hooks/topics/orders/verifier", "DELETE"),
]


@pytest.mark.parametrize(("path", "method"), _GRANTABLE_WRITE_DOORS)
async def test_unfenced_write_doors_are_grantable_to_editor(mem: _MemStore, path, method):
    from tai42_skeleton.access_control import role_grants as role_grants_module
    from tai42_skeleton.access_control.role_gate import reset_route_index, resolve_route_meta

    role_grants_module.reset_role_grants_cache()
    reset_route_index()
    await seed_default_roles()
    meta = resolve_route_meta(path, method)
    assert meta is not None, f"{method} {path} did not resolve"
    assert meta.action == "write"  # no fence — a grantable write
    allowed, _cause = await _level_allows("editor", path, method)
    assert allowed is True, f"{method} {path} should be grantable to a seeded editor"


async def test_supply_chain_read_routes_stay_grantable(mem: _MemStore):
    from tai42_skeleton.access_control import role_grants as role_grants_module
    from tai42_skeleton.access_control.role_gate import reset_route_index

    # The read twins stay grantable too: a non-admin editor keeps reading runs, listing
    # sub-MCP mounts, listing schedules, reading MCP status, reading a tool's extensions, and
    # listing hooks + topic verifiers.
    role_grants_module.reset_role_grants_cache()
    reset_route_index()
    await seed_default_roles()
    for path, method in (
        ("/api/tool-runs/some-run-id", "GET"),  # get_run — a non-admin reads its own run
        ("/api/tool-runs", "GET"),  # list_tool_runs
        ("/api/sub-mcp", "GET"),  # list_sub_mcp
        ("/api/schedules", "GET"),  # list_schedules
        ("/api/mcp-status", "GET"),  # get_mcp_status
        ("/api/tools/shout/extensions", "GET"),  # get_tool_extensions
        ("/api/hooks", "GET"),  # list_hooks — now a grantable read
        ("/api/hooks/verifiers", "GET"),  # list_verifiers
    ):
        allowed, _cause = await _level_allows("editor", path, method)
        assert allowed is True, f"{method} {path} should be grantable via its read route"


async def test_supply_chain_tags_remain_grantable_feature_groups(mem: _MemStore):
    from tai42_skeleton.access_control.roles import grantable_feature_tags

    # The tags of the un-fenced routes are grantable feature groups the seeded editor/viewer
    # maps cover.
    grantable = grantable_feature_tags()
    assert {"tool-runs", "sub-mcp", "manifest", "schedules", "extensions", "hooks"} <= grantable


# -- seed idempotence --------------------------------------------------------


async def test_seed_creates_three_roles(mem: _MemStore):
    await seed_default_roles()
    names = {r["name"] for r in await role_store().list_roles()}
    assert names == {"admin", "editor", "viewer"}


async def test_reseed_does_not_overwrite_operator_edit(mem: _MemStore):
    await seed_default_roles()
    # An operator edits the editor template (a new active version).
    await mem.save_version(
        "role", "editor", {"scopes": ["*"], "condition": ".custom", "policy_data": {}, "description": "edited"}
    )
    await seed_default_roles()  # re-seed must not clobber it
    body = await role_store().get_active_body("editor")
    assert body["condition"] == ".custom"


# -- apply_role copy + upsert semantics --------------------------------------


async def test_apply_role_upserts_when_no_prior_policy(mem, pg: FakeAccessControlPg, redis_mgmt):
    await seed_default_roles()
    await apply_role("owner-bob", "admin")  # no prior policy row → create path
    body = pg.policy_body("owner-bob")
    assert body is not None
    assert body["scopes"] == ["*"]
    assert body["condition"] is None


async def test_apply_role_copies_and_does_not_retroapply(mem, pg: FakeAccessControlPg, redis_mgmt):
    await seed_default_roles()
    await apply_role("bob", "editor")
    assert pg.policy_body("bob")["condition"] == EDITOR_JQ
    # Editing the template afterwards does NOT change bob's already-applied policy.
    await mem.save_version(
        "role", "editor", {"scopes": ["*"], "condition": ".changed", "policy_data": {}, "description": "x"}
    )
    assert pg.policy_body("bob")["condition"] == EDITOR_JQ


async def test_apply_role_preserves_disabled_marker(mem, pg: FakeAccessControlPg, redis_mgmt):
    # A disabled (kill-switched) user that is then re-roled must KEEP the disabled marker:
    # apply_role writes only the scopes+condition dimension and never policy_data, so a
    # set_user_disabled followed by apply_role can never revive the killed keys (a
    # fail-OPEN on exactly the credentials the disable kills).
    await seed_default_roles()
    pg.add_policy("bob", scopes=["old"], policy_data={"disabled": True})
    await apply_role("bob", "editor")
    body = pg.policy_body("bob")
    assert body["policy_data"]["disabled"] is True  # the marker survived the re-role
    assert body["scopes"] == ["*"]  # scopes WERE updated
    assert body["condition"] == EDITOR_JQ  # condition WAS updated


async def test_apply_role_unknown_raises_keyerror(mem, pg: FakeAccessControlPg, redis_mgmt):
    await seed_default_roles()
    with pytest.raises(KeyError, match="unknown role"):
        await apply_role("bob", "nope")


async def test_apply_role_normalizes_condition_dimension(mem, pg: FakeAccessControlPg, redis_mgmt):
    await seed_default_roles()
    # A prior role leaves a condition; re-assigning admin (no condition) clears it.
    await apply_role("bob", "editor")
    assert pg.policy_body("bob")["condition"] == EDITOR_JQ
    await apply_role("bob", "admin")
    body = pg.policy_body("bob")
    assert body["condition"] is None
    assert body["condition_id"] is None


# -- SkeletonAccountsAdminServices -------------------------------------------


def test_services_satisfies_protocol():
    assert isinstance(SkeletonAccountsAdminServices(), AccountsAdminServices)


async def test_services_apply_role_bumps_version(mem, pg: FakeAccessControlPg, redis_mgmt):
    await seed_default_roles()
    services = SkeletonAccountsAdminServices()
    await services.apply_role("bob", "viewer")
    assert pg.policy_body("bob")["condition"] == VIEWER_JQ
    assert int(redis_mgmt._strings["ac:policy_version"]) >= 1


async def test_services_set_user_disabled_flips_marker(mem, pg: FakeAccessControlPg, redis_mgmt):
    pg.add_policy("bob", scopes=["a"])
    services = SkeletonAccountsAdminServices()
    await services.set_user_disabled("bob", True)
    assert pg.policy_body("bob")["policy_data"]["disabled"] is True
    await services.set_user_disabled("bob", False)
    assert "disabled" not in pg.policy_body("bob")["policy_data"]


async def test_services_set_user_disabled_unknown_user_raises(mem, pg: FakeAccessControlPg, redis_mgmt):
    services = SkeletonAccountsAdminServices()
    with pytest.raises(KeyError, match="unknown user"):
        await services.set_user_disabled("ghost", True)


async def test_services_remove_policy_revokes_owned_keys(mem, pg: FakeAccessControlPg, provider, redis_mgmt):
    # Bob owns an api key; removing Bob's policy revokes the owned key.
    await management.add_user_api_key("bob", "bob-user", [])
    await management.add_user_api_key("bob-key", "machine", [], owner_user_id="bob")
    services = SkeletonAccountsAdminServices()
    await services.remove_policy("bob")
    assert "bob-key" not in provider.identities  # owned key revoked
    assert pg.policy("bob") is None


async def test_services_remove_policy_unknown_user_raises(mem, pg: FakeAccessControlPg, provider, redis_mgmt):
    services = SkeletonAccountsAdminServices()
    with pytest.raises(KeyError, match="unknown user"):
        await services.remove_policy("ghost")


async def test_services_remove_policy_validator_only_deployment(mem, pg: FakeAccessControlPg, redis_mgmt, monkeypatch):
    # On a validator-only deployment (no mint-capable provider) there are no api-keys to
    # own, so remove_policy skips the owned-key walk (which requires a mint provider)
    # and still deletes the user's enforced policy instead of raising.
    class _Validator(IdentityProvider):
        async def validate_token(self, token: str) -> AuthIdentity | None:
            return None

    registry._REGISTRY["accounts"] = lambda _settings: _Validator()
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["accounts"]')
    reset_all_settings()
    try:
        await access_control_store().create_policy("bob", [])
        services = SkeletonAccountsAdminServices()
        await services.remove_policy("bob")
        assert pg.policy("bob") is None
    finally:
        registry._REGISTRY.pop("accounts", None)
        reset_all_settings()


# -- wildcard write-side special-case (store) --------------------------------


async def test_wildcard_scope_writes_cleanly_via_apply(mem, pg: FakeAccessControlPg, redis_mgmt):
    # The seeded admin role carries ["*"]; apply_role must write it despite "*" being
    # unrouted (the store's write-side wildcard special-case).
    await seed_default_roles()
    await apply_role("first-owner", "admin")
    assert pg.policy_body("first-owner")["scopes"] == ["*"]
