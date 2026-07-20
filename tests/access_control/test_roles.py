"""Role templates: the seeded jq through the REAL enforcer, seed idempotence,
``apply_role`` copy/upsert semantics, and the injected ``AccountsAdminServices``.
"""

from __future__ import annotations

import pytest
from tai_contract.access_control import registry
from tai_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity, IdentityProvider
from tai_contract.accounts import AccountsAdminServices
from tai_kit.settings import reset_all_settings
from tai_kit.utils.data import run_jq_first

import tai_skeleton.versioning as versioning_module
from tai_skeleton.access_control import management
from tai_skeleton.access_control.roles import (
    EDITOR_JQ,
    VIEWER_JQ,
    SkeletonAccountsAdminServices,
    apply_role,
    role_store,
    seed_default_roles,
)
from tai_skeleton.access_control.store import access_control_store

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


@pytest.mark.parametrize(
    ("path", "method", "allowed"),
    [
        ("/api/tools/run", "POST", True),  # outside /api/auth → allowed
        ("/api/run-tool", "POST", False),  # admin-only mutation fence — god-tool
        ("/api/tools/reload", "POST", False),  # admin-only mutation fence
        ("/api/tools/remove", "POST", False),  # admin-only mutation fence
        ("/api/config/reload", "POST", False),  # admin-only mutation fence
        ("/api/fleet/reload-config", "POST", False),  # fence — fleet soft-restart (recovery/ops)
        ("/api/fleet/workers", "GET", True),  # fleet census read — NOT fenced
        ("/api/manifest/replace", "POST", False),  # fence — whole-manifest replace
        ("/api/mcp-status/reload-failed", "POST", False),  # fence — bulk re-probe
        ("/api/mcp-status/gh/deregister", "POST", False),  # fence — per-server detach (shape match)
        ("/api/mcp-status/failed", "GET", True),  # read door — outside the fence
        ("/api/mcp-status/gh/reload", "POST", True),  # single-server reload — NOT fenced
        ("/api/auth/api-keys", "POST", True),  # own keys carve-out
        ("/api/auth/api-keys/x", "DELETE", True),  # own keys carve-out
        ("/api/auth/claim-links", "POST", True),  # one-time claim-link creation carve-in
        ("/api/auth/tokens-payload", "GET", True),
        ("/api/auth/capabilities", "GET", True),
        ("/api/auth/me", "GET", True),  # capability projection carve-in
        ("/api/auth/scopes", "GET", True),  # read-only scopes listing
        ("/api/auth/scopes", "POST", False),  # scope ADMINISTRATION stays admin-only
        ("/api/auth/public-routes", "POST", False),  # admin area fenced
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
        ("/api/tools/run", "POST", False),  # state-changing outside self-service → denied
        ("/api/run-tool", "POST", False),  # admin-only mutation fence
        ("/api/config/reload", "POST", False),  # admin-only mutation fence
        ("/api/fleet/reload-config", "POST", False),  # fence — fleet soft-restart (recovery/ops)
        ("/api/fleet/workers", "GET", True),  # fleet census read — NOT fenced
        ("/api/run-tool", "GET", False),  # fenced regardless of method (no such GET route exists)
        ("/api/manifest/replace", "POST", False),  # fence — whole-manifest replace
        ("/api/mcp-status/reload-failed", "POST", False),  # fence — bulk re-probe
        ("/api/mcp-status/gh/deregister", "GET", False),  # fence — shape match, fenced regardless of method
        ("/api/mcp-status/gh/reload", "GET", True),  # single-server reload NOT fenced — a read is allowed
        ("/api/mcp-status/failed", "GET", True),  # read door — outside the fence
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
        ("/api/auth/public-routes", "GET", False),  # admin area fenced even for reads
    ],
)
async def test_viewer_jq_matrix(path, method, allowed):
    assert await _allows(VIEWER_JQ, path, method) is allowed


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
