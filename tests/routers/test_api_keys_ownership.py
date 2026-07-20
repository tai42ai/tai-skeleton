"""Owner-aware ownership rules on the api-keys router.

Drives the route handlers directly with the caller contextvar bound and the
management backends faked, covering the admin discriminator, the non-admin
create/edit/revoke/list matrix, the laundered-key pin, and the capabilities/roles
routes. The router modules are bound by ``tests/routers/conftest.py``.
"""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from tai42_contract.access_control import OWNER_USER_ID_CLAIM, registry
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity, IdentityProvider
from tai42_kit.settings import reset_all_settings

import tai42_skeleton.operations.api_keys as ops_api_keys
import tai42_skeleton.routers.api_keys as router
import tai42_skeleton.versioning as versioning_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx
from tests.access_control.test_policy_store import _MemStore


class _SpyProvider(ApiKeyIdentityProvider):
    def __init__(self) -> None:
        self.identities: dict[str, str] = {}
        self.provision_owners: dict[str, str | None] = {}

    async def validate_token(self, token: str):  # pragma: no cover - unused
        return None

    async def provision(self, user_id: str, description: str, *, owner_user_id: str | None = None) -> str:
        self.identities[user_id] = description
        self.provision_owners[user_id] = owner_user_id
        return f"sk-{user_id}"

    async def revoke(self, user_id: str) -> bool:
        return self.identities.pop(user_id, None) is not None

    async def update_description(self, user_id: str, description: str) -> bool:
        if user_id not in self.identities:
            return False
        self.identities[user_id] = description
        return True

    async def list_identities(self) -> list[tuple[str, str]]:
        return list(self.identities.items())


@pytest.fixture
def wired(monkeypatch):
    pg = FakeAccessControlPg()
    redis = FakeRedis(strings={})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    # ``_record_policy_version`` writes history through the versioned store — fake it.
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: _MemStore())
    spy = _SpyProvider()
    # Overwrite the "redis" provider with the spy, snapshotting the registry so the
    # mutation never leaks into another test (the module-global registry is shared).
    saved = dict(registry._REGISTRY)
    registry._REGISTRY["redis"] = lambda _s: spy
    try:
        yield pg, spy
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def _req(payload: dict | None = None, *, path_params: dict | None = None, method: str = "POST") -> Request:
    body = json.dumps(payload or {}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": "/api/auth/api-keys",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": path_params or {},
    }
    return Request(scope, receive=receive)


async def _call(handler, caller_id: str, req: Request):
    token = set_request_user_id(caller_id)
    try:
        return await handler(req)
    finally:
        reset_request_user_id(token)


def _body(resp):
    return json.loads(bytes(resp.body))


def _seed_caller(pg: FakeAccessControlPg, caller_id: str, *, scopes, condition=None, owner=None):
    policy_data = {OWNER_USER_ID_CLAIM: owner} if owner is not None else {}
    pg.add_policy(caller_id, scopes=scopes, condition=condition, policy_data=policy_data)


# -- admin discriminator -----------------------------------------------------


async def test_admin_creates_ownerless_key(wired):
    pg, spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    resp = await _call(router.create_api_key, "admin1", _req({"user_id": "k1", "description": "d", "scopes": []}))
    assert resp.status_code == 200
    assert spy.provision_owners["k1"] is None  # admin may mint ownerless


async def test_star_scope_with_condition_is_not_admin(wired):
    # A role-holder: ["*"] scopes PLUS a condition classifies NON-admin, so its explicit
    # other-owner create is forced/rejected.
    pg, _spy = wired
    _seed_caller(pg, "editor1", scopes=["*"], condition='.request.path == "/x"')
    resp = await _call(
        router.create_api_key,
        "editor1",
        _req({"user_id": "k1", "description": "d", "scopes": [], "owner_user_id": "someone-else"}),
    )
    assert resp.status_code == 403


async def test_laundered_owned_star_key_is_not_admin(wired):
    # A condition-free ["*"] key that is itself OWNED must still classify NON-admin (the
    # owner-claim conjunct), so it cannot revoke another user's key.
    pg, _spy = wired
    _seed_caller(pg, "laundered", scopes=["*"], owner="human")
    pg.add_policy("victim-key", scopes=["a"], policy_data={OWNER_USER_ID_CLAIM: "other"})
    resp = await _call(router.revoke_api_key, "laundered", _req(path_params={"user_id": "victim-key"}, method="DELETE"))
    assert resp.status_code == 403


# -- non-admin create --------------------------------------------------------


async def test_non_admin_create_forces_self_owner(wired):
    pg, spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    # Empty scopes keep the mint's route-validation out of scope; the pin is the forced
    # self-ownership, not scope validity.
    resp = await _call(router.create_api_key, "alice", _req({"user_id": "k1", "description": "d", "scopes": []}))
    assert resp.status_code == 200
    assert spy.provision_owners["k1"] == "alice"


async def test_non_admin_create_rejects_other_owner(wired):
    pg, _spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    resp = await _call(
        router.create_api_key,
        "alice",
        _req({"user_id": "k1", "description": "d", "scopes": ["read"], "owner_user_id": "bob"}),
    )
    assert resp.status_code == 403


async def test_non_admin_create_rejects_superset_scopes(wired):
    pg, _spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    resp = await _call(
        router.create_api_key, "alice", _req({"user_id": "k1", "description": "d", "scopes": ["read", "write"]})
    )
    assert resp.status_code == 400


async def test_owned_key_cannot_mint(wired):
    pg, _spy = wired
    _seed_caller(pg, "ownedcaller", scopes=["read"], owner="human")
    resp = await _call(router.create_api_key, "ownedcaller", _req({"user_id": "k1", "description": "d", "scopes": []}))
    assert resp.status_code == 403


# -- non-admin edit / revoke -------------------------------------------------


async def test_non_admin_revoke_own_key(wired):
    pg, _spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    # A key alice owns.
    await management.add_user_api_key("k1", "d", [], owner_user_id="alice")
    resp = await _call(router.revoke_api_key, "alice", _req(path_params={"user_id": "k1"}, method="DELETE"))
    assert resp.status_code == 200


async def test_non_admin_revoke_others_key_forbidden(wired):
    pg, _spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    pg.add_policy("k1", scopes=["a"], policy_data={OWNER_USER_ID_CLAIM: "bob"})
    resp = await _call(router.revoke_api_key, "alice", _req(path_params={"user_id": "k1"}, method="DELETE"))
    assert resp.status_code == 403


async def test_owner_claim_immutable_on_edit(wired):
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    pg.add_policy("k1", scopes=["a"], policy_data={OWNER_USER_ID_CLAIM: "bob"})
    # Changing the owner claim is rejected even for an admin.
    resp = await _call(
        router.edit_api_key,
        "admin1",
        _req({"policy_data": {OWNER_USER_ID_CLAIM: "carol"}}, path_params={"user_id": "k1"}, method="PUT"),
    )
    assert resp.status_code == 403


async def test_owner_claim_echo_accepted_on_edit(wired):
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    await management.add_user_api_key("k1", "d", [], owner_user_id="bob")
    # Echoing the same owner claim back is accepted.
    resp = await _call(
        router.edit_api_key,
        "admin1",
        _req(
            {"description": "d2", "policy_data": {OWNER_USER_ID_CLAIM: "bob"}},
            path_params={"user_id": "k1"},
            method="PUT",
        ),
    )
    assert resp.status_code == 200


# -- listing filter ----------------------------------------------------------


async def test_non_admin_listing_shows_only_own(wired):
    pg, _spy = wired
    _seed_caller(pg, "alice", scopes=["read"])
    await management.add_user_api_key("mine", "d", [], owner_user_id="alice")
    await management.add_user_api_key("theirs", "d", [], owner_user_id="bob")
    resp = await _call(router.list_tokens_payload, "alice", _req(method="GET"))
    users = {entry["user_id"] for entry in _body(resp)["data"]}
    assert users == {"mine"}


async def test_admin_listing_shows_all(wired):
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    await management.add_user_api_key("mine", "d", [], owner_user_id="alice")
    await management.add_user_api_key("theirs", "d", [], owner_user_id="bob")
    resp = await _call(router.list_tokens_payload, "admin1", _req(method="GET"))
    users = {entry["user_id"] for entry in _body(resp)["data"]}
    assert {"mine", "theirs"} <= users


async def test_validator_only_listing_returns_empty(wired, monkeypatch):
    # A validator-only deployment (no mint-capable provider) has no api-keys to list, so
    # the route returns {"data": []} rather than resolving a mint provider it does not
    # have. ``wired`` registers a mintable ``redis`` provider, so this test re-points the
    # chain at a validator-only provider and restores the settings in a finally
    # (``wired`` restores the registry).
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])

    class _Validator(IdentityProvider):
        async def validate_token(self, token: str) -> AuthIdentity | None:
            return None

    registry._REGISTRY["validator"] = lambda _s: _Validator()
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["validator"]')
    reset_all_settings()
    try:
        resp = await _call(router.list_tokens_payload, "admin1", _req(method="GET"))
        assert resp.status_code == 200
        assert _body(resp)["data"] == []
    finally:
        reset_all_settings()


# -- capabilities + roles ----------------------------------------------------


async def test_capabilities_reports_mintable(wired):
    _pg, _spy = wired
    resp = await router.get_capabilities(_req(method="GET"))
    data = _body(resp)["data"]
    assert data["mintable"] is True
    assert data["providers"] == [{"name": "redis", "mintable": True}]


async def test_list_roles_returns_seeded_roles(wired, monkeypatch):
    from tai42_skeleton.access_control.roles import seed_default_roles

    # A shared store so the seed survives the handler's own ``role_store()`` rebuild.
    mem = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
    await seed_default_roles()
    resp = await router.list_roles(_req(method="GET"))
    assert resp.status_code == 200
    data = _body(resp)["data"]
    assert {r["name"] for r in data} == {"admin", "editor", "viewer"}
    for role in data:
        assert set(role) == {"name", "scopes", "condition", "description"}


async def test_list_roles_empty_on_store_less_deployment(wired, monkeypatch):
    # A store-less deployment (no versioned store) has no role templates — the route
    # serves an empty list rather than attempting a Postgres read. The versioned store
    # is patched to RAISE if invoked, so the guard is load-bearing: without it the
    # handler would read the store and blow up rather than return an empty list.
    def _boom():
        raise AssertionError("versioned store must not be read when store-less")

    monkeypatch.setattr(versioning_module, "versioned_store", _boom)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: False)
    resp = await router.list_roles(_req(method="GET"))
    assert resp.status_code == 200
    assert _body(resp)["data"] == []


# -- policy-administration routes are admin-only -----------------------------
#
# ``EDITOR_JQ``/``VIEWER_JQ`` admit the whole ``/api/auth/api-keys`` subtree for own-key
# CRUD, but the policy version-history + rollback routes beneath it are enforced
# ADMIN-ONLY at the route level: a non-admin editor/viewer must never read another
# user's policy history (which leaks raw jq conditions) nor roll an enforced policy back.


async def test_list_policy_versions_forbidden_for_non_admin_editor(wired):
    pg, _spy = wired
    _seed_caller(pg, "editor1", scopes=["*"], condition='.request.path == "/x"')
    resp = await _call(router.list_policy_versions, "editor1", _req(path_params={"user_id": "victim"}, method="GET"))
    assert resp.status_code == 403


async def test_list_policy_versions_forbidden_for_non_admin_viewer(wired):
    pg, _spy = wired
    _seed_caller(pg, "viewer1", scopes=["*"], condition='.request.method == "GET"')
    resp = await _call(router.list_policy_versions, "viewer1", _req(path_params={"user_id": "victim"}, method="GET"))
    assert resp.status_code == 403


async def test_rollback_policy_forbidden_for_non_admin_editor(wired):
    pg, _spy = wired
    _seed_caller(pg, "editor1", scopes=["*"], condition='.request.path == "/x"')
    resp = await _call(
        router.rollback_policy, "editor1", _req({"version": 1}, path_params={"user_id": "victim"}, method="POST")
    )
    assert resp.status_code == 403


async def test_rollback_policy_forbidden_for_self_escalation(wired):
    # A non-admin may not roll back even its OWN policy (a self-escalation to a prior,
    # possibly more-privileged version).
    pg, _spy = wired
    _seed_caller(pg, "editor1", scopes=["*"], condition='.request.path == "/x"')
    resp = await _call(
        router.rollback_policy, "editor1", _req({"version": 1}, path_params={"user_id": "editor1"}, method="POST")
    )
    assert resp.status_code == 403


def _policy_body(scopes: list[str]) -> dict:
    return {
        "scopes": scopes,
        "policy_data": {},
        "condition": None,
        "condition_id": None,
        "condition_kwargs": None,
    }


async def test_list_policy_versions_admin_succeeds(wired, monkeypatch):
    pg, _spy = wired
    mem = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
    _seed_caller(pg, "admin1", scopes=["*"])
    await ops_api_keys.ac_policy_store().write("target", _policy_body(["a"]))
    resp = await _call(router.list_policy_versions, "admin1", _req(path_params={"user_id": "target"}, method="GET"))
    assert resp.status_code == 200
    assert len(_body(resp)["data"]) == 1


async def test_rollback_policy_admin_succeeds(wired, monkeypatch):
    pg, _spy = wired
    mem = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
    _seed_caller(pg, "admin1", scopes=["*"])
    # A live target policy plus two history versions; admin rolls back to version 1.
    pg.add_policy("target", scopes=["b"])
    store = ops_api_keys.ac_policy_store()
    await store.write("target", _policy_body(["a"]))
    await store.write("target", _policy_body(["b"]))
    resp = await _call(
        router.rollback_policy, "admin1", _req({"version": 1}, path_params={"user_id": "target"}, method="POST")
    )
    assert resp.status_code == 200
    assert _body(resp)["data"]["active_version"] == 1


# -- policy-history routes degrade gracefully on a store-less deployment ------
#
# A store-less AC deployment (no versioned store configured) keeps no policy version
# history, so both admin routes short-circuit (empty list / clean 404) rather than
# raw-500 on an absent Postgres backend. The versioned store is patched to RAISE if
# invoked, so the guard is load-bearing: without it the handler would read the store and
# blow up rather than return the store-less answer.


def _boom_store():
    raise AssertionError("versioned store must not be read when store-less")


async def test_list_policy_versions_admin_store_less_returns_empty(wired, monkeypatch):
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    monkeypatch.setattr(versioning_module, "versioned_store", _boom_store)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: False)
    resp = await _call(router.list_policy_versions, "admin1", _req(path_params={"user_id": "target"}, method="GET"))
    assert resp.status_code == 200
    assert _body(resp)["data"] == []


async def test_rollback_policy_admin_store_less_returns_404(wired, monkeypatch):
    pg, _spy = wired
    _seed_caller(pg, "admin1", scopes=["*"])
    monkeypatch.setattr(versioning_module, "versioned_store", _boom_store)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: False)
    resp = await _call(
        router.rollback_policy, "admin1", _req({"version": 1}, path_params={"user_id": "target"}, method="POST")
    )
    assert resp.status_code == 404
