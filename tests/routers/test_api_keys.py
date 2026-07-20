"""The API-keys/scopes router driven against the Postgres policy store.

Covers the full scope + key CRUD round-trip, malformed-body and not-found
handling, the route-order pin (``/scopes/urls`` must not hit the ``{scope_id}``
capture), and the port's definition of done: a key minted through the create
route authenticates against the live ``RedisApiKeyProvider`` (its identity record
lives on Redis) and carries its scopes through the live ``PolicyEnforcer`` (which
reads the policy from the PG store); a revoked key stops authenticating within one
request; and a scope edit is visible to a warm policy cache the instant the version
is bumped, without waiting out the ttl.

Backends: the POLICY store is Postgres (the ``FakeAccessControlPg``); the identity
record, live context, and version counter are on Redis (the ``FakeRedis``); the
``ac_policy`` version HISTORY is the in-memory ``_MemStore``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient
from tai42_identity_redis import redis_api_key_provider as provider_module
from tai42_identity_redis.redis_api_key_provider import RedisApiKeyProvider

from tai42_skeleton.access_control import claim_links as claim_links_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.policy_store import AcPolicyStore
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.operations import api_keys as operations_api_keys
from tai42_skeleton.routers import api_keys

# Registered for its import side effect: the route-table probe (test_probe_request_app_is_route_bearing)
# needs every router's custom routes present on the built app, and a custom route registers only when
# its module is imported. Importing here (after the conftest binds ``tai42_app``) makes that probe
# deterministic in isolation instead of relying on another test module's collection order.
from tai42_skeleton.routers import hooks as _hooks  # noqa: F401
from tests.access_control.conftest import (
    FakeAccessControlPg,
    FakeRedis,
    _FakeApp,
    make_client_ctx,
    make_pg_ctx,
)
from tests.access_control.test_policy_store import _MemStore

S = access_control_settings()


@dataclass
class _Fakes:
    """The two backends the router drives: the Postgres policy store and the Redis
    that holds the identity record + live context + version counter."""

    pg: FakeAccessControlPg
    redis: FakeRedis


class _Req:
    """A structural stand-in for Starlette's ``Request`` the handlers read
    ``json()`` / ``path_params`` / ``query_params`` off of."""

    def __init__(
        self,
        *,
        body: object = None,
        path_params: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.path_params = path_params or {}
        self.query_params = query or {}

    async def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _req(**kwargs: Any) -> Request:
    return cast(Request, _Req(**kwargs))


def _body(response: Response) -> dict:
    return json.loads(bytes(response.body))


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Fakes:
    """Wire the two fakes behind the management module (which the router drives) AND
    the live provider/policy readers, so a key written by a route resolves back
    through the real enforcement code."""
    pg = FakeAccessControlPg()
    redis = FakeRedis(strings={}, hashes={})
    rctx = make_client_ctx(redis)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(management, "client_ctx", rctx)
    monkeypatch.setattr(provider_module, "client_ctx", rctx)
    monkeypatch.setattr(policy_module, "client_ctx", rctx)
    monkeypatch.setattr(verifier_module, "client_ctx", rctx)
    monkeypatch.setattr(claim_links_module, "client_ctx", rctx)
    return _Fakes(pg, redis)


# -- scope CRUD round-trip (all four scope routes) ---------------------------


async def test_scope_crud_round_trip(store: _Fakes) -> None:
    # add scope url
    resp = await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"scope_id": "scope-a", "url": "/a"}

    # a second url on the same scope
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/b"}))

    # list scopes
    resp = await api_keys.list_scopes(_req())
    assert _body(resp)["data"] == {"/a": "scope-a", "/b": "scope-a"}

    # remove one url (route-order pin exercised in its own test)
    resp = await api_keys.remove_scope_url(_req(body={"url": "/a"}))
    assert _body(resp)["data"] == {"url": "/a"}
    assert (await management.get_all_existing_scopes()) == {"/b": "scope-a"}

    # delete the scope
    resp = await api_keys.delete_scope(_req(path_params={"scope_id": "scope-a"}))
    assert resp.status_code == 200
    assert _body(resp)["data"]["scope_id"] == "scope-a"
    assert (await management.get_all_existing_scopes()) == {}


async def test_delete_unknown_scope_is_404(store: _Fakes) -> None:
    resp = await api_keys.delete_scope(_req(path_params={"scope_id": "ghost"}))
    assert resp.status_code == 404
    assert "not found" in _body(resp)["error"]


async def test_remove_scope_url_does_not_delete_scope(store: _Fakes) -> None:
    # Route-order regression pin: DELETE /scopes/urls must remove a URL and NOT
    # be swallowed by the /scopes/{scope_id} capture (which would delete a scope
    # named "urls"). Two urls on the scope; removing one keeps the scope alive.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/b"}))
    await api_keys.remove_scope_url(_req(body={"url": "/a"}))
    assert (await management.get_all_existing_scopes()) == {"/b": "scope-a"}


def test_route_order_urls_matches_before_scope_id(store: _Fakes) -> None:
    # The same pin at the Starlette dispatch layer: with the literal
    # ``/scopes/urls`` route registered BEFORE the ``/scopes/{scope_id}`` capture,
    # a DELETE to ``/scopes/urls`` reaches ``remove_scope_url`` (200, url-removal
    # body) — not ``delete_scope`` with ``scope_id="urls"`` (which would 404). ``/a``
    # is mapped first so the correct handler answers 200 (an unmapped url would itself
    # 404, blurring the discriminator). Seeded directly since this test is sync.
    store.pg.add_route("/a", "scope-a")
    app = Starlette(
        routes=[
            Route("/api/auth/scopes/urls", api_keys.remove_scope_url, methods=["DELETE"]),
            Route("/api/auth/scopes/{scope_id}", api_keys.delete_scope, methods=["DELETE"]),
        ]
    )
    resp = TestClient(app).request("DELETE", "/api/auth/scopes/urls", json={"url": "/a"})
    assert resp.status_code == 200
    assert resp.json()["data"] == {"url": "/a"}


# -- key CRUD ----------------------------------------------------------------


async def test_create_returns_raw_key_once(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    resp = await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["scope-a"]}))
    assert resp.status_code == 200
    raw = _body(resp)["data"]
    assert isinstance(raw, str)
    assert raw.startswith("sk-")


async def test_create_forwards_policy_data_and_condition_into_stored_policy(store: _Fakes) -> None:
    # The create route forwards the optional policy_data + condition/condition_id/
    # condition_kwargs through create_api_key into the stored policy record.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    resp = await api_keys.create_api_key(
        _req(
            body={
                "user_id": "u1",
                "description": "desc",
                "scopes": ["scope-a"],
                "policy_data": {"limit": 7},
                "condition": ".context.used < .policy.limit",
                "condition_id": "quota",
                "condition_kwargs": {"tier": "pro"},
            }
        )
    )
    assert resp.status_code == 200
    assert store.pg.policy_body("u1") == {
        "scopes": ["scope-a"],
        "policy_data": {"limit": 7},
        "condition": ".context.used < .policy.limit",
        "condition_id": "quota",
        "condition_kwargs": {"tier": "pro"},
    }


async def test_create_unknown_scope_is_400(store: _Fakes) -> None:
    resp = await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["ghost"]}))
    assert resp.status_code == 400
    assert "does not exist" in _body(resp)["error"]


# -- claim links -------------------------------------------------------------


async def _mint_key(store: _Fakes) -> str:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    resp = await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    return _body(resp)["data"]


async def test_create_claim_link_admin_round_trip(store: _Fakes) -> None:
    raw = await _mint_key(store)
    resp = await api_keys.create_claim_link(_req(body={"api_key": raw}))
    assert resp.status_code == 200
    data = _body(resp)["data"]
    assert data["token"].startswith("clm-")
    assert data["claim_path"] == f"/login#claim={data['token']}"
    assert "expires_at" in data
    # The response NEVER echoes the submitted raw key.
    assert raw not in json.dumps(data)


async def test_create_claim_link_invalid_key_is_400(store: _Fakes) -> None:
    resp = await api_keys.create_claim_link(_req(body={"api_key": "sk-not-real"}))
    assert resp.status_code == 400
    assert _body(resp)["error"] == "not a valid API key"


async def test_create_claim_link_non_owner_is_403(store: _Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    from tai42_contract.access_control.models import AccessPolicy

    from tai42_skeleton.operations.api_keys import _Caller

    raw = await _mint_key(store)

    async def _mallory() -> _Caller:
        # A non-admin caller who neither owns nor IS the resolved key.
        return _Caller(caller_id="mallory", policy=AccessPolicy(scopes=["read"]), is_admin=False, owner_claim=None)

    monkeypatch.setattr(operations_api_keys, "_resolve_caller", _mallory)
    resp = await api_keys.create_claim_link(_req(body={"api_key": raw}))
    assert resp.status_code == 403


async def test_create_claim_link_missing_api_key_is_400(store: _Fakes) -> None:
    resp = await api_keys.create_claim_link(_req(body={}))
    assert resp.status_code == 400


async def test_create_claim_link_forwards_ttl_seconds(store: _Fakes) -> None:
    # The extractor forwards a body ``ttl_seconds`` end-to-end at the route: the returned
    # expiry must reflect the REQUESTED lifetime, not the settings default.
    from datetime import UTC, datetime

    raw = await _mint_key(store)
    requested_ttl = 1234
    assert requested_ttl != S.claim_link_ttl_seconds  # distinct from the default lifetime
    before = datetime.now(UTC)
    resp = await api_keys.create_claim_link(_req(body={"api_key": raw, "ttl_seconds": requested_ttl}))
    assert resp.status_code == 200
    expires_at = datetime.fromisoformat(_body(resp)["data"]["expires_at"])
    delta = (expires_at - before).total_seconds()
    # Within a few seconds of now + requested_ttl (nowhere near the default lifetime).
    assert requested_ttl - 5 <= delta <= requested_ttl + 5


async def test_create_claim_link_over_ceiling_ttl_is_400(store: _Fakes) -> None:
    # An over-ceiling ttl_seconds reaches the store through the ROUTE and is a loud 400.
    raw = await _mint_key(store)
    over = S.claim_link_max_ttl_seconds + 1
    resp = await api_keys.create_claim_link(_req(body={"api_key": raw, "ttl_seconds": over}))
    assert resp.status_code == 400
    assert "exceeds the maximum" in _body(resp)["error"]


@pytest.mark.parametrize("ttl", [True, "30"])
async def test_create_claim_link_non_integer_ttl_is_400(store: _Fakes, ttl: Any) -> None:
    # ``ttl_seconds`` is guarded at the HTTP edge by ``_opt_int``: a bool (``true`` — an
    # ``int`` subclass, so it would otherwise coerce silently to 1) and a non-int string are
    # both loud 400s carrying an "integer" message, never a silently coerced lifetime. The
    # extractor rejects before the operation runs (so ``api_key`` need not be a live key),
    # mirroring the sibling optional-field guards' negative tests
    # (``_opt_str`` -> test_add_scope_url_non_string_pattern_400; ``_opt_dict`` ->
    # test_create_non_dict_policy_data_400).
    resp = await api_keys.create_claim_link(_req(body={"api_key": "sk-anything", "ttl_seconds": ttl}))
    assert resp.status_code == 400
    assert "integer" in _body(resp)["error"]


async def test_edit_and_revoke_key(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-b", "url": "/b"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))

    resp = await api_keys.edit_api_key(
        _req(path_params={"user_id": "u1"}, body={"description": "d2", "scopes": ["scope-b"]})
    )
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"user_id": "u1", "updated": True}
    assert store.pg.policy_body("u1")["scopes"] == ["scope-b"]

    resp = await api_keys.revoke_api_key(_req(path_params={"user_id": "u1"}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"user_id": "u1", "revoked": True}


async def test_edit_description_only_preserves_policy_and_condition(store: _Fakes) -> None:
    # The Studio Save footgun: the edit dialog sends only {description, scopes}.
    # A description-only edit must PRESERVE the key's policy_data + condition gate
    # rather than silently erasing them (a privilege escalation).
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(
        _req(
            body={
                "user_id": "u1",
                "description": "desc",
                "scopes": ["scope-a"],
                "policy_data": {"limit": 7},
                "condition": ".context.used < .policy.limit",
                "condition_id": "quota",
                "condition_kwargs": {"tier": "pro"},
            }
        )
    )

    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"description": "desc2"}))
    assert resp.status_code == 200
    assert store.pg.policy_body("u1") == {
        "scopes": ["scope-a"],
        "policy_data": {"limit": 7},
        "condition": ".context.used < .policy.limit",
        "condition_id": "quota",
        "condition_kwargs": {"tier": "pro"},
    }


async def test_edit_scopes_only_preserves_policy_and_condition(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-b", "url": "/b"}))
    await api_keys.create_api_key(
        _req(
            body={
                "user_id": "u1",
                "description": "desc",
                "scopes": ["scope-a"],
                "policy_data": {"limit": 7},
                "condition": "c",
            }
        )
    )

    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"scopes": ["scope-b"]}))
    assert resp.status_code == 200
    policy = store.pg.policy_body("u1")
    assert policy["scopes"] == ["scope-b"]
    assert policy["policy_data"] == {"limit": 7}
    assert policy["condition"] == "c"


async def test_edit_explicit_null_clears_policy_and_condition(store: _Fakes) -> None:
    # An explicit null in the body IS an intentional clear (distinct from omission).
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(
        _req(
            body={
                "user_id": "u1",
                "description": "desc",
                "scopes": ["scope-a"],
                "policy_data": {"limit": 7},
                "condition": "c",
            }
        )
    )

    resp = await api_keys.edit_api_key(
        _req(path_params={"user_id": "u1"}, body={"policy_data": None, "condition": None})
    )
    assert resp.status_code == 200
    policy = store.pg.policy_body("u1")
    assert policy["policy_data"] == {}
    assert policy["condition"] is None


async def test_edit_condition_only_leaves_policy_data_untouched(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(
        _req(
            body={
                "user_id": "u1",
                "description": "desc",
                "scopes": ["scope-a"],
                "policy_data": {"limit": 7},
            }
        )
    )

    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "x"}))
    assert resp.status_code == 200
    policy = store.pg.policy_body("u1")
    assert policy["condition"] == "x"
    assert policy["policy_data"] == {"limit": 7}


async def test_edit_unknown_user_is_404(store: _Fakes) -> None:
    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "ghost"}, body={"description": "d", "scopes": []}))
    assert resp.status_code == 404


async def test_revoke_unknown_user_is_404(store: _Fakes) -> None:
    resp = await api_keys.revoke_api_key(_req(path_params={"user_id": "ghost"}))
    assert resp.status_code == 404


# -- OIDC-subject identifiers (charset allowlist dropped for Postgres) --------


async def test_oidc_subject_user_id_round_trips_through_routes(store: _Fakes) -> None:
    # Postgres identities are parameterized column values, so a subject containing
    # ``:``/``@``/unicode works natively through the create + edit + revoke routes —
    # the Redis-era charset allowlist is gone.
    await api_keys.add_scope_url(_req(body={"scope_id": "team:read@x", "url": "/a"}))
    resp = await api_keys.create_api_key(
        _req(body={"user_id": "auth0|abc:123", "description": "d", "scopes": ["team:read@x"]})
    )
    assert resp.status_code == 200
    assert store.pg.policy_body("auth0|abc:123")["scopes"] == ["team:read@x"]


# -- malformed bodies --------------------------------------------------------


@pytest.mark.parametrize(
    ("handler", "kwargs"),
    [
        (api_keys.add_scope_url, {"body": {"scope_id": "s"}}),  # missing url
        (api_keys.add_scope_url, {"body": {"url": "/a"}}),  # missing scope_id
        (api_keys.remove_scope_url, {"body": {}}),  # missing url
        (api_keys.create_api_key, {"body": {"description": "d", "scopes": []}}),  # missing user_id
        (api_keys.create_api_key, {"body": {"user_id": "u", "scopes": []}}),  # missing description
        (api_keys.create_api_key, {"body": {"user_id": "u", "description": "d"}}),  # missing scopes
        (api_keys.create_api_key, {"body": [1, 2]}),  # non-object body
        (api_keys.create_api_key, {"body": ValueError("bad json")}),  # invalid JSON
    ],
)
async def test_malformed_body_is_400(store: _Fakes, handler: Any, kwargs: dict) -> None:
    resp = await handler(_req(**kwargs))
    assert resp.status_code == 400


# -- payload read never leaks key material -----------------------------------


async def test_tokens_payload_carries_no_key_material(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    raw = _body(
        await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["scope-a"]}))
    )["data"]

    resp = await api_keys.list_tokens_payload(_req())
    payload = _body(resp)["data"]
    assert payload[0]["user_id"] == "u1"
    # No raw key and no sha256 hash of it anywhere in the enumerated payload.
    from tai42_kit.utils.data.string_util import hash_api_key

    blob = json.dumps(payload)
    assert raw not in blob
    assert hash_api_key(raw) not in blob


async def test_no_stored_value_contains_the_raw_key(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    raw = _body(
        await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["scope-a"]}))
    )["data"]
    # Neither the PG policy store nor the Redis identity/context store holds the raw key.
    blob = json.dumps(store.pg.policies) + repr(store.redis._hashes) + repr(store.redis._strings)
    assert raw not in blob


# -- the port's definition of done -------------------------------------------


async def test_created_key_authenticates_and_carries_scopes(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    raw = _body(
        await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["scope-a"]}))
    )["data"]

    # Present the raw key to the live provider — identity resolves (Redis record).
    identity = await RedisApiKeyProvider(S).validate_token(raw)
    assert identity is not None
    assert identity.user_id == "u1"

    # The live enforcer returns the key's scopes (read from the PG policy store).
    policy = await PolicyEnforcer(S).get_policy("u1")
    assert policy.scopes == ["scope-a"]


async def test_revocation_stops_authentication_next_request(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    raw = _body(
        await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "desc", "scopes": ["scope-a"]}))
    )["data"]
    provider = RedisApiKeyProvider(S)
    assert await provider.validate_token(raw) is not None

    await api_keys.revoke_api_key(_req(path_params={"user_id": "u1"}))
    # The identity record is uncached, so the next read fails to authenticate.
    assert await provider.validate_token(raw) is None


async def test_scope_edit_visible_to_warm_cache_after_version_bump(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-b", "url": "/b"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))

    enforcer = PolicyEnforcer(S)
    # Warm the per-worker cache at the current version.
    assert (await enforcer.get_policy("u1")).scopes == ["scope-a"]

    # Edit the policy in the store WITHOUT bumping the version: the warm cache
    # still serves the stale scopes (proves the cache is actually warm).
    await management.edit_user_payload("u1", "d", ["scope-a", "scope-b"])
    assert (await enforcer.get_policy("u1")).scopes == ["scope-a"]

    # Bump the version (what the mutating routes do) → cross-worker cache miss,
    # the edit is visible immediately without waiting out the ttl.
    await management.bump_policy_version()
    assert set((await enforcer.get_policy("u1")).scopes) == {"scope-a", "scope-b"}


# -- AC-policy versioning (store-first write-through + history + rollback) ----


@pytest.fixture(autouse=True)
def _admin_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the key routes' byte-for-byte ADMIN path (the ownership
    matrix is covered in ``test_api_keys_ownership``), so short-circuit the caller
    resolution to a condition-free ``"*"`` admin instead of binding a request contextvar
    and seeding a caller policy per test."""
    from tai42_contract.access_control.models import AccessPolicy

    from tai42_skeleton.operations.api_keys import _Caller

    async def _admin() -> _Caller:
        return _Caller(caller_id="test-admin", policy=AccessPolicy(scopes=["*"]), is_admin=True, owner_claim=None)

    monkeypatch.setattr(operations_api_keys, "_resolve_caller", _admin)


@pytest.fixture(autouse=True)
def pg_store(monkeypatch: pytest.MonkeyPatch) -> _MemStore:
    """A faithful in-memory generic store standing in for the durable PG version
    history, injected into the router's ``ac_policy_store`` factory so the
    store-first write-through orchestration is exercised end to end against it.

    Autouse so every key-create/edit route in this module records its policy version
    against the in-memory store instead of reaching for a real Postgres pool. The store
    is reported CONFIGURED so the policy-history routes take their store-backed path
    rather than the store-less short-circuit."""
    mem = _MemStore()
    monkeypatch.setattr(operations_api_keys, "ac_policy_store", lambda: AcPolicyStore(mem))
    monkeypatch.setattr("tai42_skeleton.versioning.versioned_store_configured", lambda: True)
    return mem


@pytest.fixture
def bound_app():
    """Bind a fake ``tai42_app`` exposing the template manager the validate route
    renders through, then restore the unbound state."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    tai42_app.bind(app)
    try:
        yield app
    finally:
        tai42_app.bind(None)


async def _seed_key(store: _Fakes, *, condition: str) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(
        _req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"], "condition": condition})
    )


async def test_first_policy_write_creates_version_one(store: _Fakes, pg_store: _MemStore) -> None:
    # The first policy write for a fresh user_id must ``create`` version 1 — a uniform
    # ``save_version`` would raise DocumentNotFoundError. No error surfaces (200).
    await _seed_key(store, condition="a")
    doc = pg_store.docs[("ac_policy", "u1")]
    assert doc["active"] == 1
    assert list(doc["versions"]) == [1]
    assert doc["versions"][1][0]["condition"] == "a"


async def test_policy_edit_is_store_first_then_history_and_bumps(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")
    version_before = int(store.redis._strings[S.policy_version_key])

    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))
    assert resp.status_code == 200

    # The enforced store (the authority) holds the new condition.
    assert store.pg.policy_body("u1")["condition"] == "b"
    # PG history appended v2 and advanced the active pointer.
    doc = pg_store.docs[("ac_policy", "u1")]
    assert doc["active"] == 2
    assert doc["versions"][2][0]["condition"] == "b"
    # The version key was bumped so sibling-worker enforcer caches invalidate.
    assert int(store.redis._strings[S.policy_version_key]) > version_before
    # Enforcement reads the store, seeing the new condition.
    assert (await PolicyEnforcer(S).get_policy("u1")).condition == "b"


async def test_description_only_edit_does_not_pollute_history(store: _Fakes, pg_store: _MemStore) -> None:
    # A description-only edit re-writes an identical policy record; history must not
    # gain a duplicate version.
    await _seed_key(store, condition="a")
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"description": "d2"}))
    assert list(pg_store.docs[("ac_policy", "u1")]["versions"]) == [1]


async def test_policy_version_history_lists_from_pg(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))

    resp = await api_keys.list_policy_versions(_req(path_params={"user_id": "u1"}))
    assert resp.status_code == 200
    versions = _body(resp)["data"]
    assert [v["version"] for v in versions] == [1, 2]
    assert [v["is_current"] for v in versions] == [False, True]
    assert versions[0]["body"]["condition"] == "a"
    assert versions[1]["body"]["condition"] == "b"


async def test_policy_version_history_absent_is_404(store: _Fakes, pg_store: _MemStore) -> None:
    resp = await api_keys.list_policy_versions(_req(path_params={"user_id": "ghost"}))
    assert resp.status_code == 404


async def test_policy_rollback_restores_store_first_and_bumps(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))  # v2 active
    version_before = int(store.redis._strings[S.policy_version_key])

    resp = await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": 1}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"user_id": "u1", "active_version": 1}

    # Target version body written back to the enforced store (the authority).
    assert store.pg.policy_body("u1")["condition"] == "a"
    # PG history pointer advanced to the rolled-back version.
    assert pg_store.docs[("ac_policy", "u1")]["active"] == 1
    # Version key bumped so enforcement follows on cache invalidation.
    assert int(store.redis._strings[S.policy_version_key]) > version_before
    assert (await PolicyEnforcer(S).get_policy("u1")).condition == "a"


async def test_route_edit_busts_a_warm_enforcer_cache(store: _Fakes, pg_store: _MemStore) -> None:
    # The generic warm-cache test drives ``bump_policy_version`` directly; this pins that
    # the EDIT ROUTE's OWN bump busts a warm sibling enforcer. Warm the SAME enforcer at v1,
    # mutate through the route, and it must return the new condition — no fresh enforcer, no
    # ttl wait — proving the route (not just a manual bump) invalidates a warm cache.
    await _seed_key(store, condition="a")
    enforcer = PolicyEnforcer(S)
    assert (await enforcer.get_policy("u1")).condition == "a"  # warm the per-worker cache

    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))
    assert resp.status_code == 200

    # The route's bump forced the warm cache to re-read the store (the authority) at once.
    assert (await enforcer.get_policy("u1")).condition == "b"


async def test_route_rollback_busts_a_warm_enforcer_cache(store: _Fakes, pg_store: _MemStore) -> None:
    # Same pin for the ROLLBACK route: warm at the post-edit version, roll back through the
    # route, and the SAME warm enforcer serves the rolled-back body immediately.
    await _seed_key(store, condition="a")  # v1
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))  # v2 active
    enforcer = PolicyEnforcer(S)
    assert (await enforcer.get_policy("u1")).condition == "b"  # warm at the current version

    resp = await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": 1}))
    assert resp.status_code == 200

    # The rollback route's bump busts the same warm cache — it now serves the prior version.
    assert (await enforcer.get_policy("u1")).condition == "a"


async def test_policy_rollback_absent_version_is_404(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")
    resp = await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": 99}))
    assert resp.status_code == 404


async def test_edit_store_failure_raises_history_untouched_key_intact(store: _Fakes, pg_store: _MemStore) -> None:
    # A forced enforced-store failure on an EDIT must raise (not be swallowed), leave the
    # history untouched (no version appended, pointer not advanced), and — because the
    # write ran inside a transaction — leave the stored policy intact.
    await _seed_key(store, condition="a")
    policy_before = store.pg.policy_body("u1")

    store.pg.fault = ("UPDATE access_control_policies SET scopes", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))

    assert list(pg_store.docs[("ac_policy", "u1")]["versions"]) == [1]
    assert pg_store.docs[("ac_policy", "u1")]["active"] == 1
    assert store.pg.policy_body("u1") == policy_before


async def test_rollback_store_failure_raises_pointer_not_advanced(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))  # v2 active
    policy_before = store.pg.policy_body("u1")

    store.pg.fault = ("UPDATE access_control_policies SET scopes", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": 1}))

    # Store-first order means the history pointer never advanced and the key is intact.
    assert pg_store.docs[("ac_policy", "u1")]["active"] == 2
    assert store.pg.policy_body("u1") == policy_before


async def test_edit_history_failure_still_bumps_and_raises(store: _Fakes, pg_store: _MemStore) -> None:
    # The cache-buster bump runs BEFORE the durable history append, so a history-store
    # outage during a policy tightening raises loudly YET still invalidates enforcer
    # caches: the authoritative body is enforced immediately, never a stale looser policy.
    await _seed_key(store, condition="a")  # v1 recorded
    version_before = int(store.redis._strings[S.policy_version_key])

    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("history down")

    # Fail the durable audit append (create-or-append → save_version on the 2nd write).
    pg_store.save_version = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="history down"):
        await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))

    # The enforced store holds the tightened condition, and the cache-buster WAS bumped
    # so every worker re-reads it at once — enforcement follows despite the failed audit.
    assert store.pg.policy_body("u1")["condition"] == "b"
    assert int(store.redis._strings[S.policy_version_key]) > version_before
    # The history pointer never advanced past v1 (the append never landed).
    assert pg_store.docs[("ac_policy", "u1")]["active"] == 1
    assert (await PolicyEnforcer(S).get_policy("u1")).condition == "b"


async def test_revoke_preserves_history_and_recreate_resumes(store: _Fakes, pg_store: _MemStore) -> None:
    await _seed_key(store, condition="a")  # v1
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))  # v2

    await api_keys.revoke_api_key(_req(path_params={"user_id": "u1"}))
    # Revoke clears the key record + enforced policy + context; the version HISTORY persists.
    assert list(pg_store.docs[("ac_policy", "u1")]["versions"]) == [1, 2]

    # Re-creating the same user_id RESUMES history — the next write is a save_version
    # (v3), never a fresh version 1.
    resp = await api_keys.create_api_key(
        _req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"], "condition": "c"})
    )
    assert resp.status_code == 200
    doc = pg_store.docs[("ac_policy", "u1")]
    assert sorted(doc["versions"]) == [1, 2, 3]
    assert doc["active"] == 3
    assert doc["versions"][3][0]["condition"] == "c"


async def test_scope_delete_cascade_records_policy_version(store: _Fakes, pg_store: _MemStore) -> None:
    # Deleting a scope cascades it out of a referencing key's enforced policy; that
    # rewrite must be recorded as a new PG version so ``is_current`` stays honest
    # against enforcement — otherwise a rollback to the (stale) current version would
    # silently re-grant the removed scope.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-b", "url": "/b"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a", "scope-b"]}))
    assert list(pg_store.docs[("ac_policy", "u1")]["versions"]) == [1]  # v1 from create

    resp = await api_keys.delete_scope(_req(path_params={"scope_id": "scope-a"}))
    assert resp.status_code == 200

    # The enforced store stripped scope-a from the policy.
    assert store.pg.policy_body("u1")["scopes"] == ["scope-b"]
    # A new PG version recorded the stripped body, so is_current now matches the store.
    doc = pg_store.docs[("ac_policy", "u1")]
    assert doc["active"] == 2
    assert doc["versions"][2][0]["scopes"] == ["scope-b"]


async def test_remove_url_cascade_records_policy_version(store: _Fakes, pg_store: _MemStore) -> None:
    # Removing a scope's LAST url cascades the same way; the rewrite is versioned too.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    assert list(pg_store.docs[("ac_policy", "u1")]["versions"]) == [1]

    resp = await api_keys.remove_scope_url(_req(body={"url": "/a"}))
    assert resp.status_code == 200
    assert store.pg.policy_body("u1")["scopes"] == []
    doc = pg_store.docs[("ac_policy", "u1")]
    assert doc["active"] == 2
    assert doc["versions"][2][0]["scopes"] == []


# -- validate-condition (fail-closed jq guard, never persists) ---------------


async def test_validate_condition_valid_ok(bound_app: Any) -> None:
    resp = await api_keys.validate_condition(_req(body={"condition": ".policy.limit"}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"ok": True, "result": None}


async def test_validate_condition_broken_is_400_with_message(bound_app: Any) -> None:
    resp = await api_keys.validate_condition(_req(body={"condition": ".("}))
    assert resp.status_code == 400
    # The jq compiler's own message is surfaced VERBATIM (not a generic placeholder), so
    # the author can see exactly what is wrong and fix the lock-out condition.
    error = _body(resp)["error"]
    assert "syntax error" in error
    assert "jq" in error


async def test_validate_condition_both_set_is_400(bound_app: Any) -> None:
    resp = await api_keys.validate_condition(_req(body={"condition": ".a", "condition_id": "x"}))
    assert resp.status_code == 400
    assert "not both" in _body(resp)["error"]


async def test_validate_condition_renders_named_template_by_id(bound_app: Any) -> None:
    # Named-template mode: the condition is authored by condition_id (+ kwargs) with NO
    # inline content. The route renders it via the template manager exactly as enforcement
    # (render_by_id_or_content — content=None, template_id + kwargs passed through), compiles
    # the rendered expression, and reports ok. Pins the template-id branch, not just inline.
    async def render_template(*, content: Any, template_id: Any, kwargs: Any) -> str:
        assert content is None
        assert template_id == "ac_quota"
        assert kwargs == {"limit": 5}
        return ".policy.limit > .context.used"

    bound_app.storage.resource_manager.render_by_id_or_content = render_template
    resp = await api_keys.validate_condition(_req(body={"condition_id": "ac_quota", "condition_kwargs": {"limit": 5}}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"ok": True, "result": None}


async def test_validate_condition_sample_eval_returns_boolean(bound_app: Any) -> None:
    resp = await api_keys.validate_condition(
        _req(
            body={
                "condition": ".policy.limit > .context.used",
                "sample_context": {"policy": {"limit": 5}, "context": {"used": 3}},
            }
        )
    )
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"ok": True, "result": True}


async def test_validate_condition_sample_eval_deny_returns_false(bound_app: Any) -> None:
    # The DENY direction: a sample that fails the condition returns result=False, mirroring
    # enforcement's allow-ONLY-on-exact-True rule. Pins the deny outcome (all other sample
    # tests assert True/None), so a regression that flipped the allow/deny result is caught.
    resp = await api_keys.validate_condition(
        _req(
            body={
                "condition": ".policy.limit > .context.used",
                "sample_context": {"policy": {"limit": 3}, "context": {"used": 5}},
            }
        )
    )
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"ok": True, "result": False}


async def test_validate_condition_sample_eval_truthy_non_bool_denies(bound_app: Any) -> None:
    # The load-bearing ``is True`` coercion: enforcement allows ONLY on exact ``True``, so a
    # truthy NON-boolean jq output (here a number) must surface as result=False, never leak
    # the raw value into the ``bool | null`` result. Dropping the ``is True`` coercion (bare
    # ``.first()``) would return the number and fail this — pinning the coercion.
    resp = await api_keys.validate_condition(
        _req(body={"condition": ".policy.limit", "sample_context": {"policy": {"limit": 7}}})
    )
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"ok": True, "result": False}


async def test_validate_condition_configured_but_renders_empty_is_400(bound_app: Any) -> None:
    # A configured condition that renders to an EMPTY string denies at enforcement
    # (fail-closed lock-out), so the guard must reject it loudly rather than report
    # ``ok`` and let a lock-out condition be saved.
    async def render_empty(*, content: Any, template_id: Any, kwargs: Any) -> str:
        return ""

    bound_app.storage.resource_manager.render_by_id_or_content = render_empty
    resp = await api_keys.validate_condition(_req(body={"condition": ".policy.limit"}))
    assert resp.status_code == 400
    assert "lock the key out" in _body(resp)["error"]


async def test_validate_condition_present_but_empty_string_is_400(bound_app: Any) -> None:
    # A PRESENT-but-empty condition (``""``) is "configured" at enforcement (which
    # tests ``is not None``) and denies as configured-but-empty. The guard mirrors
    # that exactly, so an empty-string condition is a loud 400, never a false ok — a
    # truthiness ``bool("")`` test would have wrongly green-lit this lock-out input.
    resp = await api_keys.validate_condition(_req(body={"condition": ""}))
    assert resp.status_code == 400
    assert "lock the key out" in _body(resp)["error"]


async def test_validate_condition_infra_error_propagates_as_500(bound_app: Any) -> None:
    # An infra fault while rendering (an unconfigured resource manager, a redis/storage
    # outage) is NOT an author error: ``validate_condition`` catches only the author
    # exception set ``(ValueError, ValidationError, TemplateError, TemplateNotFoundError)``
    # → 400 and lets everything else propagate → 500. Drive a plain ``RuntimeError``
    # through the render path and assert it propagates rather than being masked as a 400.
    async def _boom(*, content: Any, template_id: Any, kwargs: Any) -> str:
        raise RuntimeError("resource manager down")

    bound_app.storage.resource_manager.render_by_id_or_content = _boom
    with pytest.raises(RuntimeError, match="resource manager down"):
        await api_keys.validate_condition(_req(body={"condition": ".policy.limit"}))


async def test_validate_condition_never_persists(store: _Fakes, bound_app: Any) -> None:
    # Compiling/evaluating a condition must never write any store — a broken
    # condition can never reach enforcement from the validate path.
    await api_keys.validate_condition(_req(body={"condition": ".policy.limit"}))
    assert store.pg.policies == []
    assert store.pg.routes == []
    assert store.redis._hashes == {}
    assert store.redis._strings == {}


# -- optional-field type guards + edit/rollback error branches ---------------


async def test_add_scope_url_non_string_pattern_400(store: _Fakes) -> None:
    resp = await api_keys.add_scope_url(_req(body={"scope_id": "s", "url": "/a", "pattern": 123}))
    assert resp.status_code == 400
    assert "pattern" in _body(resp)["error"]


async def test_create_non_dict_policy_data_400(store: _Fakes) -> None:
    resp = await api_keys.create_api_key(
        _req(body={"user_id": "u1", "description": "d", "scopes": [], "policy_data": 123})
    )
    assert resp.status_code == 400
    assert "policy_data" in _body(resp)["error"]


async def test_edit_condition_id_and_kwargs_written(store: _Fakes, pg_store: _MemStore) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    resp = await api_keys.edit_api_key(
        _req(path_params={"user_id": "u1"}, body={"condition_id": "quota", "condition_kwargs": {"tier": "pro"}})
    )
    assert resp.status_code == 200
    policy = store.pg.policy_body("u1")
    assert policy["condition_id"] == "quota"
    assert policy["condition_kwargs"] == {"tier": "pro"}


async def test_edit_empty_description_is_400(store: _Fakes) -> None:
    # ``description`` is a required identity field: an empty value is a loud 400, so
    # it can be changed but never cleared.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"description": ""}))
    assert resp.status_code == 400
    assert "description" in _body(resp)["error"]


async def test_edit_unknown_scope_is_400(store: _Fakes) -> None:
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))
    await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    resp = await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"scopes": ["ghost"]}))
    assert resp.status_code == 400
    assert "does not exist" in _body(resp)["error"]


async def test_rollback_non_integer_version_400(store: _Fakes) -> None:
    resp = await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": "x"}))
    assert resp.status_code == 400
    assert "integer" in _body(resp)["error"]


async def test_rollback_no_live_key_is_404(store: _Fakes, pg_store: _MemStore) -> None:
    # The PG version history survives a revoke, but with no live key the store restore
    # returns None — a rollback surfaces a loud 404 rather than resurrecting a revoked key.
    await _seed_key(store, condition="a")  # v1
    await api_keys.edit_api_key(_req(path_params={"user_id": "u1"}, body={"condition": "b"}))  # v2
    await api_keys.revoke_api_key(_req(path_params={"user_id": "u1"}))
    resp = await api_keys.rollback_policy(_req(path_params={"user_id": "u1"}, body={"version": 1}))
    assert resp.status_code == 404
    assert "not found" in _body(resp)["error"]


async def test_record_policy_version_does_not_reread_store(
    store: _Fakes, pg_store: _MemStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The audit record is the exact body the mutation committed and returned — it is
    # NOT re-read from the store (which would let two concurrent A→B edits both read B
    # and drop A's version). Break ``get_policy_body`` loudly; create must still record.
    await api_keys.add_scope_url(_req(body={"scope_id": "scope-a", "url": "/a"}))

    async def _boom(_user_id: str) -> None:
        raise AssertionError("audit must not re-read the store via get_policy_body")

    monkeypatch.setattr(management, "get_policy_body", _boom)
    resp = await api_keys.create_api_key(_req(body={"user_id": "u1", "description": "d", "scopes": ["scope-a"]}))
    assert resp.status_code == 200
    assert pg_store.docs[("ac_policy", "u1")]["versions"][1][0]["scopes"] == ["scope-a"]


async def test_remove_unmapped_url_is_404(store: _Fakes) -> None:
    # A url that was never mapped must 404 (a typo'd unmap is not a silent success),
    # matching the surface's other not-found behavior.
    resp = await api_keys.remove_scope_url(_req(body={"url": "/never-mapped"}))
    assert resp.status_code == 404
    assert "not mapped" in _body(resp)["error"]


async def test_delete_public_marker_scope_is_400(store: _Fakes) -> None:
    resp = await api_keys.delete_scope(_req(path_params={"scope_id": S.public_resource_id}))
    assert resp.status_code == 400
    assert "public marker" in _body(resp)["error"]


async def test_validate_condition_malformed_body_400(bound_app: Any) -> None:
    resp = await api_keys.validate_condition(_req(body={"condition": 123}))
    assert resp.status_code == 400
    assert "condition" in _body(resp)["error"]


# -- route catalog (GET /api/auth/routes) ------------------------------------


async def test_list_routes_catalogs_route_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hand-built Starlette app with a GET route, a multi-method route, an unmapped
    # route, and a Mount. The catalog excludes the Mount, strips HEAD, sorts by path,
    # and reports each route's mapping (scope / public marker / null).
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    async def _h(request: Request) -> Response:
        return Response()

    marker = S.public_resource_id
    hand_built = Starlette(
        routes=[
            Route("/api/z", _h, methods=["GET"]),
            Route("/api/a", _h, methods=["POST", "GET"]),
            Route("/api/n", _h, methods=["GET"]),
            Mount("/app", app=Starlette()),
        ]
    )

    async def _mappings() -> dict[str, str]:
        return {"/api/z": "scope-x", "/api/a": marker}

    monkeypatch.setattr(management, "get_all_route_mappings", _mappings)
    resp = await api_keys.list_routes(cast(Request, SimpleNamespace(app=hand_built)))
    assert resp.status_code == 200
    data = _body(resp)["data"]
    # Sorted by path, Mount excluded (three Route entries, no /app).
    assert [e["path"] for e in data] == ["/api/a", "/api/n", "/api/z"]
    # HEAD stripped, multi-method sorted, mapped = public marker.
    assert data[0] == {"path": "/api/a", "methods": ["GET", "POST"], "mapped": marker}
    # The unassigned-routes bucket: no mapping → null.
    assert data[1] == {"path": "/api/n", "methods": ["GET"], "mapped": None}
    # A scope-mapped route reports its scope id.
    assert data[2] == {"path": "/api/z", "methods": ["GET"], "mapped": "scope-x"}


async def test_list_routes_raises_on_unclassified_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    # A route-table entry that is neither Route nor Mount must raise loudly — never a
    # silent skip (a new starlette routing type has to be classified here).
    class _Weird: ...

    async def _mappings() -> dict[str, str]:
        return {}

    monkeypatch.setattr(management, "get_all_route_mappings", _mappings)
    req = cast(Request, SimpleNamespace(app=SimpleNamespace(routes=[_Weird()])))
    with pytest.raises(TypeError, match="unclassified"):
        await api_keys.list_routes(req)


async def test_probe_request_app_is_route_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    # PROBE: at runtime, ``request.app`` inside a handler is the route-bearing Starlette
    # app, so ``request.app.routes`` is the live table ``list_routes`` reads. Starlette
    # sets ``scope["app"]`` to this app and ``HttpSurface.finalize``'s pure-ASGI
    # middleware wrappers do not replace it. The ``api_keys`` and ``hooks`` routers are
    # imported at module top, so their custom routes are registered regardless of run
    # order (this check is deterministic in isolation, not reliant on sibling collection).
    from starlette.routing import Mount

    from tai42_skeleton.app.instance import build_app

    star = build_app().http_app()
    routing_app = getattr(star, "mcp_lifespan_app", star)
    paths = {getattr(route, "path", None) for route in routing_app.routes}
    assert {"/api/auth/routes", "/api/auth/public-routes", "/api/hooks/verifiers"} <= paths
    assert any(isinstance(route, Mount) for route in routing_app.routes)

    # Drive ``list_routes`` against that real route-bearing app: the endpoint reads
    # ``request.app.routes``, so the catalog it returns IS the live table — proving it
    # reads the right object (the custom routes present, the sub-MCP Mount excluded),
    # not a hand-built stand-in like the other list_routes tests.
    async def _mappings() -> dict[str, str]:
        return {}

    monkeypatch.setattr(management, "get_all_route_mappings", _mappings)
    catalog = _body(await api_keys.list_routes(cast(Request, SimpleNamespace(app=routing_app))))["data"]
    catalog_paths = {entry["path"] for entry in catalog}
    assert {"/api/auth/routes", "/api/auth/public-routes", "/api/hooks/verifiers"} <= catalog_paths
    assert "/app" not in catalog_paths


# -- public route pins (/api/auth/public-routes) -----------------------------


async def test_public_route_pin_lifecycle(store: _Fakes) -> None:
    # POST pins a url public.
    resp = await api_keys.pin_public_route(_req(body={"url": "/open"}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"url": "/open"}
    # GET lists it, and it is a marker-valued route row — not a scope.
    assert _body(await api_keys.list_public_routes(_req()))["data"] == ["/open"]
    assert store.pg.route("/open")["scope_id"] == S.public_resource_id
    assert (await management.get_all_existing_scopes()) == {}
    # DELETE unpins it.
    resp = await api_keys.unpin_public_route(_req(body={"url": "/open"}))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"url": "/open"}
    assert _body(await api_keys.list_public_routes(_req()))["data"] == []


async def test_pin_public_route_with_pattern_registers_it(store: _Fakes) -> None:
    resp = await api_keys.pin_public_route(_req(body={"url": "/orders/{id}", "pattern": r"/orders/\d+"}))
    assert resp.status_code == 200
    assert await management.get_all_existing_patterns() == {"/orders/{id}": r"/orders/\d+"}


async def test_unpin_unpinned_route_is_404(store: _Fakes) -> None:
    resp = await api_keys.unpin_public_route(_req(body={"url": "/never"}))
    assert resp.status_code == 404
    # The offending url is echoed back verbatim, not just a generic message.
    assert _body(resp)["error"] == "url is not pinned public: '/never'"


async def test_pin_public_route_rejects_reserved_management_prefix(store: _Fakes) -> None:
    # The pin door must not be usable to de-authenticate the control plane: a url under
    # a reserved management prefix is a loud 400 and writes nothing (no version bump).
    resp = await api_keys.pin_public_route(_req(body={"url": "/api/auth/api-keys"}))
    assert resp.status_code == 400
    assert "reserved" in _body(resp)["error"]
    assert store.pg.routes == []
    assert S.policy_version_key not in store.redis._strings


async def test_add_scope_url_rejects_public_marker(store: _Fakes) -> None:
    resp = await api_keys.add_scope_url(_req(body={"scope_id": S.public_resource_id, "url": "/x"}))
    assert resp.status_code == 400
    assert "public marker" in _body(resp)["error"]
    # Nothing written to the scope machinery.
    assert store.pg.routes == []


async def test_pin_and_unpin_bump_policy_version(store: _Fakes) -> None:
    v0 = int(store.redis._strings.get(S.policy_version_key, 0))
    await api_keys.pin_public_route(_req(body={"url": "/open"}))
    v1 = int(store.redis._strings[S.policy_version_key])
    assert v1 > v0
    await api_keys.unpin_public_route(_req(body={"url": "/open"}))
    v2 = int(store.redis._strings[S.policy_version_key])
    assert v2 > v1


async def test_unpin_404_does_not_bump_version(store: _Fakes) -> None:
    # A failed unpin wrote nothing, so it must not bump the version.
    assert S.policy_version_key not in store.redis._strings
    resp = await api_keys.unpin_public_route(_req(body={"url": "/never"}))
    assert resp.status_code == 404
    assert S.policy_version_key not in store.redis._strings


async def test_pin_public_route_missing_url_400(store: _Fakes) -> None:
    resp = await api_keys.pin_public_route(_req(body={}))
    assert resp.status_code == 400
    assert "url" in _body(resp)["error"]


async def test_unpin_public_route_missing_url_400(store: _Fakes) -> None:
    resp = await api_keys.unpin_public_route(_req(body={}))
    assert resp.status_code == 400
    assert "url" in _body(resp)["error"]


# -- GET /api/auth/me (the capability projection route) ----------------------


def _authed_me_request(user_id: str, scopes: list[str], claims: dict) -> Request:
    """A Request carrying an authenticated ``TaiUser`` in scope, as the guard stack
    binds it — the ``/me`` extractor reads ``request.user`` / ``request.auth``."""
    from fastmcp.server.auth import AccessToken
    from starlette.authentication import AuthCredentials
    from starlette.requests import Request as StarletteRequest

    from tai42_skeleton.access_control.user import TaiUser

    user = TaiUser(AccessToken(token="t", client_id=user_id, scopes=list(scopes), claims=claims))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/auth/me",
        "headers": [],
        "query_string": b"",
        "user": user,
        "auth": AuthCredentials(list(scopes)),
    }
    return StarletteRequest(scope)


async def test_get_me_gate_off_returns_synthetic_total(store: _Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    # With the gate OFF the edge passes no identity, so the route answers a synthetic
    # TOTAL projection: admin, ["*"], the __no_auth__ identity, and empty list fields.
    from tai42_skeleton.access_control.settings import AccessControlSettings

    disabled = AccessControlSettings(enable=False, auth_providers=["redis"])
    monkeypatch.setattr(api_keys, "access_control_settings", lambda: disabled)
    resp = await api_keys.get_me(_req())
    assert resp.status_code == 200
    data = _body(resp)["data"]
    assert data["user_id"] == "__no_auth__"
    assert data["admin"] is True
    assert data["scopes"] == ["*"]
    assert data["routes"] == []
    assert data["route_patterns"] == []
    assert data["sub_mcp"] == []
    assert data["tools"] == []
    assert data["agents"] == []
    # ``mintable`` is DERIVED from the provider chain (redis mints), never hardcoded.
    assert data["mintable"] is True


async def test_get_me_gate_on_wraps_projection(store: _Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    # Gate ON: the extractor derives the caller's identity/scopes/claims from the authed
    # request and the operation wraps ``build_projection`` in the ``{"data": ...}``
    # envelope. The projection itself is exercised in tests/access_control/test_projection.
    from tai42_skeleton.access_control.projection import ProjectionResult

    captured: dict = {}

    async def fake_build(user_id: str, effective_scopes: list[str], claims: dict) -> ProjectionResult:
        captured["args"] = (user_id, list(effective_scopes), dict(claims))
        return ProjectionResult(
            user_id=user_id,
            owner_user_id=None,
            admin=False,
            scopes=list(effective_scopes),
            routes=[],
            route_patterns=[],
            sub_mcp=[],
            tools=[],
            agents=[],
            mintable=True,
        )

    monkeypatch.setattr(operations_api_keys, "build_projection", fake_build)
    resp = await api_keys.get_me(_authed_me_request("u1", ["read"], {"sub": "u1"}))
    assert resp.status_code == 200
    data = _body(resp)["data"]
    assert data["user_id"] == "u1"
    assert data["scopes"] == ["read"]
    # The edge passed the effective scopes + claims through verbatim (scopes are READ).
    assert captured["args"] == ("u1", ["read"], {"sub": "u1"})
