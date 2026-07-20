"""The public login-methods aggregator + the authed logout dispatcher.

Handler-level tests over fake accounts providers, plus a real-middleware-stack pin
that the always-public ``/api/login/methods`` is reachable unauthenticated while a
protected ``/api/auth`` route still 401s.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
from starlette.testclient import TestClient
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity
from tai42_contract.accounts import registry as accounts_registry
from tai42_contract.accounts.models import FormField, FormMethod, LoginMethod
from tai42_contract.accounts.provider import AccountsProvider

import tai42_skeleton.routers.api_keys as api_keys_router
import tai42_skeleton.routers.login as login_router
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings


class _FakeAccounts(AccountsProvider):
    """A configurable accounts provider fake for the aggregator/logout tests."""

    def __init__(
        self,
        *,
        methods: list[LoginMethod],
        bootstrap: bool = False,
        bootstrap_raises: bool = False,
        revocable: set[str] | None = None,
        revoke_raises: bool = False,
    ) -> None:
        self._methods = methods
        self._bootstrap = bootstrap
        self._bootstrap_raises = bootstrap_raises
        self._revocable = revocable or set()
        self._revoke_raises = revoke_raises
        self.revoke_calls: list[str] = []

    async def validate_token(self, token: str) -> AuthIdentity | None:  # pragma: no cover - unused
        return None

    def login_methods(self) -> list[LoginMethod]:
        return self._methods

    async def needs_bootstrap(self) -> bool:
        if self._bootstrap_raises:
            raise RuntimeError("bootstrap failed")
        return self._bootstrap

    async def revoke_session(self, token: str) -> bool:
        self.revoke_calls.append(token)
        if self._revoke_raises:
            raise RuntimeError("provider down")
        return token in self._revocable


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/api/auth/logout", "query_string": b"", "headers": raw})


def _form_method(mid: str) -> LoginMethod:
    return FormMethod(id=mid, title="Sign in", fields=[FormField(name="email", label="Email")], submit_path="/api/x")


@pytest.fixture(autouse=True)
def _clean_accounts_registry():
    saved = dict(accounts_registry._REGISTRY)
    accounts_registry._REGISTRY.clear()
    try:
        yield
    finally:
        accounts_registry._REGISTRY.clear()
        accounts_registry._REGISTRY.update(saved)


def _register(name: str, provider) -> None:
    accounts_registry._REGISTRY[name] = lambda _settings: provider


# -- methods aggregator ------------------------------------------------------


async def test_methods_empty_registry():
    resp = await login_router.login_methods(_request())

    assert json.loads(bytes(resp.body)) == {"data": {"methods": [], "bootstrap": False}}


async def test_methods_concatenated_and_bootstrap_or_ed():
    _register("a", _FakeAccounts(methods=[_form_method("m1")], bootstrap=False))
    _register("b", _FakeAccounts(methods=[_form_method("m2")], bootstrap=True))
    resp = await login_router.login_methods(_request())

    data = json.loads(bytes(resp.body))["data"]
    assert [m["id"] for m in data["methods"]] == ["m1", "m2"]
    assert data["bootstrap"] is True


async def test_methods_provider_error_propagates():
    _register("a", _FakeAccounts(methods=[], bootstrap_raises=True))
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        await login_router.login_methods(_request())


# -- logout dispatcher -------------------------------------------------------


async def test_logout_first_false_second_true_revokes():
    p1 = _FakeAccounts(methods=[], revocable=set())
    p2 = _FakeAccounts(methods=[], revocable={"tok"})
    _register("a", p1)
    _register("b", p2)

    resp = await login_router.logout(_request({"Authorization": "Bearer tok"}))
    assert json.loads(bytes(resp.body)) == {"data": {"revoked": True}}
    # Both consulted for the single candidate, registry order (a before b).
    assert p1.revoke_calls == ["tok"]
    assert p2.revoke_calls == ["tok"]


async def test_logout_iterates_all_candidates():
    # A stale Authorization value alongside a live X-Api-Key: the live one still logs out.
    p = _FakeAccounts(methods=[], revocable={"live"})
    _register("a", p)

    resp = await login_router.logout(_request({"Authorization": "Bearer stale", "X-Api-Key": "live"}))
    assert json.loads(bytes(resp.body)) == {"data": {"revoked": True}}
    assert p.revoke_calls == ["stale", "live"]


async def test_logout_all_false_is_404():
    _register("a", _FakeAccounts(methods=[], revocable=set()))
    resp = await login_router.logout(_request({"X-Api-Key": "sk-not-a-session"}))
    assert resp.status_code == 404


async def test_logout_no_provider_is_404():
    resp = await login_router.logout(_request({"X-Api-Key": "sk-x"}))
    assert resp.status_code == 404


async def test_logout_provider_error_propagates():
    _register("a", _FakeAccounts(methods=[], revoke_raises=True))
    with pytest.raises(RuntimeError, match="provider down"):
        await login_router.logout(_request({"Authorization": "Bearer tok"}))


# -- real middleware stack ---------------------------------------------------


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


def test_public_methods_reachable_while_protected_still_401(monkeypatch):
    settings = AccessControlSettings()
    ac_fake = _AcFake({})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)

    routes = [
        Route("/api/login/methods", login_router.login_methods, methods=["GET"]),
        Route("/api/auth/scopes", api_keys_router.list_scopes, methods=["GET"]),
    ]
    client = TestClient(Starlette(routes=routes, middleware=AuthAdapter(settings).get_middleware()))
    # Always-public: reachable with no credential (empty accounts registry).
    ok = client.get("/api/login/methods")
    assert ok.status_code == 200
    assert ok.json() == {"data": {"methods": [], "bootstrap": False}}
    # A reserved /api/auth route is still never public.
    assert client.get("/api/auth/scopes").status_code in (401, 403)


# -- public claim exchange ---------------------------------------------------


class _ClaimProvider(ApiKeyIdentityProvider):
    """Resolves the one stored key so the exchange re-validation passes; ``forget``
    models a revoke during the TTL window."""

    def __init__(self, valid: set[str]) -> None:
        self._valid = set(valid)

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return AuthIdentity(user_id="u1", claims={}) if token in self._valid else None

    async def provision(self, user_id, description, *, owner_user_id=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def revoke(self, user_id):  # pragma: no cover - unused
        raise NotImplementedError

    async def update_description(self, user_id, description):  # pragma: no cover - unused
        raise NotImplementedError

    async def list_identities(self):  # pragma: no cover - unused
        return []


def _claim_exchange_client(monkeypatch, *, seeded: dict[str, str], valid: set[str]) -> TestClient:
    import json as _json

    from tai42_contract.access_control import registry as id_registry
    from tai42_kit.utils.data.string_util import hash_api_key

    from tai42_skeleton.access_control import claim_links as claim_links_module
    from tests.access_control.conftest import FakeRedis, make_client_ctx

    settings = AccessControlSettings()
    strings = {
        f"{settings.claim_prefix}{hash_api_key(token)}": _json.dumps(
            {"api_key": api_key, "user_id": "u1", "created_by": "admin"}
        )
        for token, api_key in seeded.items()
    }
    fake = FakeRedis(strings=strings)
    monkeypatch.setattr(claim_links_module, "client_ctx", make_client_ctx(fake))
    monkeypatch.setitem(id_registry._REGISTRY, "redis", lambda _s: _ClaimProvider(valid))

    routes = [Route("/api/login/claim", login_router.exchange_claim_token, methods=["POST"])]
    return TestClient(Starlette(routes=routes, middleware=AuthAdapter(settings).get_middleware()))


def test_public_claim_exchange_reachable_and_single_use(monkeypatch):
    client = _claim_exchange_client(monkeypatch, seeded={"clm-tok": "sk-live"}, valid={"sk-live"})
    # Reachable with NO credential; returns the loginResult mirror {token, user_id}.
    resp = client.post("/api/login/claim", json={"token": "clm-tok"})
    assert resp.status_code == 200
    assert resp.json() == {"data": {"token": "sk-live", "user_id": "u1"}}
    # Single-use: the second exchange of the same token is the uniform 404.
    second = client.post("/api/login/claim", json={"token": "clm-tok"})
    assert second.status_code == 404
    assert second.json()["error"] == "unknown or already used claim token"


def test_claim_exchange_unknown_token_uniform_404(monkeypatch):
    client = _claim_exchange_client(monkeypatch, seeded={}, valid=set())
    resp = client.post("/api/login/claim", json={"token": "clm-never"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown or already used claim token"


def test_claim_exchange_ignores_garbage_credential_on_public_path(monkeypatch):
    # A stale/garbage credential in the headers must NOT 401 the public path (the
    # backend's always-public short-circuit): an unknown token still answers the uniform
    # 404, never a 401.
    client = _claim_exchange_client(monkeypatch, seeded={}, valid=set())
    resp = client.post(
        "/api/login/claim",
        json={"token": "clm-never"},
        headers={"x-api-key": "sk-garbage", "Authorization": "Bearer x"},
    )
    assert resp.status_code == 404


# -- registration + boot log -------------------------------------------------


def test_claim_route_registered_public():
    from tai42_skeleton.app.route_registry import load_api_routes

    by_pair = {(method, meta.path): meta for meta in load_api_routes() for method in meta.methods}
    meta = by_pair[("POST", "/api/login/claim")]
    assert meta.authed is False


async def test_claim_route_enumerated_in_boot_public_log(caplog):
    from tai42_skeleton.access_control.startup import check_always_public_routes

    with caplog.at_level("INFO"):
        await check_always_public_routes()
    assert "POST /api/login/claim" in caplog.text
