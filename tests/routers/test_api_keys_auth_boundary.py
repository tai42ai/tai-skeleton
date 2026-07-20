"""The API-keys router's auth boundary, pinned with access control ENABLED.

Every ``/api/auth/*`` door provisions or reads credential material, so all of
them are AUTHED: a call with no credential is rejected before the handler runs.
This mounts the real routes behind the real ``AuthAdapter`` stack and asserts the
no-credential rejection across every method/path (including the route-order pair
``/scopes/urls`` vs ``/scopes/{scope_id}``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai_skeleton.routers.api_keys as router
from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: every /api/auth path resolves to one protected template.
_PATH_PATTERNS = {
    r"/api/auth/scopes": "auth-api",
    r"/api/auth/scopes/.+": "auth-api",
    r"/api/auth/routes": "auth-api",
    r"/api/auth/public-routes": "auth-api",
    r"/api/auth/tokens-payload": "auth-api",
    r"/api/auth/api-keys": "auth-api",
    r"/api/auth/api-keys/.+": "auth-api",
    r"/api/auth/claim-links": "auth-api",
    r"/api/auth/validate-condition": "auth-api",
}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


@pytest.fixture
def boundary_client(monkeypatch):
    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    # tier 2: the template maps to a PROTECTED resource id (not the public id).
    ac_fake = _AcFake({"auth-api": "auth-api-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/auth/scopes", router.list_scopes, methods=["GET"]),
        Route("/api/auth/scopes", router.add_scope_url, methods=["POST"]),
        # /scopes/urls registered BEFORE the /scopes/{scope_id} capture.
        Route("/api/auth/scopes/urls", router.remove_scope_url, methods=["DELETE"]),
        Route("/api/auth/scopes/{scope_id}", router.delete_scope, methods=["DELETE"]),
        Route("/api/auth/routes", router.list_routes, methods=["GET"]),
        Route("/api/auth/public-routes", router.list_public_routes, methods=["GET"]),
        Route("/api/auth/public-routes", router.pin_public_route, methods=["POST"]),
        Route("/api/auth/public-routes", router.unpin_public_route, methods=["DELETE"]),
        Route("/api/auth/me", router.get_me, methods=["GET"]),
        Route("/api/auth/tokens-payload", router.list_tokens_payload, methods=["GET"]),
        Route("/api/auth/api-keys", router.create_api_key, methods=["POST"]),
        Route("/api/auth/api-keys/{user_id}", router.edit_api_key, methods=["PUT"]),
        Route("/api/auth/api-keys/{user_id}", router.revoke_api_key, methods=["DELETE"]),
        Route("/api/auth/claim-links", router.create_claim_link, methods=["POST"]),
        Route("/api/auth/validate-condition", router.validate_condition, methods=["POST"]),
        Route("/api/auth/api-keys/{user_id}/policy/versions", router.list_policy_versions, methods=["GET"]),
        Route("/api/auth/api-keys/{user_id}/policy/rollback", router.rollback_policy, methods=["POST"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_list_scopes_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/auth/scopes").status_code in (401, 403)


def test_add_scope_rejected_without_auth(boundary_client):
    assert boundary_client.post("/api/auth/scopes", json={"scope_id": "s", "url": "/a"}).status_code in (401, 403)


def test_remove_scope_url_rejected_without_auth(boundary_client):
    assert boundary_client.request("DELETE", "/api/auth/scopes/urls", json={"url": "/a"}).status_code in (401, 403)


def test_delete_scope_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/auth/scopes/scope-a").status_code in (401, 403)


def test_routes_catalog_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/auth/routes").status_code in (401, 403)


def test_list_public_routes_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/auth/public-routes").status_code in (401, 403)


def test_pin_public_route_rejected_without_auth(boundary_client):
    assert boundary_client.post("/api/auth/public-routes", json={"url": "/x"}).status_code in (401, 403)


def test_unpin_public_route_rejected_without_auth(boundary_client):
    assert boundary_client.request("DELETE", "/api/auth/public-routes", json={"url": "/x"}).status_code in (401, 403)


def test_tokens_payload_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/auth/tokens-payload").status_code in (401, 403)


def test_me_rejected_without_auth(boundary_client):
    # The authenticated-always-allowed carve-out must NOT weaken the unauthenticated
    # stance: a no-credential request to /api/auth/me is challenged 401 by the guard.
    assert boundary_client.get("/api/auth/me").status_code in (401, 403)


def test_create_key_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/auth/api-keys", json={"user_id": "u", "description": "d", "scopes": []})
    assert resp.status_code in (401, 403)


def test_edit_key_rejected_without_auth(boundary_client):
    resp = boundary_client.put("/api/auth/api-keys/u", json={"description": "d", "scopes": []})
    assert resp.status_code in (401, 403)


def test_revoke_key_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/auth/api-keys/u").status_code in (401, 403)


def test_validate_condition_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/auth/validate-condition", json={"condition": ".a"})
    assert resp.status_code in (401, 403)


def test_create_claim_link_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/auth/claim-links", json={"api_key": "sk-x"})
    assert resp.status_code in (401, 403)


def test_policy_versions_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/auth/api-keys/u/policy/versions").status_code in (401, 403)


def test_policy_rollback_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/auth/api-keys/u/policy/rollback", json={"version": 1})
    assert resp.status_code in (401, 403)
