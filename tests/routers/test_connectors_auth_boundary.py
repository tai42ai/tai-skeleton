"""The connectors router's auth boundary, pinned with access control ENABLED.

Every ``/api/connectors/*`` door lists providers/connections or starts, deletes,
reconnects, or reconfigures a connection (which can carry connector tokens), so
all are AUTHED. Each asserts an unauthenticated request is denied before the
handler runs. ``oauth/complete`` is included: it mutates connection state and is
authed like the rest (the OAuth provider redirect that reaches it carries the
session credential).
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.connectors as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/connectors/providers", router.providers, methods=["GET"]),
    Route("/api/connectors/connections", router.connections, methods=["GET"]),
    Route("/api/connectors/connections/{connection_id}", router.get_connection, methods=["GET"]),
    Route("/api/connectors/connections/start", router.start_connect, methods=["POST"]),
    Route("/api/connectors/connections/{connection_id}", router.disconnect, methods=["DELETE"]),
    Route("/api/connectors/connections/{connection_id}/reconnect", router.reconnect, methods=["POST"]),
    Route("/api/connectors/connections/{connection_id}/sub-services", router.patch_sub_services, methods=["PATCH"]),
    Route("/api/connectors/oauth/complete", router.oauth_complete, methods=["POST"]),
]
_STANCES = {
    r"/api/connectors/providers": AUTHED,
    r"/api/connectors/connections": AUTHED,
    r"/api/connectors/connections/start": AUTHED,
    r"/api/connectors/oauth/complete": AUTHED,
    r"/api/connectors/connections/[^/]+": AUTHED,
    r"/api/connectors/connections/[^/]+/reconnect": AUTHED,
    r"/api/connectors/connections/[^/]+/sub-services": AUTHED,
}


def test_providers_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/connectors/providers").status_code in (401, 403)


def test_connections_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/connectors/connections").status_code in (401, 403)


def test_get_connection_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/connectors/connections/abc").status_code in (401, 403)


def test_start_connect_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/connectors/connections/start", json={}).status_code in (401, 403)


def test_disconnect_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/connectors/connections/abc").status_code in (401, 403)


def test_reconnect_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/connectors/connections/abc/reconnect", json={}).status_code in (401, 403)


def test_patch_sub_services_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.patch("/api/connectors/connections/abc/sub-services", json={}).status_code in (401, 403)


def test_oauth_complete_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/connectors/oauth/complete", json={}).status_code in (401, 403)


# The REAL authed stance of each registered connectors route, keyed by
# (path, methods) — every connectors door is AUTHED. The tests below pin the
# actual ``@custom_route(authed=...)`` metadata the OpenAPI emitter and the auth
# middleware both consume, so a future ``authed=False`` slip on any door (e.g. a
# regression re-opening ``oauth/complete``) fails here instead of shipping a
# credential-less door — the hand-built ``_ROUTES``/``_STANCES`` app above cannot
# catch that on its own because it never reads the decorator's flag.
_REGISTERED_AUTHED = {
    ("/api/connectors/providers", ("GET",)): True,
    ("/api/connectors/connections", ("GET",)): True,
    ("/api/connectors/connections/start", ("POST",)): True,
    ("/api/connectors/connections/{connection_id}", ("DELETE",)): True,
    ("/api/connectors/connections/{connection_id}", ("GET",)): True,
    ("/api/connectors/connections/{connection_id}/reconnect", ("POST",)): True,
    ("/api/connectors/connections/{connection_id}/sub-services", ("PATCH",)): True,
    ("/api/connectors/oauth/complete", ("POST",)): True,
}


def test_registered_routes_match_declared_auth_stance():
    from tai42_skeleton.app.route_registry import load_api_routes

    actual = {
        (meta.path, meta.methods): meta.authed for meta in load_api_routes() if meta.path.startswith("/api/connectors/")
    }
    assert actual == _REGISTERED_AUTHED


def test_oauth_complete_registered_route_requires_auth():
    """The concrete ``oauth/complete`` registration — the one the running server
    actually mounts — carries authed=True, so the OAuth-bridge XHR that POSTs it
    must present the session credential."""
    from tai42_skeleton.app.route_registry import load_api_routes

    meta = next(m for m in load_api_routes() if m.path == "/api/connectors/oauth/complete" and "POST" in m.methods)
    assert meta.authed is True
