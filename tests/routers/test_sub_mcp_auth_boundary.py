"""The sub-MCP router's auth boundary, pinned with access control ENABLED.

The three ``/api/sub-mcp*`` doors list, register, and unregister sub-MCP apps
over the live sub-app router, so all are AUTHED. Each asserts an unauthenticated
request is denied before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.sub_mcp as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/sub-mcp", router.list_sub_mcp, methods=["GET"]),
    Route("/api/sub-mcp", router.register_sub_mcp, methods=["POST"]),
    Route("/api/sub-mcp/{slug}", router.unregister_sub_mcp, methods=["DELETE"]),
]
_STANCES = {
    r"/api/sub-mcp": AUTHED,
    r"/api/sub-mcp/[^/]+": AUTHED,
}


def test_list_sub_mcp_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/sub-mcp").status_code in (401, 403)


def test_register_sub_mcp_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/sub-mcp", json={"slug": "x", "tools": []}).status_code in (401, 403)


def test_unregister_sub_mcp_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/sub-mcp/some_slug").status_code in (401, 403)
