"""The manifest router's auth boundary, pinned with access control ENABLED.

Every ``/api/manifest`` / ``/api/mcp-*`` door reads or mutates the live manifest
and MCP config (which can embed connector tokens in MCP client config), so all
are AUTHED. Each asserts an unauthenticated request is denied before the handler
runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.manifest as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/manifest", router.get_manifest, methods=["GET"]),
    Route("/api/mcp-config", router.set_mcp_config, methods=["POST"]),
    Route("/api/mcp-config/schema", router.get_mcp_config_schema, methods=["GET"]),
    Route("/api/mcp-status", router.get_mcp_status, methods=["GET"]),
    Route("/api/mcp-status/{title}/reload", router.reload_mcp, methods=["POST"]),
]
_STANCES = {
    r"/api/manifest": AUTHED,
    r"/api/mcp-config": AUTHED,
    r"/api/mcp-config/schema": AUTHED,
    r"/api/mcp-status": AUTHED,
    r"/api/mcp-status/[^/]+/reload": AUTHED,
}


def test_get_manifest_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/manifest").status_code in (401, 403)


def test_set_mcp_config_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/mcp-config", json={}).status_code in (401, 403)


def test_mcp_config_schema_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/mcp-config/schema").status_code in (401, 403)


def test_mcp_status_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/mcp-status").status_code in (401, 403)


def test_reload_mcp_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/mcp-status/some_title/reload").status_code in (401, 403)
