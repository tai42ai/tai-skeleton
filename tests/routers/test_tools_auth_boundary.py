"""The tools router's auth boundary, pinned with access control ENABLED.

The four ``/api/tools*`` / ``/api/run-tool`` doors read the tool registry or
EXECUTE tools, so all are AUTHED. Each asserts an unauthenticated request is
denied before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.tools as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/tools", router.list_tools, methods=["GET"]),
    Route("/api/tools/tags", router.tool_tags, methods=["GET"]),
    Route("/api/tools/{tool_name}/schema", router.tool_schema, methods=["GET"]),
    Route("/api/tools-schema", router.tools_schema, methods=["GET"]),
    Route("/api/run-tool", router.run_tool, methods=["POST"]),
]
_STANCES = {
    r"/api/tools": AUTHED,
    r"/api/tools/tags": AUTHED,
    r"/api/tools/[^/]+/schema": AUTHED,
    r"/api/tools-schema": AUTHED,
    r"/api/run-tool": AUTHED,
}


def test_list_tools_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tools").status_code in (401, 403)


def test_tool_tags_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tools/tags").status_code in (401, 403)


def test_tool_schema_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tools/some_tool/schema").status_code in (401, 403)


def test_tools_schema_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tools-schema").status_code in (401, 403)


def test_run_tool_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/run-tool", json={"tool_name": "x", "arguments": {}}).status_code in (401, 403)
