"""The tool-runs router's auth boundary, pinned with access control ENABLED.

All three doors submit or read background tool runs, so all are AUTHED. Each
asserts an unauthenticated request is denied before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.tool_runs as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/tool-runs", router.submit_run, methods=["POST"]),
    Route("/api/tool-runs", router.list_tool_runs, methods=["GET"]),
    Route("/api/tool-runs/{run_id}", router.get_run, methods=["GET"]),
]
_STANCES = {
    r"/api/tool-runs": AUTHED,
    r"/api/tool-runs/[^/]+": AUTHED,
}


def test_submit_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/tool-runs", json={"tool_name": "x", "arguments": {}}).status_code in (401, 403)


def test_list_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tool-runs?tool_name=x").status_code in (401, 403)


def test_get_run_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tool-runs/some-run-id").status_code in (401, 403)
