"""The tool-extensions auth boundary, pinned with access control ENABLED.

Both ``/api/tools/{name}/extensions`` doors read the live manifest and mutate the
persisted one (attachment wiring), so both are AUTHED — an unauthenticated
request is denied before the handler runs. Mirrors the presets boundary harness.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.tool_extensions as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/tools/{name}/extensions", router.get_tool_extensions, methods=["GET"]),
    Route("/api/tools/{name}/extensions", router.set_tool_extensions, methods=["POST"]),
]
_STANCES = {r"/api/tools/[^/]+/extensions": AUTHED}


def test_get_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tools/shout/extensions").status_code in (401, 403)


def test_post_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/tools/shout/extensions", json={"combos": []}).status_code in (401, 403)
