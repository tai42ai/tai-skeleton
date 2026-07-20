"""The resources router's auth boundary, pinned with access control ENABLED.

``POST /api/resources/get`` reads arbitrary stored resource content (an
arbitrary-file-read primitive), so it is AUTHED; an unauthenticated request is
denied before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai_skeleton.routers.resources as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/resources/get", router.get_resource_by_id, methods=["POST"]),
]
_STANCES = {
    r"/api/resources/get": AUTHED,
}


def test_get_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/resources/get", json={"resource_id": "x"}).status_code in (401, 403)
