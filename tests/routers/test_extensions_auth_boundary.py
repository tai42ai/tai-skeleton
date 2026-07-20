"""The extensions router's auth boundary, pinned with access control ENABLED.

The single ``GET /api/extensions`` door lists the registered extension surface
for the UI's picker — AUTHED. Asserts an unauthenticated request is denied
before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai_skeleton.routers.extensions as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [Route("/api/extensions", router.list_extensions, methods=["GET"])]
_STANCES = {r"/api/extensions": AUTHED}


def test_list_extensions_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/extensions").status_code in (401, 403)
