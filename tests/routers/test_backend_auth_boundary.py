"""The backend/fleet router's auth boundary, pinned with access control ENABLED.

Every backend-identity + ``/api/fleet/*`` door exposes fleet topology or drives a
fleet op, so all are AUTHED. Each asserts an unauthenticated request is denied
before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai_skeleton.routers.backend as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/backend", router.backend_info, methods=["GET"]),
    Route("/api/fleet/workers", router.list_workers, methods=["GET"]),
    Route("/api/fleet/reload-config", router.reload_config, methods=["POST"]),
]
_STANCES = {
    r"/api/backend": AUTHED,
    r"/api/fleet/workers": AUTHED,
    r"/api/fleet/reload-config": AUTHED,
}


def test_info_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/backend").status_code in (401, 403)


def test_workers_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/fleet/workers").status_code in (401, 403)


def test_reload_config_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/fleet/reload-config", json={"targets": None}).status_code in (401, 403)
