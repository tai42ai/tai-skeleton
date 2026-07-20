"""The health router's auth boundary, pinned with access control ENABLED.

``GET /health`` is a liveness probe with no user data — PUBLIC by design, so a
load balancer / orchestrator can hit it before any credential exists. This pins
that stance: the probe is reachable WITHOUT credentials, and a future accidental
auth-flip that starts denying it is caught here.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.health as router
from tests.routers._auth_boundary import PUBLIC, boundary_client

_ROUTES = [Route("/health", router.health_check, methods=["GET"])]
_STANCES = {r"/health": PUBLIC}


def test_health_reachable_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    resp = client.get("/health")
    assert resp.status_code == 200  # handler reached = public by design
    assert resp.text == "OK"
