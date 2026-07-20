"""The metrics router's auth boundary, pinned with access control ENABLED.

``GET /metrics`` is the Prometheus scrape endpoint — infra, not the ``/api/*``
data surface — and carries no user data, so it is PUBLIC by design (scrapers hit
it without an app credential; the network is the gate). This pins that stance:
it is reachable WITHOUT credentials, and a future accidental auth-flip that
starts denying it is caught here.
"""

from __future__ import annotations

from starlette.routing import Route

import tai_skeleton.routers.metrics as router
from tests.routers._auth_boundary import PUBLIC, boundary_client

_ROUTES = [Route("/metrics", router.metrics_endpoint, methods=["GET"])]
_STANCES = {r"/metrics": PUBLIC}


def test_metrics_reachable_without_auth(monkeypatch, tmp_path):
    # The endpoint renders whatever the multiproc collector env points at; an
    # empty dir yields an empty exposition — enough to prove the handler ran.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    resp = client.get("/metrics")
    assert resp.status_code == 200  # handler reached = public by design
