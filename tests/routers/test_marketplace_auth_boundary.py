"""The marketplace routes' auth boundary, pinned with access control ENABLED.

Every ``/api/marketplace/*`` route is AUTHED. This pins the stance for the three
representative doors — the install POST (an environment mutation), the search
GET (an outbound proxy), and the upgrade-all POST (the batch environment
mutation) — so an accidental auth-flip is caught: an unauthenticated request is
rejected before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.marketplace as router
from tests.routers._auth_boundary import AUTHED, boundary_client


def test_install_and_search_reject_without_auth(monkeypatch) -> None:
    routes = [
        Route("/api/marketplace/install", router.marketplace_install, methods=["POST"]),
        Route("/api/marketplace/search", router.marketplace_search, methods=["GET"]),
        Route("/api/marketplace/upgrade-all", router.marketplace_upgrade_all, methods=["POST"]),
    ]
    client = boundary_client(
        monkeypatch,
        routes,
        {
            "/api/marketplace/install": AUTHED,
            "/api/marketplace/search": AUTHED,
            "/api/marketplace/upgrade-all": AUTHED,
        },
    )
    assert client.post("/api/marketplace/install", json={"ref": "tai42/toolbox"}).status_code in (401, 403)
    assert client.get("/api/marketplace/search").status_code in (401, 403)
    assert client.post("/api/marketplace/upgrade-all", json={}).status_code in (401, 403)
