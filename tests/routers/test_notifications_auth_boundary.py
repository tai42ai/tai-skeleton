"""The notifications router's auth boundary, pinned with access control ENABLED.

``GET /api/notifications`` returns the deployment's internal notifications, so it
is AUTHED. This asserts the inbox read is rejected with no credential — the
handler never runs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai42_skeleton.routers.notifications as router
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

_PATH_PATTERNS = {r"/api/notifications": "notifications-api"}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


def _boundary_client(monkeypatch) -> TestClient:
    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake({"notifications-api": "notifications-api-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [Route("/api/notifications", router.list_notifications, methods=["GET"])]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_list_notifications_rejected_without_auth(monkeypatch):
    client = _boundary_client(monkeypatch)
    resp = client.get("/api/notifications")
    assert resp.status_code in (401, 403)
