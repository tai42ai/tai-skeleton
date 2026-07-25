"""The schedules router's auth boundary, pinned with access control ENABLED.

Every ``/api/schedules*`` door is AUTHED — list/create/delete drive the
scheduling backend's tools and ``server-datetime`` reads the server clock. A
small ASGI app mounts the routes behind the real ``AuthAdapter`` stack; the
verifier's tier-1 mapping is seeded via ``AccessControlSettings(path_patterns=...)``
and tier 2 as an ``ac:route:`` entry pointing at a protected resource. The pin
asserts that a call with NO credential is rejected on every route.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.routers import schedules as router
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key; tier 2 (Redis ``ac:route:``): template -> resource id.
_PATH_PATTERNS = {
    r"/api/schedules": "schedules",
    r"/api/schedules/server-datetime": "schedules",
    r"/api/schedules/[^/]+": "schedules",
}


class _AcFake:
    """Minimal redis surface the verifier's route fetch uses: ``get`` over the
    ``ac:route:`` tier-2 map."""

    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


@pytest.fixture
def boundary_client(monkeypatch):
    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake({"schedules": "schedules-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/schedules", router.list_schedules, methods=["GET"]),
        Route("/api/schedules", router.create_schedule, methods=["POST"]),
        Route("/api/schedules/server-datetime", router.server_datetime, methods=["GET"]),
        Route("/api/schedules/{schedule_name}", router.delete_schedule, methods=["DELETE"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_list_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/schedules").status_code in (401, 403)


def test_create_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/schedules", json={"tool_name": "send_report"})
    assert resp.status_code in (401, 403)


def test_server_datetime_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/schedules/server-datetime").status_code in (401, 403)


def test_delete_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/schedules/nightly").status_code in (401, 403)
