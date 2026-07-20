"""The agents auth boundary, pinned with access control ENABLED.

A small ASGI app mounts the agents routes behind the real three-middleware stack
(Authentication -> AuthContext -> ResourceGuard). Every route is protected (no
public door on this surface), so an unauthenticated request to any is rejected
before it reaches the handler.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tai_skeleton.routers import agents as router
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key; tier 2 (Redis ``ac:route:``): template -> resource id.
_PATH_PATTERNS = {
    r"/api/agents": "agents",
    r"/api/agents/spec-runnable": "agents",
    r"/api/agents/[^/]+/runs": "agents",
    r"/api/agents/authored/[^/]+/runs": "agents",
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
    ac_fake = _AcFake({"agents": "agents-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/agents", router.list_agents, methods=["GET"]),
        Route("/api/agents/spec-runnable", router.list_spec_runnable_agents, methods=["GET"]),
        Route("/api/agents/{name}/runs", router.run_agent, methods=["POST"]),
        Route("/api/agents/authored/{name}/runs", router.run_authored_agent, methods=["POST"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_list_rejected_without_auth(boundary_client):
    resp = boundary_client.get("/api/agents")
    assert resp.status_code in (401, 403)


def test_spec_runnable_list_rejected_without_auth(boundary_client):
    resp = boundary_client.get("/api/agents/spec-runnable")
    assert resp.status_code in (401, 403)


def test_run_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/agents/faker/runs", json={"prompt": "hi"})
    assert resp.status_code in (401, 403)


def test_authored_run_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/agents/authored/faker/runs", json={"user_message": "hi"})
    assert resp.status_code in (401, 403)
