"""The observability router's auth boundary, pinned with access control ENABLED.

Every ``/api/observability/*`` route is AUTHED — a trace embeds the full
input/output of a run's tool and model calls, so all five reads sit behind the
Studio credential. This asserts the whole surface: each route, called with no
credential, is rejected by the middleware before the handler runs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai42_skeleton.routers.observability as router
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key. Every observability door maps to one protected
# template — none is public.
_PATH_PATTERNS = {
    r"/api/observability/metrics": "observability",
    r"/api/observability/runs": "observability",
    r"/api/observability/runs/export": "observability",
    r"/api/observability/runs/.+/trace": "observability",
    r"/api/observability/runs/.+/trace/export": "observability",
}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


@pytest.fixture
def boundary_client(monkeypatch):
    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake({"observability": "observability-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/observability/metrics", router.get_metrics, methods=["GET"]),
        Route("/api/observability/runs", router.list_runs, methods=["GET"]),
        Route("/api/observability/runs/export", router.export_runs, methods=["GET"]),
        Route("/api/observability/runs/{trace_id}/trace", router.get_run_trace, methods=["GET"]),
        Route("/api/observability/runs/{trace_id}/trace/export", router.export_run_trace, methods=["GET"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_metrics_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/observability/metrics").status_code in (401, 403)


def test_runs_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/observability/runs").status_code in (401, 403)


def test_runs_export_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/observability/runs/export").status_code in (401, 403)


def test_trace_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/observability/runs/t1/trace").status_code in (401, 403)


def test_trace_export_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/observability/runs/t1/trace/export").status_code in (401, 403)
