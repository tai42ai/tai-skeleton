"""The config router's auth boundary, pinned with access control ENABLED.

The ``/api/config/*`` doors read and mutate deployment config (env values,
secret marks, the settings-schema surface with resolved secret values), so they
are all AUTHED. This asserts the settings-schema read is rejected with no
credential — the handler never runs, so its config deps are irrelevant.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai_skeleton.routers.config as router
from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key. Every config door maps to one protected template.
_PATH_PATTERNS = {
    r"/api/config/settings-schema": "config-api",
    r"/api/config/env": "config-api",
}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


def _boundary_client(monkeypatch) -> TestClient:
    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake({"config-api": "config-api-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/config/settings-schema", router.read_settings_schema, methods=["GET"]),
        Route("/api/config/env", router.read_env, methods=["GET"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_settings_schema_rejected_without_auth(monkeypatch):
    client = _boundary_client(monkeypatch)
    assert client.get("/api/config/settings-schema").status_code in (401, 403)


def test_env_read_rejected_without_auth(monkeypatch):
    client = _boundary_client(monkeypatch)
    assert client.get("/api/config/env").status_code in (401, 403)
