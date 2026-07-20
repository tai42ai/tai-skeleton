"""The backup router's auth boundary, pinned with access control ENABLED.

Every ``/api/backup/*`` door reads or writes a full-fidelity backup document that
can carry secret sections (env, access_control), so all three are AUTHED: a call
with no credential is rejected before the handler runs. This mounts the real
routes behind the real ``AuthAdapter`` stack and asserts the no-credential
rejection across every method/path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai_skeleton.routers.backup as router
from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

_PATH_PATTERNS = {
    r"/api/backup/sections": "backup-api",
    r"/api/backup/export": "backup-api",
    r"/api/backup/import": "backup-api",
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
    ac_fake = _AcFake({"backup-api": "backup-api-protected"})

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/backup/sections", router.list_sections, methods=["GET"]),
        Route("/api/backup/export", router.export_backup, methods=["POST"]),
        Route("/api/backup/import", router.import_backup, methods=["POST"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_sections_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/backup/sections").status_code in (401, 403)


def test_export_rejected_without_auth(boundary_client):
    assert boundary_client.post("/api/backup/export", json={"sections": []}).status_code in (401, 403)


def test_import_rejected_without_auth(boundary_client):
    assert boundary_client.post(
        "/api/backup/import", json={"document": {"version": 1, "sections": {}}, "sections": []}
    ).status_code in (401, 403)
