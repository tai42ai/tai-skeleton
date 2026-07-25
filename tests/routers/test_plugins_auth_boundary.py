"""The Studio-plugin auth boundary, pinned with access control ENABLED.

SPA + plugin-bundle assets are PUBLIC (the login screen must load before any
credential exists); the registry listing and every data route stay AUTHED. The
prefix-collision pin asserts that even a BROAD public SPA matcher that also
matches ``/api/plugins`` does NOT un-auth the registry listing — deny wins.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai42_skeleton.plugins.registry as reg
import tai42_skeleton.routers.plugins as router
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.plugins.registry import build_registry, set_current_registry
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key. The SPA matcher is deliberately BROAD (``/.*``) —
# it also matches ``/api/plugins``, so this doubles as the prefix-collision pin.
_PATH_PATTERNS = {
    r"/api/plugins": "studio-registry",
    r"/api/plugins/[^/]+/studio/.+": "studio-asset",
    r"/.*": "studio-spa",
}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


def _build_env(tmp_path: Path) -> tuple[Path, Path]:
    plugin_studio = tmp_path / "plugin" / "studio"
    plugin_studio.mkdir(parents=True)
    (plugin_studio / "index-a1b2c3.js").write_text("export const x=1;\n", encoding="utf-8")
    digest = reg._hash_file(plugin_studio / "index-a1b2c3.js")
    (plugin_studio / "studio-manifest.json").write_text(
        json.dumps(
            {
                "name": "acme_plugin",
                "version": "0.1.0",
                "api_version": 1,
                "entry": "index-a1b2c3.js",
                "integrity": {"index-a1b2c3.js": digest},
                "contributions": {},
            }
        ),
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    (dist / "vendor").mkdir(parents=True)
    for rel in reg.VENDOR_MODULES.values():
        (dist / rel).write_text("export {};\n", encoding="utf-8")
    (dist / "index.html").write_text("<head><!--tai-importmap--></head>", encoding="utf-8")
    return plugin_studio, dist


@pytest.fixture
def boundary_client(tmp_path, monkeypatch):
    plugin_studio, dist = _build_env(tmp_path)
    monkeypatch.setattr(reg, "_studio_root", lambda package: plugin_studio)
    prev = reg._current
    set_current_registry(build_registry(["acme_plugin"], str(dist)))
    monkeypatch.setattr(router, "plugins_settings", lambda: type("S", (), {"studio_dist_path": str(dist)})())

    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake(
        {
            "studio-registry": "studio-registry-protected",
            "studio-asset": ac_settings.public_resource_id,
            "studio-spa": ac_settings.public_resource_id,
        }
    )

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/plugins", router.list_studio_plugins, methods=["GET"]),
        Route("/api/plugins/{name}/studio/{path:path}", router.serve_studio_asset, methods=["GET"]),
        Route("/{spa_path:path}", router.serve_spa, methods=["GET"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    client = TestClient(app)
    yield client
    reg._current = prev


def test_asset_reachable_unauthenticated(boundary_client):
    resp = boundary_client.get("/api/plugins/acme_plugin/studio/index-a1b2c3.js")
    assert resp.status_code == 200  # handler reached = public


def test_spa_reachable_unauthenticated(boundary_client):
    resp = boundary_client.get("/tools")
    assert resp.status_code == 200  # index.html fallback, public


def test_registry_listing_rejected_without_auth(boundary_client):
    resp = boundary_client.get("/api/plugins")
    assert resp.status_code in (401, 403)


def test_broad_public_matcher_does_not_unauth_registry(boundary_client):
    # Even though the ``/.*`` public matcher also matches ``/api/plugins``, the
    # protected mapping wins (deny-wins accumulation) — the listing stays authed.
    resp = boundary_client.get("/api/plugins")
    assert resp.status_code in (401, 403)
