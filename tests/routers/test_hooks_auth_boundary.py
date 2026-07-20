"""The hooks router's auth boundary, pinned with access control ENABLED.

The ``/api/hooks`` management doors are AUTHED (list/register/unregister carry
and mutate hook config); the ``/universal_webhook/{topic}`` ingress is PUBLIC
(external systems POST events to it with no Studio credential). This asserts the
split: a management call with no credential is rejected, while the ingress is
reachable unauthenticated.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import tai_skeleton.routers.hooks as router
from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key. Management doors map to one protected template;
# the public webhook ingress maps to another.
_PATH_PATTERNS = {
    r"/api/hooks": "hooks-api",
    r"/api/hooks/.+": "hooks-api",
    r"/universal_webhook/.+": "hooks-webhook",
}


class _AcFake:
    def __init__(self, strings: dict) -> None:
        self._strings = strings

    async def get(self, key):
        return self._strings.get(key)

    async def hgetall(self, key):
        return {}


class _Manager:
    async def list_hooks(self) -> dict:
        return {}

    async def register(self, params) -> bool:
        return True

    async def unregister(self, name: str) -> bool:
        return False

    async def on_event(self, topic: str, payload: dict) -> None:
        return None

    async def get_topic_verifier(self, topic: str) -> dict | None:
        return None

    async def all_topic_verifiers(self) -> dict:
        return {}

    async def set_topic_verifier(self, topic: str, binding: dict) -> None:
        return None

    async def delete_topic_verifier(self, topic: str) -> bool:
        return False


@pytest.fixture
def boundary_client(monkeypatch):
    manager = _Manager()
    monkeypatch.setattr(router, "get_hooks_manager", lambda: manager)

    async def _parse(request, include_query=True):
        return {"ok": True}

    monkeypatch.setattr(router, "parse_any_payload", _parse)

    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake(
        {
            "hooks-api": "hooks-api-protected",
            "hooks-webhook": ac_settings.public_resource_id,
        }
    )

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    routes = [
        Route("/api/hooks", router.list_hooks, methods=["GET"]),
        Route("/api/hooks", router.register_hook, methods=["POST"]),
        Route("/api/hooks/verifiers", router.list_verifiers, methods=["GET"]),
        Route("/api/hooks/{name}", router.unregister_hook, methods=["DELETE"]),
        Route("/api/hooks/topics/{topic}/verifier", router.set_topic_verifier, methods=["PUT"]),
        Route("/api/hooks/topics/{topic}/verifier", router.delete_topic_verifier, methods=["DELETE"]),
        Route("/universal_webhook/{topic}", router.universal_webhook, methods=["POST", "GET"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def test_list_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/hooks").status_code in (401, 403)


def test_register_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/hooks", json={"name": "a", "topic": "t", "tool": "notify"})
    assert resp.status_code in (401, 403)


def test_unregister_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/hooks/a").status_code in (401, 403)


def test_list_verifiers_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/hooks/verifiers").status_code in (401, 403)


def test_set_topic_verifier_rejected_without_auth(boundary_client):
    resp = boundary_client.put("/api/hooks/topics/orders/verifier", json={"verifier": "shared_secret", "config": {}})
    assert resp.status_code in (401, 403)


def test_delete_topic_verifier_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/hooks/topics/orders/verifier").status_code in (401, 403)


def test_webhook_ingress_reachable_unauthenticated(boundary_client):
    # The public ingress is reached (handler runs) with no credential.
    resp = boundary_client.get("/universal_webhook/orders")
    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted", "topic": "orders"}
