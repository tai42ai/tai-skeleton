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

import tai42_skeleton.routers.hooks as router
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key. Management doors (hooks + trigger-links) map to one
# protected template; the public webhook ingress and trigger-link door map to public.
_PATH_PATTERNS = {
    r"/api/hooks": "hooks-api",
    r"/api/hooks/.+": "hooks-api",
    r"/universal_webhook/.+": "hooks-webhook",
    r"/trigger/.+": "hooks-trigger",
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

    async def on_event(self, topic: str, payload: dict, *, tool_kwargs_override: dict | None = None) -> None:
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

    async def _resolve(token):
        return "orders", None

    monkeypatch.setattr(router, "resolve_trigger_token", _resolve)

    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    ac_fake = _AcFake(
        {
            "hooks-api": "hooks-api-protected",
            "hooks-webhook": ac_settings.public_resource_id,
            "hooks-trigger": ac_settings.public_resource_id,
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
        Route("/api/hooks/trigger-links", router.create_trigger_link, methods=["POST"]),
        Route("/api/hooks/trigger-links", router.list_trigger_links, methods=["GET"]),
        Route("/api/hooks/trigger-links/{name}", router.delete_trigger_link, methods=["DELETE"]),
        Route("/universal_webhook/{topic}", router.universal_webhook, methods=["POST", "GET"]),
        Route("/trigger/{token}", router.trigger_link, methods=["POST", "GET"]),
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


def test_create_trigger_link_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/hooks/trigger-links", json={"topic": "t", "ttl_seconds": None})
    assert resp.status_code in (401, 403)


def test_list_trigger_links_rejected_without_auth(boundary_client):
    assert boundary_client.get("/api/hooks/trigger-links").status_code in (401, 403)


def test_delete_trigger_link_rejected_without_auth(boundary_client):
    assert boundary_client.delete("/api/hooks/trigger-links/some-name").status_code in (401, 403)


def test_trigger_door_reachable_unauthenticated(boundary_client):
    # The public trigger-link door is reached (handler runs) with no credential —
    # the route-table public stance, exactly like the webhook ingress door.
    resp = boundary_client.get("/trigger/some-token")
    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted"}


def test_trigger_door_validates_a_presented_credential(boundary_client):
    # The trigger door is route-table public (not always-public), so the auth layer
    # still validates any PRESENTED credential and 401s a garbage one — the correct,
    # consistent posture with universal_webhook (a presented credential is never
    # silently ignored on a scope-resolved public route).
    resp = boundary_client.get("/trigger/some-token", headers={"Authorization": "Bearer garbage"})
    assert resp.status_code in (401, 403)


# -- per-tag grant matrix through the real role gate -------------------------

_TRIGGER_ROUTES = [
    ("/api/hooks/trigger-links", "POST"),
    ("/api/hooks/trigger-links", "GET"),
    ("/api/hooks/trigger-links/some-name", "DELETE"),
]


def test_trigger_routes_grant_matrix():
    from tai42_skeleton.access_control.role_gate import grant_map_admits, reset_route_index, resolve_route_meta

    reset_route_index()
    for path, method in _TRIGGER_ROUTES:
        meta = resolve_route_meta(path, method)
        assert meta is not None, f"{method} {path} did not resolve"
        assert "hooks" in meta.tags
        # Grantable read/write — never the admin-only fence, so admin (all grants)
        # admits it and editors reach it via the hooks tag.
        assert meta.action in ("read", "write")

        write_ok, _ = grant_map_admits(meta, method, {"hooks": "write"})
        read_ok, _ = grant_map_admits(meta, method, {"hooks": "read"})
        none_ok, _ = grant_map_admits(meta, method, {"hooks": "none"})

        assert write_ok is True  # hooks:write reaches all three
        assert none_ok is False  # hooks:none is denied everywhere
        if method == "GET":
            assert read_ok is True  # hooks:read lists
        else:
            assert read_ok is False  # hooks:read cannot mint/revoke
