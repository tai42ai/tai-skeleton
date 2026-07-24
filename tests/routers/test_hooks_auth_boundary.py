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
from tai42_identity_redis import redis_api_key_provider as provider_module
from tai42_kit.utils.data.string_util import hash_api_key

import tai42_skeleton.routers.hooks as router
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz import execution as execution_module
from tai42_skeleton.hooks.trigger_links import ResolvedTrigger
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, _FakeApp, make_client_ctx, make_pg_ctx
from tests.routers._auth_boundary import wire_store_from_route_strings

# The enforcer's alru cache is created in the test's loop and first used in the
# TestClient's portal loop — a benign loop-change reset that is a test artifact.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

# The credential the identity store admits in the credentialed client.
_VALID_KEY = "trigger-caller-key"

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


def _wire_router(monkeypatch) -> None:
    """Point the router's collaborators at in-memory stand-ins: the hooks manager, the
    payload parser, and the trigger resolver (``api-key-token`` is the link minted
    ``require_api_key``, every other token is a plain one)."""
    manager = _Manager()
    monkeypatch.setattr(router, "get_hooks_manager", lambda: manager)

    async def _parse(request, include_query=True):
        return {"ok": True}

    monkeypatch.setattr(router, "parse_any_payload", _parse)

    async def _resolve(token):
        return ResolvedTrigger(
            topic="orders",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            require_api_key=token == "api-key-token",
            tool_kwargs=None,
        )

    monkeypatch.setattr(router, "resolve_trigger_token", _resolve)
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=False))


def _routes() -> list[Route]:
    return [
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


@pytest.fixture
def boundary_client(monkeypatch):
    _wire_router(monkeypatch)

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

    app = Starlette(routes=_routes(), middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


@pytest.fixture
def bound_app():
    """A fake ``tai42_app`` so the enforcer can render the admitted key's (empty)
    policy condition."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    with tai42_app.bound(app):
        yield app


@pytest.fixture
def credentialed_client(monkeypatch, bound_app):
    """The same doors behind the same real stack, plus a credential the identity store
    ADMITS — the only way to drive the POSITIVE leg of the ``require_api_key`` door.

    ``_authenticated_caller`` is ``request.user.is_authenticated``, which is decided by
    ``AccessControlAuthBackend`` and by the resource guard that runs after it. A stub
    request user cannot see either: carving ``/trigger`` into
    ``always_public_path_prefixes``, for instance, makes the backend return an
    unauthenticated user for every credential, and every ``require_api_key`` link then
    answers 403 forever while the negative-leg assertions all still pass."""
    _wire_router(monkeypatch)

    ac_settings = AccessControlSettings(path_patterns=_PATH_PATTERNS)
    # The identity record the redis provider resolves the presented key to, plus the
    # policy row and the route rows the enforcement reads.
    fake = FakeRedis(
        hashes={f"{ac_settings.key_prefix}{hash_api_key(_VALID_KEY)}": {"user_id": "u1", "description": "d"}}
    )
    pg = FakeAccessControlPg()
    pg.add_route("hooks-api", "hooks-api-protected")
    pg.add_route("hooks-webhook", ac_settings.public_resource_id)
    pg.add_route("hooks-trigger", ac_settings.public_resource_id)
    # A non-admin holding the management doors' own resource scope, so any refusal is
    # the route's action class rather than the scope layer.
    pg.add_policy("u1", scopes=["hooks", "hooks-api-protected"])

    ctx = make_client_ctx(fake)
    monkeypatch.setattr(verifier_module, "client_ctx", ctx)
    monkeypatch.setattr(policy_module, "client_ctx", ctx)
    monkeypatch.setattr(provider_module, "client_ctx", ctx)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))

    app = Starlette(routes=_routes(), middleware=AuthAdapter(ac_settings).get_middleware())
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


def test_api_key_trigger_door_403s_without_a_credential(boundary_client):
    # A link minted ``require_api_key`` demands an authenticated principal on top of the
    # token; the 403 is reachable only by a token holder, and nothing is dispatched.
    resp = boundary_client.get("/trigger/api-key-token")
    assert resp.status_code == 403
    assert resp.json()["error"] == router._API_KEY_REQUIRED


def test_api_key_trigger_door_runs_with_an_admitted_credential(credentialed_client):
    # Positive leg through the REAL auth stack: the link that 403s a bare token holder
    # dispatches for a caller the identity store admits.
    resp = credentialed_client.get("/trigger/api-key-token", headers={"X-Api-Key": _VALID_KEY})
    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted"}


def test_unregister_hook_reachable_by_an_authenticated_non_admin(credentialed_client):
    # Control leg: an ordinary hooks WRITE door is reachable by this principal, so the
    # verifier refusals below are the fence, not the credential. The 404 is the
    # operation's own answer — the door ran.
    assert credentialed_client.delete("/api/hooks/some-hook", headers={"X-Api-Key": _VALID_KEY}).status_code == 404


def test_set_topic_verifier_refused_for_an_authenticated_non_admin(credentialed_client):
    # A verifier binding is the topic's only ingress lock and binding REPLACES it, so
    # the door is admin-only even for a principal that may register hooks.
    resp = credentialed_client.put(
        "/api/hooks/topics/orders/verifier",
        json={"verifier": "shared_secret", "config": {}},
        headers={"X-Api-Key": _VALID_KEY},
    )
    assert resp.status_code == 403


def test_delete_topic_verifier_refused_for_an_authenticated_non_admin(credentialed_client):
    # Unbinding reopens the topic's webhook to anyone, and its hooks then fire under
    # keys this caller could never pass the bind gate to delegate.
    resp = credentialed_client.delete("/api/hooks/topics/orders/verifier", headers={"X-Api-Key": _VALID_KEY})
    assert resp.status_code == 403


def test_topic_verifier_routes_are_fenced_at_every_grant_level():
    from tai42_skeleton.access_control.role_gate import (
        DenialCause,
        grant_map_admits,
        reset_route_index,
        resolve_route_meta,
    )

    reset_route_index()
    for method in ("PUT", "DELETE"):
        meta = resolve_route_meta("/api/hooks/topics/orders/verifier", method)
        assert meta is not None, f"{method} /api/hooks/topics/orders/verifier did not resolve"
        assert meta.action == "fenced"
        # No per-tag level opens a fence — ``hooks: write`` included.
        for level in ("none", "read", "write"):
            allowed, cause = grant_map_admits(meta, method, {"hooks": level})
            assert allowed is False
            assert cause is DenialCause.HARD_FENCE


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
