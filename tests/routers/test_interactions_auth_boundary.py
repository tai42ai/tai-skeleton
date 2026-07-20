"""The interactions auth boundary, pinned with access control ENABLED.

A small ASGI app mounts the interactions routes behind the real three-middleware
stack (Authentication -> AuthContext -> ResourceGuard). The verifier's tier-1
mapping is seeded via ``AccessControlSettings(path_patterns=...)`` (the mechanism
the deploy docs prescribe) and tier 2 as ``ac:route:`` entries. The pin needs no
authenticated identity: it asserts the two callback doors reach the handlers with
NO credentials, and that ``/stream`` and ``/answer`` are rejected without auth.
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
from tai_skeleton.interactions.settings import InteractionsSettings
from tai_skeleton.routers import interactions as router
from tests._fakes.interactions_redis import FakeRedis as InteractionsFake
from tests.routers._auth_boundary import wire_store_from_route_strings

# tier 1: path -> template key; tier 2 (Redis ``ac:route:``): template -> resource id.
_PATH_PATTERNS = {
    r"/api/interactions/callback/[^/]+": "interactions-callback",
    r"/api/interactions/stream": "interactions",
    r"/api/interactions/[^/]+/answer": "interactions",
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
    ac_fake = _AcFake(
        {
            "interactions-callback": ac_settings.public_resource_id,
            "interactions": "interactions-protected",
        }
    )

    @asynccontextmanager
    async def ac_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield ac_fake

    monkeypatch.setattr(verifier_module, "client_ctx", ac_ctx)
    wire_store_from_route_strings(monkeypatch, ac_fake._strings)

    # The callback handlers must not reach real Redis — a resolve of an unknown
    # ticket returning 404 is the "handler was reached" signal.
    interactions_fake = InteractionsFake()
    isettings = InteractionsSettings(public_base_url="https://cb.example")

    @asynccontextmanager
    async def interactions_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield interactions_fake

    monkeypatch.setattr(router, "client_ctx", interactions_ctx)
    monkeypatch.setattr(router, "interactions_settings", lambda: isettings)

    routes = [
        Route("/api/interactions/stream", router.stream, methods=["GET"]),
        Route("/api/interactions/{interaction_id}/answer", router.answer, methods=["POST"]),
        Route("/api/interactions/callback/{ticket}", router.callback, methods=["GET", "POST"]),
    ]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    client = TestClient(app)
    # Expose the seam a verifier-boundary test seeds a bound question into.
    client.interactions_fake = interactions_fake
    client.interactions_settings = isettings
    return client


def test_callback_post_reachable_unauthenticated(boundary_client):
    resp = boundary_client.post("/api/interactions/callback/UNKNOWN", content=b"{}")
    # 404 from the handler (unknown ticket) proves the request reached it.
    assert resp.status_code == 404
    assert resp.json() == {"error": "not found"}


def test_callback_get_reachable_unauthenticated(boundary_client):
    resp = boundary_client.get("/api/interactions/callback/UNKNOWN")
    assert resp.status_code == 404


def test_stream_rejected_without_auth(boundary_client):
    resp = boundary_client.get("/api/interactions/stream")
    assert resp.status_code in (401, 403)


def test_answer_rejected_without_auth(boundary_client):
    resp = boundary_client.post("/api/interactions/i1/answer", json={"answer": "x"})
    assert resp.status_code in (401, 403)


class _PostOnlyVerifier:
    """A body-signature verifier: passes any body, signs the body only."""

    post_only = True

    async def verify(self, body, headers, config):
        return None


async def _seed_bound_question(fake, settings, ticket: str, verifier: dict) -> str:
    from datetime import UTC, datetime, timedelta

    from tai_contract.interactions import AnswerFormat, InteractionRequest

    from tai_skeleton.interactions.store import InteractionStore

    store = InteractionStore(settings.key_prefix)
    now = datetime.now(UTC)
    request = InteractionRequest(
        interaction_id="i1",
        group_id="g1",
        question="Sign?",
        answer_format=AnswerFormat.EXTERNAL,
        format_payload={"url": "https://ext.example/resource", "verifier": verifier},
        reply_to=store.reply_key("i1"),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
    )
    await store.add(fake, request, idle_ttl=86400, ticket=ticket, ticket_ttl=60)
    return "i1"


def test_post_only_empty_body_query_answer_denied_unauthenticated(boundary_client):
    # The security fix holds through the real middleware stack with NO credentials:
    # a post_only verifier + empty body + ``?approved=true`` is denied (400), the
    # answer never injected — proving the public door still reaches the handler.
    import asyncio

    from tai_contract.app import tai_app

    from tai_skeleton.app.instance import build_app

    app = build_app()
    tai_app.bind(app)
    reg = app._webhook_verifier_registry
    reg.register("prov", _PostOnlyVerifier())
    try:
        asyncio.run(
            _seed_bound_question(
                boundary_client.interactions_fake,
                boundary_client.interactions_settings,
                "TKT",
                {"name": "prov", "config": {}},
            )
        )
        resp = boundary_client.post("/api/interactions/callback/TKT?approved=true", content=b"")
        assert resp.status_code == 400
        assert resp.json()["error"] == router._POST_ONLY_EMPTY_BODY_DENY
    finally:
        reg.reset()
