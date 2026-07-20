"""End-to-end: the mcp-SDK route gate admits a valid access-control key.

fastmcp wraps the main ``/mcp`` (and ``/sse``) endpoints in the mcp-SDK
``RequireAuthMiddleware``, whose gate is ``isinstance(scope["user"],
AuthenticatedUser)``. This stands the REAL :class:`AuthAdapter` middleware chain
in front of that exact gate and drives it with a valid and an invalid key: a
valid key must reach the guarded endpoint (NOT a blanket 401), an invalid or
missing key must be rejected 401. A unit-level ``TaiUser`` assertion in isolation
cannot see the SDK's auth gate, so this test stands the real ``AuthAdapter`` /
``RequireAuthMiddleware`` chain in front of ``/mcp`` and drives that exact gate.
"""

from __future__ import annotations

import pytest
from mcp.server.auth.middleware.bearer_auth import RequireAuthMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.testclient import TestClient
from tai_identity_redis import redis_api_key_provider as provider_module
from tai_kit.utils.data.string_util import hash_api_key

from tai_skeleton.access_control import policy as policy_module
from tai_skeleton.access_control import store as store_module
from tai_skeleton.access_control import verifier as verifier_module
from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import AccessControlSettings
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, _FakeApp, make_client_ctx, make_pg_ctx

# The enforcer's alru cache is created in the test's loop and first used in the
# TestClient's separate portal loop — a benign loop-change reset that is a test
# artifact (the real server holds one loop), not a product warning.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_VALID_KEY = "valid-key"
_MCP_SCOPE = "mcp-scope"


async def _endpoint(scope, receive, send) -> None:
    await PlainTextResponse("mcp-ok")(scope, receive, send)


@pytest.fixture
def bound_app():
    """Bind a fake ``tai_app`` so the enforcer can render the (empty) condition."""
    from tai_contract.app import tai_app

    app = _FakeApp()
    tai_app.bind(app)
    try:
        yield app
    finally:
        tai_app.bind(None)


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = AccessControlSettings()
    identity_key = f"{settings.key_prefix}{hash_api_key(_VALID_KEY)}"
    # The identity record hash + the version counter stay on Redis; the route and
    # the enforced policy body come from the Postgres store. No context is seeded —
    # an absent context hash reads as the empty live view via ``HGETALL``.
    fake = FakeRedis(
        hashes={
            identity_key: {"user_id": "u1", "description": "d"},
        },
    )
    pg = FakeAccessControlPg()
    pg.add_route("/mcp", _MCP_SCOPE)
    pg.add_policy("u1", scopes=[_MCP_SCOPE])
    ctx = make_client_ctx(fake)
    monkeypatch.setattr(verifier_module, "client_ctx", ctx)
    monkeypatch.setattr(policy_module, "client_ctx", ctx)
    monkeypatch.setattr(provider_module, "client_ctx", ctx)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))

    # The guarded endpoint mimics fastmcp's /mcp mount: the mcp-SDK route gate wraps
    # it, and the real access-control middleware chain runs in front.
    guarded = RequireAuthMiddleware(_endpoint, required_scopes=[])
    app = Starlette(routes=[Mount("/mcp", app=guarded)], middleware=AuthAdapter(settings).get_middleware())
    return TestClient(app)


def test_mcp_admits_valid_key(monkeypatch: pytest.MonkeyPatch, bound_app) -> None:
    client = _client(monkeypatch)
    resp = client.get("/mcp", headers={"X-Api-Key": _VALID_KEY})
    # The mcp-SDK gate recognizes ``TaiUser`` as an ``AuthenticatedUser`` and admits
    # the request — NOT a blanket 401.
    assert resp.status_code != 401
    assert resp.status_code == 200
    assert resp.text == "mcp-ok"


def test_mcp_rejects_invalid_key(monkeypatch: pytest.MonkeyPatch, bound_app) -> None:
    client = _client(monkeypatch)
    resp = client.get("/mcp", headers={"X-Api-Key": "bogus"})
    assert resp.status_code == 401


def test_mcp_rejects_missing_key(monkeypatch: pytest.MonkeyPatch, bound_app) -> None:
    client = _client(monkeypatch)
    # No ``X-Api-Key`` header at all: the real auth chain rejects the caller 401,
    # the same as an invalid key — the endpoint is never reached without a key.
    resp = client.get("/mcp")
    assert resp.status_code == 401
