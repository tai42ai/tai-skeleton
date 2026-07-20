"""The app-level ``RateLimitMiddleware``: it limits BOTH public door families
(webhook + interactions callback) with disjoint per-family budgets, leaves authed
routes untouched, honours per-family enable/limit settings, trips the per-minute
window, and resolves the client bucket (XFF trust + IP collapsing) correctly."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from tai42_skeleton.middleware import rate_limit
from tai42_skeleton.settings.rate_limit import RateLimitSettings

# Import the fake as a package-relative module (tests is a package).
from tests._fakes.interactions_redis import FakeRedis


def _conn(peer: str | None, xff: str | None = None) -> Any:
    """A minimal ``HTTPConnection`` stand-in exposing the ``.client`` and
    ``.headers`` the bucket resolver reads."""
    headers: dict[str, str] = {}
    if xff is not None:
        headers["x-forwarded-for"] = xff
    client = SimpleNamespace(host=peer) if peer is not None else None
    return SimpleNamespace(client=client, headers=headers)


async def _ok(request):
    return PlainTextResponse("ok")


def _build_client(monkeypatch, settings: RateLimitSettings, fake: FakeRedis) -> TestClient:
    monkeypatch.setattr(rate_limit, "rate_limit_settings", lambda: settings)
    monkeypatch.setattr(rate_limit, "time", SimpleNamespace(time=lambda: 100.0))

    @asynccontextmanager
    async def _ctx(cls, s=None, *, fresh=False, **kw):
        yield fake

    monkeypatch.setattr(rate_limit, "client_ctx", _ctx)

    routes = [
        Route("/universal_webhook/{topic}", _ok, methods=["GET", "POST"]),
        Route("/api/interactions/callback/{ticket}", _ok, methods=["GET", "POST"]),
        Route("/api/hooks", _ok, methods=["GET"]),  # an AUTHED route family — never limited
    ]
    app = Starlette(routes=routes)
    return TestClient(rate_limit.RateLimitMiddleware(app))


def _settings(**overrides) -> RateLimitSettings:
    base: dict[str, Any] = {"webhook_burst": 2, "interactions_callback_burst": 2}
    base.update(overrides)
    return RateLimitSettings(**base)


def test_webhook_family_limited_after_burst(monkeypatch):
    client = _build_client(monkeypatch, _settings(), FakeRedis())
    statuses = [client.get("/universal_webhook/orders").status_code for _ in range(3)]
    assert statuses[-1] == 429
    assert statuses[0] == 200


def test_interactions_callback_family_limited_after_burst(monkeypatch):
    client = _build_client(monkeypatch, _settings(), FakeRedis())
    statuses = [client.post("/api/interactions/callback/TKT").status_code for _ in range(3)]
    assert statuses[-1] == 429


def test_authed_route_never_limited(monkeypatch):
    client = _build_client(monkeypatch, _settings(), FakeRedis())
    # Far past any burst; an authed (non-public-door) route is untouched.
    statuses = [client.get("/api/hooks").status_code for _ in range(10)]
    assert set(statuses) == {200}


def test_families_have_disjoint_budgets(monkeypatch):
    client = _build_client(monkeypatch, _settings(), FakeRedis())
    # Exhaust the webhook family.
    for _ in range(3):
        client.get("/universal_webhook/orders")
    assert client.get("/universal_webhook/orders").status_code == 429
    # The callback family still has its own budget.
    assert client.post("/api/interactions/callback/TKT").status_code == 200


def test_disabled_family_passes_through(monkeypatch):
    client = _build_client(monkeypatch, _settings(webhook_enabled=False), FakeRedis())
    statuses = [client.get("/universal_webhook/orders").status_code for _ in range(10)]
    assert set(statuses) == {200}


def test_retry_after_header_present_on_429(monkeypatch):
    client = _build_client(monkeypatch, _settings(), FakeRedis())
    resp = None
    for _ in range(3):
        resp = client.get("/universal_webhook/orders")
    assert resp is not None
    assert resp.status_code == 429
    assert "retry-after" in resp.headers


def test_rate_limit_config_owns_both_public_door_families():
    # The two public door families' limiter config lives on the app-level
    # ``RateLimitSettings`` (each with its own enable/limit/burst), and the
    # ``InteractionsSettings`` carries no limiter fields of its own.
    from tai42_skeleton.interactions.settings import InteractionsSettings

    rate_fields = set(RateLimitSettings.model_fields)
    for field in ("webhook_limit", "webhook_burst", "interactions_callback_limit", "interactions_callback_burst"):
        assert field in rate_fields

    interactions_fields = set(InteractionsSettings.model_fields)
    for field in ("callback_rate_limit_per_minute", "callback_rate_burst", "callback_trusted_proxies"):
        assert field not in interactions_fields


def test_minute_window_trips_with_bounded_retry_after(monkeypatch):
    # A low per-minute limit under a high burst ceiling trips the MINUTE window (not
    # the burst), returning 429 with a Retry-After bounded by the 60s window.
    client = _build_client(monkeypatch, _settings(webhook_limit=3, webhook_burst=100), FakeRedis())
    statuses = [client.get("/universal_webhook/orders").status_code for _ in range(4)]
    assert statuses[:3] == [200, 200, 200]
    resp = client.get("/universal_webhook/orders")
    assert resp.status_code == 429
    assert 0 < int(resp.headers["retry-after"]) <= 60


# -- client-bucket resolution (XFF trust model + IP collapsing) --------------


def test_bucket_ip_ipv4_used_as_is():
    assert rate_limit._bucket_ip("203.0.113.7") == "203.0.113.7"


def test_bucket_ip_ipv6_collapses_to_slash_64():
    # Two addresses in the same /64 share a bucket; the bucket is the /64 network.
    a = rate_limit._bucket_ip("2001:db8:abcd:1234::1")
    b = rate_limit._bucket_ip("2001:db8:abcd:1234::abcd")
    assert a == b == "2001:db8:abcd:1234::"


def test_bucket_ip_ipv4_mapped_unwrapped():
    assert rate_limit._bucket_ip("::ffff:1.2.3.4") == "1.2.3.4"


def test_bucket_ip_unparseable_keeps_raw_string():
    assert rate_limit._bucket_ip("not-an-ip") == "not-an-ip"


def test_client_bucket_ignores_xff_from_untrusted_peer():
    # The direct peer is NOT a trusted proxy, so a spoofable XFF is ignored entirely
    # and the peer itself is the bucket.
    conn = _conn("198.51.100.9", xff="1.2.3.4")
    assert rate_limit._client_bucket(conn, trusted_proxies=[]) == "198.51.100.9"


def test_client_bucket_honours_xff_behind_trusted_proxy():
    # The peer IS a trusted proxy, so XFF is read; the right-most hop that is not
    # itself trusted is the client.
    conn = _conn("10.0.0.1", xff="203.0.113.5, 10.0.0.2")
    bucket = rate_limit._client_bucket(conn, trusted_proxies=["10.0.0.1"])
    assert bucket == "10.0.0.2"


def test_client_bucket_skips_trusted_suffix_in_xff():
    # A trusted-proxy suffix in the XFF chain is skipped to the right-most UNTRUSTED
    # hop — the real client — so a trailing trusted hop cannot mask the origin.
    conn = _conn("10.0.0.1", xff="203.0.113.5, 10.0.0.2")
    bucket = rate_limit._client_bucket(conn, trusted_proxies=["10.0.0.1", "10.0.0.2"])
    assert bucket == "203.0.113.5"


def test_client_bucket_ipv6_client_collapses_to_slash_64():
    conn = _conn("10.0.0.1", xff="2001:db8:abcd:1234::9")
    bucket = rate_limit._client_bucket(conn, trusted_proxies=["10.0.0.1"])
    assert bucket == "2001:db8:abcd:1234::"


def test_client_bucket_no_client_is_unknown():
    conn = _conn(None)
    assert rate_limit._client_bucket(conn, trusted_proxies=[]) == "unknown"
