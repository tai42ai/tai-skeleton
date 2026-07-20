"""App-level rate limiter for the two PUBLIC webhook-receiving door families.

Applied ONLY to ``/universal_webhook/*`` (the fully public hooks ingress) and
``/api/interactions/callback/*`` (the ticket-gated interactions callback). Every
other path — including all authed routes — passes straight through: the
credential is the gate there. Each family has its own per-minute limit + a
10-second burst window and an enable switch, so a flood on one public door
cannot exhaust the other's budget.

Keying and window semantics: a fixed-window Redis counter per client bucket,
INCR + EXPIRE issued in one pipeline (a pipeline cannot branch on INCR's result;
re-setting the TTL every hit is harmless), key TTL = 2x the window. The client
bucket collapses an IPv6 address to its /64; X-Forwarded-For is honoured only
when the direct peer is a configured trusted proxy.

The app registers this middleware at construction, so both public doors are
rate-limited by default — a public door is exposed by design and must not ship
without its flood control. Each family can still be tuned or turned off via its
``TAI_RATE_LIMIT_*`` enable/limit settings (both enabled by default).
"""

from __future__ import annotations

import ipaddress
import math
import time

from redis.asyncio import Redis as AsyncRedis
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.settings.rate_limit import RateLimitSettings, rate_limit_settings

# Path prefixes of the two public door families, mapped to the settings key that
# names their limit/burst/enable fields.
_WEBHOOK_PREFIX = "/universal_webhook/"
_CALLBACK_PREFIX = "/api/interactions/callback/"


def _bucket_ip(ip_str: str) -> str:
    """Bucket key for a client IP: an IPv6 address collapses to its /64 prefix (a
    single host routinely holds a whole /64); IPv4 is used as-is. An unparseable
    value gets its own bucket by raw string."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return ip_str
    if isinstance(ip, ipaddress.IPv6Address):
        # An IPv4-mapped address (``::ffff:a.b.c.d``) is really an IPv4 client;
        # bucket it by the embedded IPv4 so a dual-stack listener doesn't collapse
        # every mapped client into the single ``::/64`` bucket.
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address)
    return ip_str


def _client_bucket(conn: HTTPConnection, trusted_proxies: list[str]) -> str:
    """Resolve the rate-limit bucket. Only when the direct peer is a trusted proxy
    do we read X-Forwarded-For (right-most hop not itself a trusted proxy);
    otherwise XFF is ignored entirely (it is spoofable)."""
    peer = conn.client.host if conn.client else "unknown"
    if peer in trusted_proxies:
        hops = [h.strip() for h in conn.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        client_ip = peer
        for hop in reversed(hops):
            if hop not in trusted_proxies:
                client_ip = hop
                break
    else:
        client_ip = peer
    return _bucket_ip(client_ip)


async def _retry_after(r: AsyncRedis, prefix: str, family: str, bucket: str, limit: int, burst: int) -> int | None:
    """Fixed-window Redis limiter. Returns the ``Retry-After`` seconds when either
    window is over its limit, else ``None``. INCR + EXPIRE are one pipeline with
    EXPIRE issued UNCONDITIONALLY (a pipeline cannot branch on INCR's result;
    re-setting the TTL every hit is harmless). Key TTL = 2x the window. The
    ``family`` segment keeps the two public door families' counters disjoint."""
    now = time.time()
    unix_minute = int(now // 60)
    unix_10s = int(now // 10)
    m_key = f"{prefix}rl:{family}:m:{bucket}:{unix_minute}"
    s_key = f"{prefix}rl:{family}:s:{bucket}:{unix_10s}"

    pipe = r.pipeline()
    pipe.incr(m_key)
    pipe.expire(m_key, 120)
    pipe.incr(s_key)
    pipe.expire(s_key, 20)
    results = await pipe.execute()
    m_count, s_count = int(results[0]), int(results[2])

    if m_count > limit:
        return max(1, math.ceil((unix_minute + 1) * 60 - now))
    if s_count > burst:
        return max(1, math.ceil((unix_10s + 1) * 10 - now))
    return None


def _family(path: str, settings: RateLimitSettings) -> tuple[str, int, int] | None:
    """Map a request path to its public door family and that family's
    (name, limit, burst), or ``None`` for any path the limiter must not touch
    (a disabled family included — an off switch means pass through, not block)."""
    if path.startswith(_WEBHOOK_PREFIX):
        if not settings.webhook_enabled:
            return None
        return "webhook", settings.webhook_limit, settings.webhook_burst
    if path.startswith(_CALLBACK_PREFIX):
        if not settings.interactions_callback_enabled:
            return None
        return "interactions_callback", settings.interactions_callback_limit, settings.interactions_callback_burst
    return None


class RateLimitMiddleware:
    """Rate-limits the public webhook door families; passes everything else through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = rate_limit_settings()
        conn = HTTPConnection(scope)
        family = _family(conn.url.path, settings)
        if family is None:
            await self.app(scope, receive, send)
            return

        name, limit, burst = family
        bucket = _client_bucket(conn, settings.trusted_proxies)
        async with client_ctx(RedisClient, settings.redis) as r:
            retry_after = await _retry_after(r, settings.key_prefix, name, bucket, limit, burst)
        if retry_after is not None:
            response = JSONResponse(
                {"error": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
