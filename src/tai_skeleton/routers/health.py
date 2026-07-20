"""``/health`` (liveness) and ``/ready`` (readiness) custom routes.

``/health`` returns a static ``OK`` — pure liveness.

``/ready`` pings exactly the backing stores THIS deployment has wired, reusing
each subsystem's existing gating logic instead of inventing new config. A worker
whose Redis/Postgres is unreachable 500s every real request while ``/health``
stays green; ``/ready`` lets an orchestrator rotate it and a load balancer drain
it. Distinct connections are deduped and pinged once, concurrently, each under a
module-constant timeout budget.
"""

from __future__ import annotations

import asyncio
import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from tai_contract.access_control.registry import get_identity_provider_factory
from tai_contract.app import tai_app
from tai_kit.clients import ClientSettings, client_ctx
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.access_control.settings import access_control_settings
from tai_skeleton.app import instance
from tai_skeleton.connectors.settings import connector_store_settings
from tai_skeleton.hooks.settings import HooksSettings
from tai_skeleton.interactions.settings import interactions_settings
from tai_skeleton.routers.tool_runs_settings import tool_runs_settings
from tai_skeleton.settings.rate_limit import rate_limit_settings
from tai_skeleton.sub_mcp.settings import sub_mcp_settings
from tai_skeleton.versioning.settings import versioning_store_settings

logger = logging.getLogger(__name__)

# Wall-clock budget for a single readiness ping (connect + command). A module
# constant, deliberately NOT a settings knob — ``/ready`` adds no config surface.
_READINESS_TIMEOUT_SECONDS = 5.0


@tai_app.http.custom_route(
    "/health",
    methods=["GET"],
    summary="Liveness probe",
    tags=["health"],
    response_model=None,
    authed=False,
)
async def health_check(request):
    return PlainTextResponse("OK")


def _wired_connections() -> list[tuple[str, type, ClientSettings]]:
    """Return ``(subsystem, client class, connection settings)`` for every backing
    store THIS deployment has wired, reusing each subsystem's existing gate.

    A subsystem may contribute more than one connection (``connectors`` uses both
    Redis and Postgres); its check is ``ok`` only when all of them ping clean.
    """
    conns: list[tuple[str, type, ClientSettings]] = []

    ac = access_control_settings()
    if ac.enable:
        # Each configured identity provider declares its OWN readiness target(s) through
        # the IdentityProvider ABC; core enumerates them generically instead of naming a
        # concrete provider or its store. A provider with no pingable backing store
        # declares none. Resolved through the module-level registry, the same deferred
        # path the auth adapter and boot probe use; identical connections across
        # providers are deduped downstream.
        for name in ac.auth_providers:
            provider = get_identity_provider_factory(name)(ac)
            for target in provider.readiness_targets():
                conns.append((target.name, target.client, target.settings))

    tr = tool_runs_settings()
    if tr.redis.redis_url:
        conns.append(("tool_runs", RedisClient, tr.redis))

    inter = interactions_settings()
    if inter.redis.redis_url:
        conns.append(("interactions", RedisClient, inter.redis))

    rl = rate_limit_settings()
    if (rl.webhook_enabled or rl.interactions_callback_enabled) and rl.redis.redis_url:
        conns.append(("rate_limit", RedisClient, rl.redis))

    hooks = HooksSettings()
    if not hooks.in_memory:
        conns.append(("hooks", RedisClient, hooks.redis))

    # The durable sub-MCP registration store: Redis-backed whenever SUB_MCP_REDIS_URL
    # is set (its rehydrate handler runs on every boot/reload and every registration
    # writes to it), in-memory otherwise — the same gate shape as hooks.
    sub_mcp = sub_mcp_settings()
    if not sub_mcp.in_memory:
        conns.append(("sub_mcp", RedisClient, sub_mcp.redis))

    if instance.connectors_in_use():
        store = connector_store_settings()
        conns.append(("connectors", RedisClient, store.redis))
        conns.append(("connectors", PostgresClient, store.pg))

    if instance.versioned_store_in_use():
        conns.append(("versioning", PostgresClient, versioning_store_settings()))

    return conns


async def _ping_redis(settings: ClientSettings) -> None:
    async with client_ctx(RedisClient, settings) as r:
        # redis-py types the async ``ping`` with the sync ``bool`` return, so pyright
        # sees the awaited value as non-awaitable; it is a coroutine at runtime.
        await r.ping()  # pyright: ignore[reportGeneralTypeIssues]


async def _ping_postgres(settings: ClientSettings) -> None:
    async with client_ctx(PostgresClient, settings) as pool, pool.connection() as conn:
        await conn.execute("SELECT 1")


async def _ping_connection(client_cls: type, settings: ClientSettings) -> Exception | None:
    """Ping one distinct connection under the readiness timeout.

    Returns ``None`` on success, or the raised exception on failure. The failure
    is never swallowed: it is logged in full here (one warning per failed
    connection, with the traceback) and returned so the response can carry the
    exception TYPE only — the message would leak internal hosts/ports.
    """
    try:
        async with asyncio.timeout(_READINESS_TIMEOUT_SECONDS):
            if client_cls is RedisClient:
                await _ping_redis(settings)
            else:
                await _ping_postgres(settings)
    except Exception as exc:  # reported to the caller and logged with detail — never swallowed
        logger.warning("readiness ping failed for %s", client_cls.__name__, exc_info=True)
        return exc
    return None


@tai_app.http.custom_route(
    "/ready",
    methods=["GET"],
    summary="Readiness probe",
    tags=["health"],
    response_model=None,
    authed=False,
)
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe: ping every backing store this deployment has wired.

    Pings exactly the Redis/Postgres connections the gated-in subsystems use,
    deduped by connection identity so a shared connection is pinged ONCE, and each
    distinct connection concurrently under a 5s budget. All pass -> 200
    ``{"status": "ready", "checks": {name: "ok"}}``; any fail -> 503
    ``{"status": "not_ready", ...}`` whose failing checks carry only the exception
    TYPE name — never the message, which would leak internal hosts/ports (full
    detail is logged server-side, one warning per failed connection). A deployment
    with nothing gated in returns 200 with empty checks.

    Mapped public by the operator like ``/health`` (there is no code-side public
    list — ``ResourceGuardMiddleware`` denies unknown routes). When the
    access-control Redis itself is down the middleware fails closed with 403 before
    this handler runs, so a probe still sees a non-200 and rotates the worker
    either way.
    """
    conns = _wired_connections()

    # Dedupe by (client class, connection identity): several subsystems default to
    # one shared redis://localhost:6379/0, so a shared connection is pinged once and
    # every subsystem on it reports that one result.
    distinct: dict[tuple[type, str], tuple[type, ClientSettings]] = {}
    subsystem_keys: dict[str, list[tuple[type, str]]] = {}
    for name, client_cls, settings in conns:
        key = (client_cls, json.dumps(settings.client_kwargs(), sort_keys=True))
        distinct.setdefault(key, (client_cls, settings))
        subsystem_keys.setdefault(name, []).append(key)

    keys = list(distinct)
    results = await asyncio.gather(*(_ping_connection(*distinct[key]) for key in keys))
    outcome: dict[tuple[type, str], Exception | None] = dict(zip(keys, results, strict=True))

    checks: dict[str, str] = {}
    ready = True
    for name, keys_for_name in subsystem_keys.items():
        failure = next((outcome[key] for key in keys_for_name if outcome[key] is not None), None)
        if failure is None:
            checks[name] = "ok"
        else:
            checks[name] = type(failure).__name__
            ready = False

    if ready:
        return JSONResponse({"status": "ready", "checks": checks})
    return JSONResponse({"status": "not_ready", "checks": checks}, status_code=503)
