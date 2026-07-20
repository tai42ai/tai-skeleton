"""The durable sub-MCP registration store — the source of truth for every
``slug -> RouteConfig`` binding.

Sub-MCP routing state is uvicorn-worker-scoped: only the HTTP/MCP workers serve
the ``/app/{slug}`` mount, so only they register/rehydrate here. Backend workers
(celery/rq/arq) serve no ``/app/{slug}`` routes and do not participate in this
store, so the worker bus is NOT the propagation mechanism for sub-MCP
registrations — a shared Redis hash is. Each worker's in-process router
(``SubMcpAppRouter``) is a per-worker cache of this store; the store is authoritative.

Two impls behind one :class:`SubMcpStore` seam:

* :class:`RedisSubMcpStore` — stateless per-op; every method opens a pooled
  ``client_ctx(RedisClient, settings.redis)`` and works one Redis hash
  (``settings.routes_key``): field = slug, value = ``RouteConfig`` JSON. A
  malformed stored value raises loudly (``ValidationError`` propagates) — never
  skip-and-continue.
* :class:`InMemorySubMcpStore` — the same surface over a plain dict, a
  module-level singleton (see :func:`get_sub_mcp_store`).
"""

from __future__ import annotations

from typing import Protocol

from tai42_contract.sub_mcp import RouteConfig
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.sub_mcp.settings import SubMcpSettings, sub_mcp_settings
from tai42_skeleton.utils.redis_typing import awaited


class SubMcpStore(Protocol):
    """The durable-registration seam both store impls satisfy."""

    async def get_route(self, slug: str) -> RouteConfig | None: ...

    async def list_routes(self) -> dict[str, RouteConfig]: ...

    async def save_route(self, slug: str, config: RouteConfig) -> None: ...

    async def delete_route(self, slug: str) -> bool:
        """Delete ``slug``; return whether the field existed."""
        ...


class RedisSubMcpStore:
    """Stateless Redis-hash view of the registrations. Every op opens a pooled
    client and touches the one ``routes_key`` hash, so a fresh instance per call is
    cheap and holds no connection of its own."""

    def __init__(self, settings: SubMcpSettings) -> None:
        self._settings = settings

    async def get_route(self, slug: str) -> RouteConfig | None:
        async with client_ctx(RedisClient, self._settings.redis) as r:
            raw = await awaited(r.hget(self._settings.routes_key, slug))
        if raw is None:
            return None
        # A wrong-shape stored value raises here (loud) rather than being skipped.
        return RouteConfig.model_validate_json(raw)

    async def list_routes(self) -> dict[str, RouteConfig]:
        async with client_ctx(RedisClient, self._settings.redis) as r:
            data = await awaited(r.hgetall(self._settings.routes_key))
        return {slug: RouteConfig.model_validate_json(raw) for slug, raw in data.items()}

    async def save_route(self, slug: str, config: RouteConfig) -> None:
        async with client_ctx(RedisClient, self._settings.redis) as r:
            await awaited(r.hset(self._settings.routes_key, slug, config.model_dump_json()))

    async def delete_route(self, slug: str) -> bool:
        async with client_ctx(RedisClient, self._settings.redis) as r:
            removed = await awaited(r.hdel(self._settings.routes_key, slug))
        return removed > 0


class InMemorySubMcpStore:
    """A per-process dict over the same surface. ``list_routes`` returns a fresh
    copy so a concurrent write never mutates a caller's iteration."""

    def __init__(self) -> None:
        self._routes: dict[str, RouteConfig] = {}

    async def get_route(self, slug: str) -> RouteConfig | None:
        return self._routes.get(slug)

    async def list_routes(self) -> dict[str, RouteConfig]:
        return dict(self._routes)

    async def save_route(self, slug: str, config: RouteConfig) -> None:
        self._routes[slug] = config

    async def delete_route(self, slug: str) -> bool:
        return self._routes.pop(slug, None) is not None


# The in-memory store is a MODULE-LEVEL singleton, deliberately EXEMPT from the
# settings-reset registry (it is neither a ``settings_cache`` nor
# ``register_settings_reset`` target). Its entire purpose is to survive
# ``reload_config``'s settings-reset wipe within the process — wiring it into the
# reset registry would reintroduce the reload data loss the durable store exists to
# fix. This is the deliberate opposite of the hooks manager singleton, which IS
# reset-registered because it exists to HONOR ``HOOKS_*`` config; the asymmetry is
# intentional, not an inconsistency.
_IN_MEMORY_STORE = InMemorySubMcpStore()


def get_sub_mcp_store() -> SubMcpStore:
    """Resolve the active store from fresh ``SUB_MCP_*`` config.

    A ``SUB_MCP_REDIS_URL`` selects a new (stateless, cheap) ``RedisSubMcpStore``;
    otherwise the module-level in-memory singleton. A reload that flips in-memory →
    Redis does NOT migrate prior in-memory registrations — the one seam this leaves
    open; a multi-worker deployment must set ``SUB_MCP_REDIS_URL`` from the start.
    """
    settings = sub_mcp_settings()
    if settings.in_memory:
        return _IN_MEMORY_STORE
    return RedisSubMcpStore(settings)
