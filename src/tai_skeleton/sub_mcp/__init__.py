"""Durable sub-MCP registration store, settings, and write service.

Sub-MCP app registrations (``slug -> {tools, transport}``) are per-uvicorn-worker
in-process routing state on the ``SubMcpAppRouter`` (``app/sub_mcp_app.py``). This
package makes them durable and cross-worker-coherent by persisting them to a shared
store (Redis, with a per-process in-memory fallback) that the router treats as its
source of truth: :mod:`settings` selects the backend from ``SUB_MCP_*`` config,
:mod:`store` is the ``slug -> RouteConfig`` seam, and :mod:`service` is the single
write choke point (store write FIRST, then the in-process router swap).

Scope: only the uvicorn HTTP/MCP workers serve ``/app/{slug}`` and use this store.
Backend workers (celery/rq/arq) serve no ``/app/{slug}`` routes and do not
participate, so the worker bus is not the propagation mechanism here.
"""

from __future__ import annotations

from tai_skeleton.sub_mcp.settings import SubMcpRedisSettings, SubMcpSettings, sub_mcp_settings
from tai_skeleton.sub_mcp.store import (
    InMemorySubMcpStore,
    RedisSubMcpStore,
    SubMcpStore,
    get_sub_mcp_store,
)

__all__ = [
    "InMemorySubMcpStore",
    "RedisSubMcpStore",
    "SubMcpRedisSettings",
    "SubMcpSettings",
    "SubMcpStore",
    "get_sub_mcp_store",
    "sub_mcp_settings",
]
