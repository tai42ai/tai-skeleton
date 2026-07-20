"""Sub-MCP app registry operations — list, register/reload, and unregister.

Three operations over the DURABLE sub-MCP registration store, which the live
``tai_app.sub_app.mcp_sub_app_router`` caches per worker:

* ``list_sub_mcp`` reads the durable store (not this worker's in-process cache) so
  the listing is coherent across workers.
* ``register_sub_mcp`` registers (or reloads) a sub-MCP app exposing ``tools``
  under ``slug`` on an optional ``transport``. The write goes through the
  store-first service, so the registration is durable before the in-process router
  swap. An unknown tool is a loud 404; the structural body validation (slug shape,
  transport, tool-name types) is a loud 400 raised by the route's extractor.
* ``unregister_sub_mcp`` unregisters a sub-MCP app and tears down its ASGI
  lifespan; a slug present in NEITHER the store NOR this worker's router is a loud
  404, but a slug registered on a sibling worker is store-only here and is still
  deletable from any worker.

Both mutating operations are reload-gated (they rebind live tools) — the adapter
holds the retriable 503 on the route edge while a reload owns the gate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from tai_contract.app import tai_app

from tai_skeleton.access_control import management
from tai_skeleton.operations import BadRequestError, NotFoundError, operation
from tai_skeleton.sub_mcp import service
from tai_skeleton.sub_mcp.store import get_sub_mcp_store


class SubMcpRegistration(BaseModel):
    """A sub-MCP app registration: the ``slug`` to mount it under, the tool names
    it exposes, and the optional ``transport`` (``http`` default)."""

    slug: str = Field(min_length=1)
    tools: list[str]
    transport: str = "http"


@operation(summary="List the registered sub-MCP apps", tags=["sub-mcp"])
async def list_sub_mcp() -> dict:
    # Read the durable store, not this worker's in-process cache, so the list is
    # coherent across workers. RouteConfig is a pydantic model; model_dump yields
    # the JSON-safe fields (tools + transport) so no live object leaks into the body.
    routes = await get_sub_mcp_store().list_routes()
    return {slug: config.model_dump() for slug, config in routes.items()}


@operation(
    summary="Register or reload a sub-MCP app",
    tags=["sub-mcp"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, NotFoundError],
    request_model=SubMcpRegistration,
)
async def register_sub_mcp(slug: str, tools: list[str], transport: str = "http") -> dict:
    # Resolve every tool against the live registry BEFORE registering: an unknown
    # name would otherwise blow up in ``_build_sub_app`` on the FIRST request (a
    # generic 500 to a different caller), so reject it loudly here up front. The
    # structural body validation (slug/transport/tool-name types) is done at the
    # HTTP edge by the route's extractor, which raises the 400s.
    registered = await tai_app.tools.get_tools()
    missing = sorted(t for t in tools if t not in registered)
    if missing:
        raise NotFoundError(f"unknown tool(s): {', '.join(missing)}")
    # Store write FIRST (durable), then the in-process router swap — the service
    # owns that ordering so a registration survives a crash between the two.
    await service.register_sub_mcp_app(slug, tools, transport=transport)
    # A mount is a reachable surface, so a mount change must invalidate cached
    # capability projections exactly as a route-table edit does — bump the version
    # AFTER the durable write so a warm projection re-reads the new mount set.
    await management.bump_policy_version()
    return {"slug": slug, "tools": tools, "transport": transport}


@operation(
    summary="Unregister a sub-MCP app",
    tags=["sub-mcp"],
    reload_gated=True,
    errors=[NotFoundError],
)
async def unregister_sub_mcp(slug: str) -> dict:
    # The service deletes from the store and, if bound here, the local router. It
    # returns whether anything was removed (store OR local): a slug present in
    # neither is a loud 404, but a slug registered on a sibling worker is store-only
    # here and is still deletable from any worker.
    removed = await service.unregister_sub_mcp_app(slug)
    if not removed:
        raise NotFoundError(f"sub-MCP app {slug!r} not found")
    # A removed mount is a surface that vanished, so invalidate cached projections
    # like a route-table edit — only after a real removal (a 404 wrote nothing).
    await management.bump_policy_version()
    return {"slug": slug, "removed": True}
