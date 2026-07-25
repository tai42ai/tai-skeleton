"""The single write choke point for sub-MCP registrations.

Both durable writers — the HTTP router (``POST``/``DELETE /api/sub-mcp``) and the
backup-restore path — go through here so the shared store and the in-process
router stay coherent. The store write happens FIRST: a registration is durable
before the (cheap) in-process router swap, so a crash between the two leaves the
registration recoverable (the next rehydrate re-materializes it). A router-first
order would durably LOSE a registration on a crash in that window — the store-first
ordering is the durability contract. Errors from either half propagate loudly.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app
from tai42_contract.sub_mcp import RouteConfig

from tai42_skeleton.app.sub_mcp_app import validate_registration
from tai42_skeleton.sub_mcp.store import get_sub_mcp_store


async def register_sub_mcp_app(slug: str, tools: list[str], transport: str = "http") -> None:
    """Persist a sub-MCP registration, then bind it into this worker's router.

    The slug/transport are validated up front so invalid input can never reach the
    store, THEN the store write lands (durability), THEN the in-process router swap
    runs. Any failure — a store error, a bad slug, a build error surfaced by the
    router — propagates loudly.
    """
    validate_registration(slug, transport)
    await get_sub_mcp_store().save_route(slug, RouteConfig(tools=tools, transport=transport))
    await tai42_app.sub_app.mcp_sub_app_router.register_sub_mcp_app(slug, tools, transport=transport)


async def unregister_sub_mcp_app(slug: str) -> bool:
    """Delete a sub-MCP registration from the store and, if bound here, the router.

    Returns whether anything was removed (present in the store OR bound locally). A
    slug registered on a sibling worker is store-only on this one, so the store
    delete alone makes it deletable from any worker; a slug also bound in this
    worker's router is torn down locally too.
    """
    removed_from_store = await get_sub_mcp_store().delete_route(slug)
    router = tai42_app.sub_app.mcp_sub_app_router
    bound_locally = slug in router.routes
    if bound_locally:
        await router.unregister_sub_mcp_app(slug)
    return removed_from_store or bound_locally
