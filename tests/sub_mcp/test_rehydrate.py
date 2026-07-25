"""The wired ``rehydrate_sub_mcp_apps`` startup/reload handler: after a
``reset()`` wipes the per-worker route cache (as every reload does), the handler
re-materializes every persisted registration from the shared store — in both the
in-memory and the (faked) Redis store modes.
"""

from __future__ import annotations

from typing import cast

import pytest
from tai42_contract.sub_mcp import RouteConfig

from tai42_skeleton.app.instance import app, rehydrate_sub_mcp_apps
from tai42_skeleton.app.sub_mcp_app import SubMcpAppRouter
from tai42_skeleton.sub_mcp import store as store_mod
from tai42_skeleton.sub_mcp.settings import sub_mcp_settings
from tai42_skeleton.sub_mcp.store import InMemorySubMcpStore


@pytest.fixture
def clean_router():
    """Reset the process router before and after so a rehydrate test never leaks
    routes into (or inherits them from) a sibling test."""
    router = cast(SubMcpAppRouter, app.sub_app.mcp_sub_app_router)
    router.reset()
    yield router
    router.reset()


async def test_rehydrate_restores_registrations_in_memory(monkeypatch, clean_router):
    fresh = InMemorySubMcpStore()
    await fresh.save_route("weather", RouteConfig(tools=["get_forecast"], transport="http"))
    await fresh.save_route("news", RouteConfig(tools=["headlines"], transport="sse"))
    monkeypatch.setattr(store_mod, "_IN_MEMORY_STORE", fresh)

    # The reload wiped the per-worker cache; the durable store still holds the routes.
    assert clean_router.routes == {}
    await rehydrate_sub_mcp_apps()

    assert clean_router.routes == {
        "weather": RouteConfig(tools=["get_forecast"], transport="http"),
        "news": RouteConfig(tools=["headlines"], transport="sse"),
    }


async def test_rehydrate_restores_registrations_from_redis(monkeypatch, clean_router, fake_redis, fake_client_ctx):
    monkeypatch.setenv("SUB_MCP_REDIS_URL", "redis://localhost:6379/0")
    sub_mcp_settings.cache_clear()
    monkeypatch.setattr(store_mod, "client_ctx", fake_client_ctx)
    try:
        # Seed the durable store through its own write surface, then rehydrate.
        store = store_mod.get_sub_mcp_store()
        assert isinstance(store, store_mod.RedisSubMcpStore)
        await store.save_route("weather", RouteConfig(tools=["get_forecast"], transport="http"))

        assert clean_router.routes == {}
        await rehydrate_sub_mcp_apps()
        assert clean_router.routes == {"weather": RouteConfig(tools=["get_forecast"], transport="http")}
    finally:
        sub_mcp_settings.cache_clear()


async def test_rehydrate_no_registrations_is_a_noop(monkeypatch, clean_router):
    monkeypatch.setattr(store_mod, "_IN_MEMORY_STORE", InMemorySubMcpStore())
    await rehydrate_sub_mcp_apps()
    assert clean_router.routes == {}
