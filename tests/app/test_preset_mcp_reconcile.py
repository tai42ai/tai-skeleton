"""Preset reconciliation across scoped MCP reload / deregister.

Drives the REAL app (``instance.app``) with a faked MCP probe so an MCP server
binds a base tool a preset is then built over, and exercises the three
interactions the reconciliation seam covers:

* ``reload_mcp`` rebinds the base — the dependent preset is re-registered from its
  spec so its ``TransformedTool`` tracks the freshly-bound base (not the stale
  pre-reload one);
* ``deregister_mcp`` removes the base — a dependent preset is quarantined
  (``conflicted``);
* a returning MCP server whose tool name a registered preset now owns is REFUSED
  (the preset is never clobbered), and the reload result surfaces the conflict.

The store is the stateful in-memory fake (the ``pg`` fixture); the probe is mocked,
so no network or live database is touched.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from tai_contract.agent.base import PresetSpec
from tai_contract.manifest import MCPConfig, TaiMCPConfig
from tai_kit.clients.impl.postgres import PostgresClient

import tai_skeleton.versioning.store as store_module
from tai_skeleton.app import instance
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.manifest import Manifest
from tests.versioning.conftest import FakeVersioningPg


class _FakeMcpTool:
    name = "ping"
    description = "ping"
    inputSchema: ClassVar[dict] = {"type": "object", "properties": {}}
    outputSchema: ClassVar[dict] = {}


def _cfg(title: str = "svc") -> TaiMCPConfig:
    return TaiMCPConfig(title=title, include=[], config=MCPConfig(type="http", url="http://x/mcp"))


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
            "mcp": [_cfg("svc").model_dump()],
        }
    )


@pytest.fixture
def pg(monkeypatch) -> FakeVersioningPg:
    fake = FakeVersioningPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(store_module, "client_ctx", fake_client_ctx)
    # Signal a store-configured deployment (the gate reconcile + rehydrate consult):
    # a versioned preset can only exist when the store is wired up, so faking the
    # store transport must also set its ``VERSIONING_STORE_*`` namespace.
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")
    return fake


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    yield
    mgr = instance.app.preset_manager

    async def _clear() -> None:
        for name in list(mgr.registered_names()):
            await mgr.remove(name)
        provider = instance.app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    for name in list(mgr.quarantined_names()):
        mgr.drop_quarantine(name)


# Reload/deregister are driven through the reload gate — the real production path
# (``routers/manifest`` and the builtin MCP tool call them exactly this way). The
# gate runs the sync admin facet on a worker thread, so the serving loop keeps
# spinning and the preset reconcile the facet marshals back onto it can run. A
# direct call from this on-loop coroutine would freeze the serving loop and
# deadlock (the facet raises loudly on that misuse).
async def _reload_mcp(title: str) -> dict[str, Any]:
    return await reload_gate.run(lambda: instance.app.admin.reload_mcp(title))


async def _deregister_mcp(title: str) -> dict[str, Any]:
    return await reload_gate.run(lambda: instance.app.admin.deregister_mcp(title))


async def _register_versioned(name: str, base_tool: str) -> None:
    await instance.app.presets.store.create_preset(
        PresetSpec(name=name, description="d", base_tool=base_tool, fixed_kwargs={}), extensions=[], tags=[]
    )
    body = await instance.app.presets.store.get_active_body(name)
    await instance.app.preset_manager.register(name, body.base_tool, body.fixed_kwargs, [], [], body.description)


def test_reload_mcp_reregisters_dependent_preset_over_new_base(pg, monkeypatch):
    async def run():
        monkeypatch.setattr(instance.app, "_probe_mcp", AsyncMock(return_value=[_FakeMcpTool()]))
        async with instance.app.app_context(_manifest()):
            assert "svc_ping" in await instance.app.tools.get_tools()
            # A preset over the MCP base tool.
            await instance.app.preset_manager.register("myp", "svc_ping", {}, [], [], "d")
            old_parent = (await instance.app.tools.get_tool("myp")).parent_tool  # type: ignore[attr-defined]

            await _reload_mcp("svc")

            # The preset survives AND its transform now tracks the freshly-bound base
            # (the reconciliation re-registered it from spec, not the stale closure).
            assert instance.app.preset_manager.is_registered("myp")
            new_base = await instance.app.tools.get_tool("svc_ping")
            new_parent = (await instance.app.tools.get_tool("myp")).parent_tool  # type: ignore[attr-defined]
            assert new_parent is new_base
            assert new_parent is not old_parent

    asyncio.run(run())


def test_deregister_mcp_quarantines_dependent_presets(pg, monkeypatch):
    async def run():
        monkeypatch.setattr(instance.app, "_probe_mcp", AsyncMock(return_value=[_FakeMcpTool()]))
        async with instance.app.app_context(_manifest()):
            await _register_versioned("verp", "svc_ping")

            await _deregister_mcp("svc")

            mgr = instance.app.preset_manager
            # The base vanished, so the dependent preset is quarantined — its store
            # row surfaces as ``conflicted`` rather than staying bound to a gone base.
            assert mgr.is_quarantined("verp")
            assert not mgr.is_registered("verp")
            assert "verp" not in await instance.app.tools.get_tools()

    asyncio.run(run())


def test_deregister_store_less_reconcile_opens_no_store(monkeypatch):
    # A store-less deployment (no VERSIONING_STORE_* env): reconcile rebinds/quarantines
    # from the in-memory spec map alone and NEVER opens the versioned Postgres store.
    for key in list(os.environ):
        if key.startswith("VERSIONING_STORE_"):
            monkeypatch.delenv(key, raising=False)

    @asynccontextmanager
    async def forbid_client_ctx(client_cls, settings=None, **kwargs):
        raise AssertionError("versioned store opened in a store-less deployment")
        yield  # pragma: no cover - unreachable, satisfies the context-manager protocol

    monkeypatch.setattr(store_module, "client_ctx", forbid_client_ctx)

    async def run():
        monkeypatch.setattr(instance.app, "_probe_mcp", AsyncMock(return_value=[_FakeMcpTool()]))
        async with instance.app.app_context(_manifest()):
            # A preset over the MCP base tool, bound directly.
            await instance.app.preset_manager.register("depp", "svc_ping", {}, [], [], "d")

            await _deregister_mcp("svc")

            # The base vanished, so the dependent preset is quarantined — reconcile
            # reads only the in-memory spec map, opening no store connection.
            mgr = instance.app.preset_manager
            assert not mgr.is_registered("depp")
            assert mgr.is_quarantined("depp")
            assert "depp" not in await instance.app.tools.get_tools()

    asyncio.run(run())


def test_reload_mcp_refuses_name_owned_by_registered_preset(pg, monkeypatch):
    async def run():
        # The MCP is DOWN at boot, so ``svc_ping`` is unbound; a preset then takes
        # that name. When the server returns, the (re)bind must REFUSE the name (the
        # preset is never clobbered) and surface the conflict in the result.
        monkeypatch.setattr(instance.app, "_probe_mcp", AsyncMock(side_effect=TimeoutError("down")))
        async with instance.app.app_context(_manifest()):
            assert "svc_ping" not in await instance.app.tools.get_tools()
            await instance.app.preset_manager.register("svc_ping", "echo", {}, [], [], "d")

            instance.app._probe_mcp = AsyncMock(return_value=[_FakeMcpTool()])  # type: ignore[method-assign]
            result = await _reload_mcp("svc")

            assert result["preset_conflicts"] == ["svc_ping"]
            assert "svc_ping" not in result["tools"]
            # The preset still owns the name and still binds ``echo``.
            assert instance.app.preset_manager.is_registered("svc_ping")
            assert await instance.app.tools.run_tool("svc_ping", {"text": "hi"}) == "hi"

    asyncio.run(run())
