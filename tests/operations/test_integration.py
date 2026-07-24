"""End-to-end: a fixture operation becomes a route + a spec entry + a projected
MCP tool that an extension wraps and ``tai tools run`` (``app.tools.run_tool``)
dispatches — and survives a reload with ``AuthzMiddleware`` intact."""

from __future__ import annotations

import asyncio

import pytest

from tai42_skeleton.app.instance import app
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import load_api_routes
from tai42_skeleton.authz.middleware import AuthzMiddleware
from tai42_skeleton.authz.resolver import resolve_dispatch
from tai42_skeleton.cli.openapi import build_openapi_spec
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations.registry import operation_registry


@pytest.fixture(autouse=True)
def _clean_server():
    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "routers_modules": ["tests.operations._fixtures.sample_router"],
            "extensions_modules": ["tests.app._fixtures.ext_kinds"],
            "api_tools": {"enabled": True, "extensions": {"sample_greet": [["argswrap"]]}},
            # "none" keeps the surface to just the sample router under test.
            "default_routers": "none",
        }
    )


def test_operation_projects_to_route_spec_and_tool():
    async def run():
        async with app.app_context(_manifest()):
            # Route recorded + in the /api surface.
            assert any(r.path == "/api/sample/greet" for r in load_api_routes())
            # Spec entry emitted.
            spec = build_openapi_spec()
            assert "/api/sample/greet" in spec["paths"]
            assert "post" in spec["paths"]["/api/sample/greet"]

            tools = await app.tools.get_tools()
            # Projected tool present.
            assert "sample_greet" in tools
            # The api_tools.extensions combo attached a branch to the projected op
            # (proving projection ran BEFORE extension wraps).
            assert "sample_greet_argswrap" in tools

            # Runnable via the tool-run path.
            result = await app.tools.run_tool("sample_greet", {"name": "ann"})
            assert result == {"greeting": "hello ann"}
            branch = await app.tools.run_tool("sample_greet_argswrap", {"name": "bea"})
            assert branch == {"greeting": "hello bea"}

    asyncio.run(run())


def test_projection_and_authz_survive_reload():
    async def run():
        async with app.app_context(_manifest()):
            assert "sample_greet" in await app.tools.get_tools()
            assert any(isinstance(m, AuthzMiddleware) for m in app._fast_mcp.middleware)

            # A reload re-runs start() (projection + the resets); run it on a worker
            # thread through the gate exactly as production does.
            await reload_gate.run(lambda: app._update(_manifest()))

            tools = await app.tools.get_tools()
            assert "sample_greet" in tools
            assert "sample_greet_argswrap" in tools
            # The security middleware is not dropped by a reload cycle.
            assert any(isinstance(m, AuthzMiddleware) for m in app._fast_mcp.middleware)

    asyncio.run(run())


def test_disabled_api_tools_projects_empty_surface():
    """With ``api_tools.enabled`` false the projection registers no tools — the
    empty surface is the disabled path, NOT an empty registry: the registry is
    fully repopulated at boot, and only the projection is gated off."""

    async def run():
        async with app.app_context(Manifest.model_validate({"api_tools": {"enabled": False}})):
            assert await app.tools.get_tools() == {}
            # The registry IS populated (the boot repopulate ran); the empty
            # surface is the ``enabled=False`` gate, not the absence of operations.
            assert operation_registry.has("list_system_kinds")

    asyncio.run(run())


def _skeleton_manifest() -> Manifest:
    """A manifest whose ONLY tool source is the operation projection: no builtin
    tool/agent/mcp modules named, ``api_tools`` enabled, and the real skeleton
    system-kinds router mounted so the projected op carries its route template."""
    return Manifest.model_validate(
        {
            "routers_modules": ["tai42_skeleton.routers.system_kinds"],
            "api_tools": {"enabled": True},
            # "none" keeps the surface to just the system-kinds router under test.
            "default_routers": "none",
        }
    )


def test_skeleton_operation_projects_at_boot():
    """A REAL skeleton leaf operation is projected as an MCP tool after start()."""

    async def run():
        async with app.app_context(_skeleton_manifest()):
            tools = await app.tools.get_tools()

            # The real skeleton leaf op is on the projected MCP surface.
            assert "list_system_kinds" in tools
            assert tools["list_system_kinds"].tags == {"system"}
            # A safe GET carries no destructive hint.
            assert getattr(tools["list_system_kinds"].annotations, "destructiveHint", None) in (None, False)
            # A destructive skeleton op projects WITH the destructive hint.
            set_annotations = tools["set_tool_extensions"].annotations
            assert set_annotations is not None
            assert set_annotations.destructiveHint is True

            # The system-kinds router re-attached its template + method to the SAME
            # registered record the projection read — proving the repopulate ran
            # before the routers, so the record the projection and authz hold is the
            # one the route decorates.
            md = operation_registry.get("list_system_kinds")
            assert md.route_template == "/api/system/kinds"
            assert md.http_method == "GET"

            # The tool-edge authorization governs the projected tool: the middleware
            # is installed and resolves the tool name back to its operation.
            assert any(isinstance(m, AuthzMiddleware) for m in app._fast_mcp.middleware)
            resolved = resolve_dispatch(
                "list_system_kinds",
                {},
                tool_registry=getattr(app, "_tool_registry", None),
                preset_manager=getattr(app, "preset_manager", None),
            )
            assert resolved is not None
            assert resolved.operation.name == "list_system_kinds"

    asyncio.run(run())


def test_skeleton_projection_survives_reload():
    """The skeleton op stays projected across a reload — the clear()+repopulate
    cycle re-registers it, so projection does not silently empty after a reload."""

    async def run():
        async with app.app_context(_skeleton_manifest()):
            assert "list_system_kinds" in await app.tools.get_tools()

            # A reload re-runs start() (clear + repopulate + projection) on a worker
            # thread through the gate exactly as production does.
            await reload_gate.run(lambda: app._update(_skeleton_manifest()))

            tools = await app.tools.get_tools()
            assert "list_system_kinds" in tools
            # The route template survived the reload on the re-registered record, so
            # the tool-edge authorization can still synthesize the concrete path.
            assert operation_registry.get("list_system_kinds").route_template == "/api/system/kinds"
            # The security middleware is not dropped by a reload cycle.
            assert any(isinstance(m, AuthzMiddleware) for m in app._fast_mcp.middleware)

    asyncio.run(run())
