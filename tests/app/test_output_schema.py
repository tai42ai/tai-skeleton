"""Branch-tool output-schema propagation.

A tool declaring an object output schema keeps it; a shape-preserving branch
(WRAPPER / BACKEND) inherits it when it declares none of its own; a shape-
changing TRANSFORMER branch does not; and a preserving branch that declares its
own output schema keeps its own.
"""

from __future__ import annotations

import asyncio

import pytest

from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior test bound would collide with this
    test's bind under ``on_duplicate="error"``."""

    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


_BASE_SCHEMA = {
    "properties": {"title": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["title", "score"],
    "type": "object",
}


def _manifest(*combos: str | list[str]) -> Manifest:
    # Every case selects the single base tool ``report`` and attaches the given
    # combos to it. Each arg is one combo: a bare name is a single-extension
    # combo, a list is a stacked combo. No args = base tool only, no extensions.
    normalized = [[c] if isinstance(c, str) else c for c in combos]
    return Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_output"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_out",
                    "include": ["report"],
                    "extensions": {"report": normalized} if normalized else {},
                }
            ],
        }
    )


def test_base_tool_surfaces_its_declared_output_schema():
    async def run() -> None:
        async with app.app_context(_manifest()):
            tool = await app.tools.get_tool("report")
            assert tool.output_schema == _BASE_SCHEMA

    asyncio.run(run())


def test_shape_preserving_wrapper_and_backend_branches_inherit_base_schema():
    async def run() -> None:
        async with app.app_context(_manifest("passw", "passb")):
            wrapper_branch = await app.tools.get_tool("report_passw")
            backend_branch = await app.tools.get_tool("report_passb")
            # WRAPPER and BACKEND both preserve output shape -> inherit the base's
            # object schema across the wrap.
            assert wrapper_branch.output_schema == _BASE_SCHEMA
            assert backend_branch.output_schema == _BASE_SCHEMA

    asyncio.run(run())


def test_shape_changing_transformer_branch_does_not_inherit_base_schema():
    async def run() -> None:
        async with app.app_context(_manifest("listtf")):
            branch = await app.tools.get_tool("report_listtf")
            # The transformer reshapes to a list; forcing the base single-object
            # schema would mis-declare its structured_content, so it must not be
            # inherited.
            assert branch.output_schema != _BASE_SCHEMA

    asyncio.run(run())


def test_preserving_branch_with_own_output_schema_keeps_its_own():
    async def run() -> None:
        async with app.app_context(_manifest("ownw")):
            branch = await app.tools.get_tool("report_ownw")
            # Declares its own object schema -> the gate's "declares none"
            # conjunct is false, so the base schema is not forced over it.
            assert branch.output_schema is not None
            assert branch.output_schema != _BASE_SCHEMA
            assert "kept" in branch.output_schema.get("properties", {})

    asyncio.run(run())


def test_stacked_wrapper_then_transformer_branch_does_not_inherit():
    """A WRAPPER→TRANSFORMER stack reshapes overall, so the fully-stacked branch
    must not inherit the base schema even though its first layer preserves."""

    async def run() -> None:
        async with app.app_context(_manifest(["passw", "listtf"])):
            # The intermediate report_passw (WRAPPER only) still inherits.
            inter = await app.tools.get_tool("report_passw")
            assert inter.output_schema == _BASE_SCHEMA
            # The full stack includes a transformer -> no inheritance.
            full = await app.tools.get_tool("report_passw_listtf")
            assert full.output_schema != _BASE_SCHEMA

    asyncio.run(run())
