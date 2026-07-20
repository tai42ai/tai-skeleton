"""MCP completions.

The platform serves ``completion/complete`` wherever a plugin registers a
completion handler through the raw server (``app.fastmcp`` — the ungoverned
escape hatch). No new facet wraps it; this test locks that the capability is
reachable end-to-end and that a handler error RAISES (never a silent empty
completion)."""

from __future__ import annotations

import asyncio

import pytest
from fastmcp import Client
from mcp import types

from tai_skeleton.app.server import TaiMCP


def _fresh() -> TaiMCP:
    return TaiMCP(name="completions-under-test", version="1.0")


def test_completion_complete_served_via_fastmcp_escape_hatch():
    a = _fresh()

    @a.fastmcp.prompt
    def greet(city: str) -> str:
        """Greet a city."""
        return f"hi {city}"

    # A plugin registers its completion handler through the raw server.
    @a.fastmcp._mcp_server.completion()
    async def complete(ref, argument, context):
        if argument.name == "boom":
            raise RuntimeError("completion handler exploded")
        return types.Completion(values=[f"{argument.value}-1", f"{argument.value}-2"])

    async def go() -> None:
        async with Client(a.fastmcp) as client:
            result = await client.complete(
                ref=types.PromptReference(type="ref/prompt", name="greet"),
                argument={"name": "city", "value": "lon"},
            )
            assert result.values == ["lon-1", "lon-2"]

            # A handler error surfaces to the caller — never swallowed into an
            # empty completion.
            with pytest.raises(Exception, match="exploded"):
                await client.complete(
                    ref=types.PromptReference(type="ref/prompt", name="greet"),
                    argument={"name": "boom", "value": "x"},
                )

    asyncio.run(go())
