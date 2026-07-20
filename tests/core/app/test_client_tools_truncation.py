"""Verify ``ToolBinding.get_client_tools`` raises on tool names that
collide after the 64-char LangChain-name truncation instead of silently
yielding two client tools with the same name."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tai42_skeleton.app import server as server_module
from tai42_skeleton.tools import binding as binding_module
from tai42_skeleton.tools.binding import ToolBinding

_LONG_PREFIX = "x" * 64


def _fake_tool(name):
    def fn():
        """A tool."""
        return name

    fn.__name__ = "fn"
    fake = MagicMock(spec=binding_module.FunctionTool)
    fake.fn = fn
    return fake


def _app(tool_names):
    binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))
    # Instance attribute shadows the method — the binding under test reads the
    # faked tool set through its own ``get_tools``.
    binding.get_tools = AsyncMock(return_value={n: _fake_tool(n) for n in tool_names})
    return binding


class TestGetClientToolsTruncation:
    def test_distinct_names_truncate_without_error(self):
        app = _app(["short_tool", _LONG_PREFIX + "_a"])
        tools = asyncio.run(app.get_client_tools())
        assert sorted(t.name for t in tools) == sorted(["short_tool", _LONG_PREFIX])

    def test_post_truncation_collision_raises(self):
        app = _app([_LONG_PREFIX + "_a", _LONG_PREFIX + "_b"])
        with pytest.raises(ValueError, match="collide after 64-char truncation"):
            asyncio.run(app.get_client_tools())

    def test_collision_outside_requested_names_is_ignored(self):
        # Only the requested set is checked — an unrequested colliding tool
        # cannot fail an unrelated lookup.
        app = _app([_LONG_PREFIX + "_a", _LONG_PREFIX + "_b", "short_tool"])
        tools = asyncio.run(app.get_client_tools([_LONG_PREFIX + "_a", "short_tool"]))
        assert sorted(t.name for t in tools) == sorted([_LONG_PREFIX, "short_tool"])

    def test_unknown_requested_name_still_raises(self):
        app = _app(["short_tool"])
        with pytest.raises(RuntimeError, match="No such tool"):
            asyncio.run(app.get_client_tools(["missing"]))
