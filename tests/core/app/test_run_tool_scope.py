"""``ToolBinding.run_tool`` is pure dispatch — it opens NO span. Tool spans
are owned by the flow interrupt-resume seam and the agent callbacks; a plain
``run_tool`` (standalone / hook / worker) is untraced unless a tool opts into
the ``:monitor`` extension.

This drives the real validate/call/extract path (no mocked adapter) so tool-body
extraction is genuinely exercised, against a mock monitoring backend whose writer
records calls installed through the process registry — proving dispatch touches
neither ``start_span`` nor ``record_span``.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from tai_skeleton.app import server as server_module
from tai_skeleton.monitoring import get_monitoring, init_monitoring, reset_monitoring
from tai_skeleton.tools import binding as binding_module
from tai_skeleton.tools.binding import ToolBinding


@pytest.fixture
def recording_monitoring():
    """Install a mock monitoring backend whose writer records calls for the
    duration of the test, then reset so it cannot leak into the next test."""
    writer = MagicMock()
    monitoring = MagicMock()
    monitoring.writer = writer
    init_monitoring(monitoring)
    try:
        yield writer
    finally:
        reset_monitoring()


def _function_tool(fn):
    tool = MagicMock(spec=binding_module.FunctionTool)
    tool.fn = fn
    return tool


class TestRunToolMonitoring:
    def test_run_tool_extracts_body_result_without_opening_a_span(self, recording_monitoring):
        writer = recording_monitoring

        def my_tool(x: int) -> dict:
            """Echo + double the input."""
            # Returns a value distinct from any echo of the raw arguments, so the
            # assertion proves the tool body actually ran and its result was
            # extracted (not merely the input passed through).
            return {"echoed": x, "doubled": x * 2}

        binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))
        binding.get_tool = _async_return(_function_tool(my_tool))

        result = asyncio.run(binding.run_tool("my_tool", {"x": 5}))

        # The real adapter validated {"x": 5}, drove the body, and the body's
        # value was extracted + json-normalized.
        assert result == {"echoed": 5, "doubled": 10}

        # The recording backend the code would reach IF it traced was installed,
        # yet dispatch opened no span.
        assert get_monitoring().writer is writer
        writer.start_span.assert_not_called()
        writer.record_span.assert_not_called()


def test_run_tool_resolves_fastmcp_context_injection():
    # A tool with a fastmcp ``Context`` param: FastMCP transforms it into a
    # ``Depends(get_context)`` injection at registration. run_tool must invoke the
    # injected-param WRAPPER (which resolves the current Context), not the raw fn
    # — the raw fn would receive the Depends sentinel as ``ctx`` (or raise).
    from fastmcp import Context, FastMCP
    from fastmcp.server.context import set_context
    from fastmcp.tools.function_tool import FunctionTool

    async def ctx_tool(x: int, ctx: Context) -> str:
        # Return ctx's concrete type so an injected real Context is observable.
        return f"x={x}|ctx={type(ctx).__name__}"

    function_tool = FunctionTool.from_function(ctx_tool)
    binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))
    binding.get_tool = _async_return(function_tool)

    server = FastMCP("ctx-test")

    async def run():
        with set_context(Context(fastmcp=server)):
            return await binding.run_tool("ctx_tool", {"x": 5})

    assert asyncio.run(run()) == "x=5|ctx=Context"


def _async_return(value):
    async def _coro(*_args, **_kwargs):
        return value

    return _coro
