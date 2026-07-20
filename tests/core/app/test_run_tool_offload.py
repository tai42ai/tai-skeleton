"""``ToolBinding.run_tool`` sync-offload seam (``offload_sync``).

With ``offload_sync=True`` a blocking SYNC tool runs on a worker thread instead
of inline on the event loop, so it cannot starve a co-running task (the
background-run supervisor's liveness refresh). Async tools and the default path
are unaffected and run inline. These drive the real validate/call/extract path
(no mocked adapter) against a mock ``FunctionTool``.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from tai_skeleton.app import server as server_module
from tai_skeleton.tools import binding as binding_module
from tai_skeleton.tools.binding import ToolBinding


def _function_tool(fn):
    tool = MagicMock(spec=binding_module.FunctionTool)
    tool.fn = fn
    return tool


def _async_return(value):
    async def _coro(*_args, **_kwargs):
        return value

    return _coro


def _binding_for(fn) -> ToolBinding:
    binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))
    binding.get_tool = _async_return(_function_tool(fn))
    return binding


async def _count_ticks_during(coro) -> tuple[object, int]:
    """Run ``coro`` while a heartbeat ticks every 10ms; return (result, ticks).

    A non-blocked event loop lets the heartbeat advance; a loop blocked by an
    inline sync call cannot."""
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat reach its first await
    baseline = ticks
    result = await coro
    hb.cancel()
    with pytest.raises(asyncio.CancelledError):
        await hb
    return result, ticks - baseline


def _slow_sync(x: int) -> int:
    """Block the calling thread for 0.3s, then echo+double so the body is proven
    to have run (a value distinct from any echo of the raw argument)."""
    time.sleep(0.3)
    return x * 2


async def test_offload_true_frees_event_loop_for_a_slow_sync_tool():
    binding = _binding_for(_slow_sync)
    result, ticks = await _count_ticks_during(binding.run_tool("slow", {"x": 7}, offload_sync=True))
    assert result == 14
    # The 0.3s block ran on a thread; the loop kept ticking (~30 heartbeats).
    assert ticks > 5


async def test_offload_false_blocks_the_event_loop_inline():
    binding = _binding_for(_slow_sync)
    result, ticks = await _count_ticks_during(binding.run_tool("slow", {"x": 7}))
    assert result == 14
    # Inline on the loop thread: the heartbeat could not advance during the block.
    assert ticks <= 2


async def test_offload_true_leaves_an_async_tool_inline():
    ran_on = {}

    async def async_tool(x: int) -> int:
        ran_on["loop"] = asyncio.get_running_loop()
        return x + 1

    binding = _binding_for(async_tool)
    result = await binding.run_tool("async_tool", {"x": 4}, offload_sync=True)
    assert result == 5
    # An async tool is a coroutine function — never offloaded, so it runs on the
    # loop it was awaited from.
    assert ran_on["loop"] is asyncio.get_running_loop()


def test_validation_wrapper_is_cached_per_resolved_fn_and_offload():
    # The per-tool validation wrapper is built once per (resolved_fn, offload) and
    # cached module-wide, so repeated run_tool calls reuse it and fastmcp's global
    # TypeAdapter cache keeps hitting instead of thrashing on a per-call throwaway.
    from fastmcp.server.dependencies import without_injected_parameters

    from tai_skeleton.tools.binding import _validation_wrapper

    def f(a: int) -> int:
        """f"""
        return a

    resolved = without_injected_parameters(f)
    w1 = _validation_wrapper(resolved, False)
    assert _validation_wrapper(resolved, False) is w1
    # A different offload flag is a distinct, separately-cached wrapper.
    assert _validation_wrapper(resolved, True) is not w1


def test_offload_true_still_resolves_a_ctx_tool():
    # An async ctx tool is never offloaded (it is a coroutine function), so
    # fastmcp Context injection resolves exactly as on the inline path.
    from fastmcp import Context, FastMCP
    from fastmcp.server.context import set_context
    from fastmcp.tools.function_tool import FunctionTool

    async def ctx_tool(x: int, ctx: Context) -> str:
        return f"x={x}|ctx={type(ctx).__name__}"

    binding = ToolBinding(MagicMock(spec=server_module.TaiMCP))
    binding.get_tool = _async_return(FunctionTool.from_function(ctx_tool))
    server = FastMCP("ctx-test")

    async def run() -> str:
        with set_context(Context(fastmcp=server)):
            return await binding.run_tool("ctx_tool", {"x": 5}, offload_sync=True)

    assert asyncio.run(run()) == "x=5|ctx=Context"
