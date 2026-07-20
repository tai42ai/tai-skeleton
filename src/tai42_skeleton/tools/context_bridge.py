"""In-process FastMCP Context bridge for the ``run_tool`` path.

An in-process caller (agent, webhook-triggered run, scheduled backend) invokes a
tool with no connected MCP client, so a tool's injected ``ctx.elicit()`` /
``ctx.sample()`` would dead-end. :class:`PlatformBridgeContext` overrides those
two capabilities to route through the platform's own machinery — elicit through
the interactions ``ask_user`` channel, sample through the platform LLM — while
inheriting every other Context capability unchanged.

:func:`bridge_context` pushes this context for an in-process invocation ONLY when
no FastMCP context is already active: a live request context (a real client,
possibly elicitation/sampling capable) wins and is left untouched, so a capable
client still resolves in-client.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastmcp.server.context import Context, set_context
from fastmcp.server.dependencies import get_context
from fastmcp.server.sampling import SamplingResult

from tai42_skeleton.interactions.elicit_bridge import resolve_elicit
from tai42_skeleton.tools.sampling_bridge import platform_sample
from tai42_skeleton.tools.sampling_settings import sampling_settings

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from fastmcp import FastMCP
    from mcp.types import SamplingMessage


class PlatformBridgeContext(Context):
    """A FastMCP Context whose ``elicit`` routes to ``ask_user`` and whose
    ``sample`` falls back to the platform LLM — for the in-process caller path
    where no elicitation/sampling-capable client exists. Every other Context
    capability is inherited unchanged.

    A fresh context is pushed per in-process invocation (one per ``bridge_context``
    push), so ``_sample_calls`` is naturally invocation-scoped: it bounds how many
    ``ctx.sample()`` calls one tool invocation may make through the bridge."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sample_calls = 0

    async def elicit(
        self,
        message: str,
        response_type: Any = None,
        *,
        response_title: str | None = None,
        response_description: str | None = None,
    ) -> Any:
        return await resolve_elicit(
            message,
            response_type,
            response_title=response_title,
            response_description=response_description,
        )

    async def sample(
        self,
        messages: str | Sequence[str | SamplingMessage],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_preferences: Any = None,
        tools: Any = None,
        result_type: Any = None,
        mask_error_details: bool | None = None,
        tool_concurrency: int | None = None,
    ) -> SamplingResult[Any]:
        self._sample_calls += 1
        budget = sampling_settings().max_calls_per_invocation
        if self._sample_calls > budget:
            raise RuntimeError(
                f"ctx.sample() call budget exhausted: {self._sample_calls} calls in one tool "
                f"invocation (TAI_SAMPLING_MAX_CALLS_PER_INVOCATION={budget})"
            )
        text = await platform_sample(
            messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            result_type=result_type,
        )
        return SamplingResult(text=text, result=text, history=[])


@contextmanager
def bridge_context(fastmcp: FastMCP) -> Iterator[None]:
    """Push a :class:`PlatformBridgeContext` for an in-process tool invocation so
    ``ctx.elicit()`` / ``ctx.sample()`` reach the platform's machinery — but ONLY
    when no FastMCP context is already active. A live request context (a real,
    possibly capable client) wins and is left untouched."""
    try:
        get_context()
        active = True
    except RuntimeError:
        active = False

    if active:
        yield
        return
    with set_context(PlatformBridgeContext(fastmcp=fastmcp)):
        yield
