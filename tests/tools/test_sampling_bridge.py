"""Sampling fallback: ``ctx.sample()`` from an in-process
caller (no client sampling capability) falls back to the platform LLM, and the
fallback is explicit and logged, never silent."""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastmcp.server.context import Context, set_context
from fastmcp.server.sampling import SamplingResult
from langchain_core.messages import AIMessage
from mcp.types import SamplingMessage, TextContent

from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.tools import context_bridge, sampling_bridge


class _FakeModel:
    """Records the messages it was invoked with and returns a fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.invoked: list = []
        self.bound_kwargs: dict | None = None

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return self

    async def ainvoke(self, messages):
        self.invoked.append(messages)
        return AIMessage(content=self.reply)


def _patch_model(monkeypatch, reply: str) -> _FakeModel:
    model = _FakeModel(reply)
    monkeypatch.setattr(sampling_bridge, "platform_llm", lambda: model)
    return model


def test_platform_sample_returns_text_and_logs_fallback(monkeypatch, caplog):
    _patch_model(monkeypatch, "the answer")

    with caplog.at_level(logging.INFO):
        out = asyncio.run(sampling_bridge.platform_sample("What is up?"))

    assert out == "the answer"
    # Explicit, logged fallback — never a silent substitution.
    assert any("falling back to the platform LLM" in r.message for r in caplog.records)


def test_platform_sample_passes_system_prompt_and_params(monkeypatch):
    model = _patch_model(monkeypatch, "ok")

    asyncio.run(sampling_bridge.platform_sample("hi", system_prompt="be terse", temperature=0.2, max_tokens=64))

    assert model.bound_kwargs == {"temperature": 0.2, "max_tokens": 64}
    kinds = [type(m).__name__ for m in model.invoked[0]]
    assert kinds == ["SystemMessage", "HumanMessage"]


def test_platform_sample_forwards_sampling_messages(monkeypatch):
    model = _patch_model(monkeypatch, "ok")
    msgs = [
        SamplingMessage(role="user", content=TextContent(type="text", text="q")),
        SamplingMessage(role="assistant", content=TextContent(type="text", text="a")),
    ]

    asyncio.run(sampling_bridge.platform_sample(msgs))

    kinds = [type(m).__name__ for m in model.invoked[0]]
    assert kinds == ["HumanMessage", "AIMessage"]


def test_platform_sample_rejects_tools_and_result_type(monkeypatch):
    _patch_model(monkeypatch, "x")
    with pytest.raises(NotImplementedError, match="tool loop"):
        asyncio.run(sampling_bridge.platform_sample("hi", tools=[lambda: None]))
    with pytest.raises(NotImplementedError, match="structured result_type"):
        asyncio.run(sampling_bridge.platform_sample("hi", result_type=int))


def test_bridge_context_sample_wraps_result(monkeypatch):
    _patch_model(monkeypatch, "wrapped")
    ctx = context_bridge.PlatformBridgeContext(fastmcp=app.fastmcp)

    result = asyncio.run(ctx.sample("hello"))

    assert isinstance(result, SamplingResult)
    assert result.text == "wrapped"
    assert result.result == "wrapped"


def test_bridge_context_noop_when_client_context_active_skips_platform_sample(monkeypatch):
    # A sampling-capable client already established a context; bridge_context must
    # NOT override it, so ctx.sample() resolves in-client and never falls back to
    # the platform LLM. If bridge_context is ever invoked, its platform_llm is a
    # spy that fails loudly rather than being silently reached.
    monkeypatch.setattr(
        sampling_bridge,
        "platform_llm",
        lambda: pytest.fail("platform LLM must not be built when a client context is active"),
    )
    real = Context(fastmcp=app.fastmcp)

    async def go() -> None:
        from fastmcp.server.dependencies import get_context

        with set_context(real), context_bridge.bridge_context(app.fastmcp):
            active = get_context()
            # The client's own context wins — not the platform bridge whose
            # .sample() would fall back to platform_sample / the platform LLM.
            assert active is real
            assert not isinstance(active, context_bridge.PlatformBridgeContext)
            # Its sample is the native client path, not the bridge override, so the
            # platform fallback is bypassed by construction.
            assert type(active).sample is Context.sample

    asyncio.run(go())


def test_platform_sample_binds_token_cap_default_when_no_max(monkeypatch):
    # No caller max_tokens -> the settings-backed cap is bound as the default, so
    # the platform fallback is never an unbounded generation.
    from tai42_skeleton.tools.sampling_settings import SamplingSettings

    model = _patch_model(monkeypatch, "ok")
    monkeypatch.setattr(sampling_bridge, "sampling_settings", lambda: SamplingSettings(max_tokens_per_call=1234))

    asyncio.run(sampling_bridge.platform_sample("hi"))

    assert model.bound_kwargs == {"max_tokens": 1234}


def test_platform_sample_over_cap_raises_naming_env_var(monkeypatch):
    # A caller asking for more than the cap is refused loudly, never silently
    # clamped.
    from tai42_skeleton.tools.sampling_settings import SamplingSettings

    _patch_model(monkeypatch, "ok")
    monkeypatch.setattr(sampling_bridge, "sampling_settings", lambda: SamplingSettings(max_tokens_per_call=100))

    with pytest.raises(ValueError, match="TAI_SAMPLING_MAX_TOKENS_PER_CALL"):
        asyncio.run(sampling_bridge.platform_sample("hi", max_tokens=101))


def test_bridge_context_sample_call_budget(monkeypatch):
    # The per-invocation call budget bounds how many ctx.sample() calls one tool
    # invocation may make; the (budget+1)th is refused loudly.
    from tai42_skeleton.tools.sampling_settings import SamplingSettings

    async def fake_sample(*args, **kwargs):
        return "text"

    monkeypatch.setattr(context_bridge, "platform_sample", fake_sample)
    monkeypatch.setattr(context_bridge, "sampling_settings", lambda: SamplingSettings(max_calls_per_invocation=2))
    ctx = context_bridge.PlatformBridgeContext(fastmcp=app.fastmcp)

    async def go() -> None:
        await ctx.sample("one")
        await ctx.sample("two")
        with pytest.raises(RuntimeError, match="TAI_SAMPLING_MAX_CALLS_PER_INVOCATION"):
            await ctx.sample("three")

    asyncio.run(go())


def test_run_tool_sample_falls_back_to_platform_llm(monkeypatch, caplog):
    _patch_model(monkeypatch, "42 apples")

    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):

            @app.tools.tool(force=True)
            async def summarize(ctx: Context) -> str:
                """A tool that asks its caller's LLM to sample."""
                res = await ctx.sample("How many apples?")
                assert res.text is not None
                return res.text

            with caplog.at_level(logging.INFO):
                out = await app.tools.run_tool("summarize", {})
            assert out == "42 apples"
            assert any("falling back to the platform LLM" in r.message for r in caplog.records)

    asyncio.run(run())
