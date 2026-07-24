"""``get_client_tools`` exposure of injected-param and preset tools, and the argument
mapping a client-tool call is decided on.

An injected fastmcp ``Context`` param must NOT reach the advertised client-tool
schema (the LLM would be asked to supply it), and the client tool must still run
through the in-process ``bridge_context`` so ``ctx.sample()`` falls back to the
platform LLM — exactly as ``run_tool`` does. A preset (a ``TransformedTool``) is
resolved to its baked partial and exposed too, serving the baked constant and
rejecting a baked key.

The runnable presents a permissive ``*args``/``**kwargs`` front, so a call may arrive
positionally, through a catch-all, or carrying the ``_UNSET`` sentinel.
``_named_call_arguments`` normalizes all of it into the by-parameter-name mapping the
execution-identity decision reads — an argument landing under the wrong name is a
wrongly-refused fire, so each shape is pinned.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import pytest
from fastmcp.server.context import Context

from tai42_skeleton.agent.binding import _UNSET
from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.tools import sampling_bridge
from tai42_skeleton.tools.binding import _named_call_arguments


class _FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


@pytest.fixture(autouse=True)
def _clean_server():
    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


def test_client_tool_strips_injected_ctx_and_runs_via_bridge(monkeypatch, caplog):
    monkeypatch.setattr(sampling_bridge, "platform_llm", lambda: _FakeModel("42 apples"))

    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):

            @app.tools.tool(force=True)
            async def summarize(q: str, ctx: Context) -> str:
                """Summarize a query by asking the caller's LLM to sample."""
                res = await ctx.sample(q)
                assert res.text is not None
                return f"{q}:{res.text}"

            [client_tool] = await app.tools.get_client_tools(["summarize"])

            # The advertised schema INCLUDES the real user param and EXCLUDES the
            # injected fastmcp Context — the LLM is never asked to supply ``ctx``.
            assert "q" in client_tool.args
            assert "ctx" not in client_tool.args

            # And the tool still executes: the in-process bridge resolves
            # ``ctx.sample()`` to the platform LLM fallback.
            with caplog.at_level(logging.INFO):
                out = await client_tool.ainvoke({"q": "How many apples?"})
            assert out == "How many apples?:42 apples"
            assert any("falling back to the platform LLM" in r.message for r in caplog.records)

    asyncio.run(run())


def test_client_tool_exposes_preset_and_bakes_constant(preset_manager_restored):
    async def run() -> None:
        manifest = Manifest.model_validate(
            {"tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather"]}]}
        )
        async with app.app_context(manifest):
            await app.preset_manager.register("paris", "weather", {"units": "imperial"}, [], [], "Paris weather")

            # A preset is a TransformedTool, not a FunctionTool — it is still
            # exposed (resolved via its baked partial), by name and in the full set.
            names = {t.name for t in await app.tools.get_client_tools()}
            assert "paris" in names
            [client_tool] = await app.tools.get_client_tools(["paris"])

            # The baked key is hidden from the advertised schema; the remaining
            # user arg keeps its real type.
            assert "units" not in client_tool.args
            assert "city" in client_tool.args

            # Invoking serves the baked constant...
            out = await client_tool.ainvoke({"city": "paris"})
            assert out == {"city": "paris", "units": "imperial"}

            # ...and the baked key can never override it: the advertised schema
            # excludes it (langchain drops it before the call), and the resolved
            # runnable itself rejects it if passed directly.
            preset = await app.tools.get_tool("paris")
            runnable = app._tool_binding._client_runnable(preset)
            with pytest.raises(TypeError):
                await runnable(city="paris", units="metric")

    asyncio.run(run())


# -- the argument mapping the call is decided on ------------------------------


def _two_params(target: str, mark: str) -> None:
    """An ordinary concrete signature: two named parameters, no catch-all."""


def _keyword_catch_all(target: str, **rest: Any) -> None:
    """A signature whose surplus keywords are collected by a ``**`` catch-all."""


def _positional_catch_all(target: str, *rest: Any) -> None:
    """A signature whose surplus positionals are collected by a ``*`` catch-all."""


def _passthrough(*args: Any, **kwargs: Any) -> None:
    """The all-VAR passthrough a tool advertising an explicit args schema is invoked
    through."""


def _sentinel_defaults(target: str, mark: Any = _UNSET) -> None:
    """A synthesized agent run tool's shape: each optional defaults to the ``_UNSET``
    sentinel so the body forwards set fields only."""


def _named(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return _named_call_arguments(inspect.signature(fn), args, kwargs)


def test_positional_arguments_land_under_their_parameter_names():
    # The decision reads arguments by NAME, so positionals are bound through the
    # signature first.
    assert _named(_two_params, "deploy", "m") == {"target": "deploy", "mark": "m"}
    assert _named(_two_params, "deploy", mark="m") == {"target": "deploy", "mark": "m"}


def test_a_keyword_catch_all_is_flattened_into_the_mapping_it_collected():
    # Left nested, the decision would see ``{"rest": {...}}`` and miss the path argument.
    assert _named(_keyword_catch_all, "deploy", mark="m") == {"target": "deploy", "mark": "m"}


def test_a_positional_catch_all_is_dropped():
    # Values naming no parameter cannot be keyed, so they are no part of the mapping.
    assert _named(_positional_catch_all, "deploy", "x", "y") == {"target": "deploy"}


def test_the_unset_sentinel_is_stripped():
    # langchain materializes defaults, so the sentinel arrives as a real value; left in
    # it lands in the synthesized resource path as ``str(_UNSET)`` and denies the call.
    assert _named(_sentinel_defaults, target="deploy", mark=_UNSET) == {"target": "deploy"}
    assert _named(_sentinel_defaults, target="deploy", mark="m") == {"target": "deploy", "mark": "m"}


def test_a_passthrough_signature_keeps_only_the_named_keywords():
    # Both catch-alls at once: the positional collects a value that names nothing, the
    # keyword catch-all flattens, and the sentinel is stripped from the flattened result.
    assert _named(_passthrough, "unnamed", mark="m", absent=_UNSET) == {"mark": "m"}
