"""Elicitation bridge: a tool's ``ctx.elicit()`` from an
in-process caller resolves through the interactions ``ask_user`` channel, with
schema fidelity, accept-or-raise, and no silent dead-end."""

from __future__ import annotations

import asyncio

import pytest
from fastmcp.server.context import Context, set_context
from fastmcp.server.elicitation import AcceptedElicitation
from pydantic import BaseModel

from tai42_skeleton.app.instance import app
from tai42_skeleton.interactions import elicit_bridge
from tai42_skeleton.interactions.helper import InteractionTimeoutError
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.tools import context_bridge


def _bridge_ctx() -> context_bridge.PlatformBridgeContext:
    return context_bridge.PlatformBridgeContext(fastmcp=app.fastmcp)


def _patch_ask_user(monkeypatch, answer=None, *, raises=None):
    """Record the ask_user call and return ``answer`` (or raise ``raises``)."""
    calls: list[dict] = []

    async def fake_ask_user(question, *, answer_format, schema=None, **kwargs):
        calls.append({"question": question, "answer_format": answer_format, "schema": schema})
        if raises is not None:
            raise raises
        return answer

    monkeypatch.setattr(elicit_bridge, "ask_user", fake_ask_user)
    return calls


# -- seam (a): ElicitBridgeContext -------------------------------------------


def test_seam_a_derives_schema_and_maps_to_accepted(monkeypatch):
    # Scalar response_type -> parse_elicit_response_type wraps it as a {value}
    # form schema; the validated answer maps back to AcceptedElicitation.data.
    calls = _patch_ask_user(monkeypatch, answer={"value": 7})
    ctx = _bridge_ctx()

    result = asyncio.run(ctx.elicit("Pick a number", int))

    assert isinstance(result, AcceptedElicitation)
    assert result.data == 7
    assert calls[0]["answer_format"] == "form"
    # Schema fidelity: the derived form schema carries the wrapped scalar field.
    assert "value" in calls[0]["schema"]["properties"]


class _Profile(BaseModel):
    name: str
    age: int


def test_seam_a_model_response_type_round_trips(monkeypatch):
    calls = _patch_ask_user(monkeypatch, answer={"name": "Ada", "age": 36})
    ctx = _bridge_ctx()

    result = asyncio.run(ctx.elicit("Your profile", _Profile))

    assert isinstance(result, AcceptedElicitation)
    assert result.data == _Profile(name="Ada", age=36)
    # The model's own object schema is carried as the form answer-schema.
    assert set(calls[0]["schema"]["properties"]) == {"name", "age"}


def test_seam_a_timeout_raises_no_decline_round_trip(monkeypatch):
    _patch_ask_user(monkeypatch, raises=InteractionTimeoutError("no answer"))
    ctx = _bridge_ctx()

    with pytest.raises(InteractionTimeoutError):
        asyncio.run(ctx.elicit("Pick a number", int))


# -- elicit_bridge_context: an active client context wins --------------------


def test_bridge_context_noop_when_a_context_is_already_active():
    # An elicit-capable client already established a context; the bridge must not
    # override it (that client answers in-client).
    real = Context(fastmcp=app.fastmcp)

    async def go() -> bool:
        from fastmcp.server.dependencies import get_context

        with set_context(real), context_bridge.bridge_context(app.fastmcp):
            return get_context() is real

    assert asyncio.run(go()) is True


def test_bridge_context_pushes_bridge_when_none_active():
    async def go() -> bool:
        with context_bridge.bridge_context(app.fastmcp):
            from fastmcp.server.dependencies import get_context

            return isinstance(get_context(), context_bridge.PlatformBridgeContext)

    assert asyncio.run(go()) is True


# -- integration: run_tool drives ctx.elicit through ask_user ----------------


def test_run_tool_routes_ctx_elicit_through_ask_user(monkeypatch):
    _patch_ask_user(monkeypatch, answer={"value": 42})

    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):

            @app.tools.tool(force=True)
            async def needs_input(ctx: Context) -> int:
                """A tool that elicits a number from the human."""
                answer = await ctx.elicit("How many?", int)
                assert isinstance(answer, AcceptedElicitation)
                return answer.data

            # run_tool is the in-process caller path: no client, so ctx.elicit
            # resolves via the ask_user bridge (seam a).
            assert await app.tools.run_tool("needs_input", {}) == 42

    asyncio.run(run())
