"""Agent seam tests: a concrete agent imported through ``tai42_skeleton.agent``
satisfies the contract ``Agent`` base, the default ``astream``/``_drain`` bodies
behave per the terminal rule, and the typed events are shaped correctly.

The skeleton adds no agent impl — the package re-exports the contract — so these
exercise the contract behavior through the skeleton namespace that consumers use.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tai42_skeleton import agent as _agent

# These are re-exported through the ``tai42_skeleton.agent`` namespace that
# consumers use; reference them via the module so the tests exercise that seam.
Agent = _agent.Agent
AgentInterruptedError = _agent.AgentInterruptedError
InterruptFinal = _agent.InterruptFinal
MessageFinal = _agent.MessageFinal
StreamEvent = _agent.StreamEvent
StructuredFinal = _agent.StructuredFinal
ToolCallStep = _agent.ToolCallStep


class _Options(BaseModel):
    loud: bool = False


class _EchoInput(BaseModel):
    text: str
    times: int = 1
    options: _Options = _Options()


class _EchoAgent(Agent):
    """A minimal non-streaming agent: implements ``run`` only and inherits the
    free ``astream``/``_drain`` defaults."""

    tool_name = "echo"
    tool_description = "Echo text a number of times."
    ToolInput = _EchoInput

    async def run(self, *, text: str = "", times: int = 1, **_: Any) -> str:
        return text * times


def test_concrete_agent_conforms_to_contract() -> None:
    """An agent that implements ``run`` is a genuine contract ``Agent``."""
    assert isinstance(_EchoAgent(), Agent)


def test_incomplete_agent_cannot_instantiate() -> None:
    """``run`` is the one abstract method — an agent that omits it is not a valid
    ``Agent`` and the ABC refuses to instantiate it."""

    class _Partial(Agent):  # no run()
        tool_name = "partial"

    with pytest.raises(TypeError):
        _Partial()  # pyright: ignore[reportAbstractUsage]


def test_from_tool_input_maps_set_fields_only() -> None:
    """``from_tool_input`` passes through exactly the fields the caller set,
    keeping nested pydantic values as instances (not flattened to dicts)."""
    validated = _EchoInput(text="hi", times=3)
    assert _EchoAgent.from_tool_input(validated) == {"text": "hi", "times": 3}

    defaulted = _EchoInput(text="hi")
    # ``times`` was left at its default, so it is not forwarded.
    assert _EchoAgent.from_tool_input(defaulted) == {"text": "hi"}


def test_from_tool_input_keeps_nested_pydantic_as_instance() -> None:
    """A set nested-pydantic field is forwarded as a model instance, not flattened
    to a dict — ``run`` and its resolvers read it by attribute."""
    validated = _EchoInput(text="hi", options=_Options(loud=True))
    forwarded = _EchoAgent.from_tool_input(validated)
    assert isinstance(forwarded["options"], _Options)
    assert forwarded["options"].loud is True


async def test_default_astream_emits_message_final_for_str_result() -> None:
    """The free ``astream`` over ``run`` yields a single terminal. A plain-text
    (``str``) ``run`` result is a ``MessageFinal`` carrying that text."""
    events = [e async for e in _EchoAgent().astream(text="ab", times=2)]
    assert len(events) == 1
    only = events[0]
    assert isinstance(only, MessageFinal)
    assert only.final is True
    assert only.text == "abab"


async def test_default_astream_emits_structured_final_for_non_str_result() -> None:
    """A non-``str`` ``run`` result becomes a ``StructuredFinal`` carrying it."""

    class _StructAgent(_EchoAgent):
        async def run(self, *, text: str = "", times: int = 1, **_: Any) -> Any:
            return {"echo": text * times}

    events = [e async for e in _StructAgent().astream(text="ab", times=2)]
    assert len(events) == 1
    only = events[0]
    assert isinstance(only, StructuredFinal)
    assert only.final is True
    assert only.data == {"echo": "abab"}


async def test_drain_returns_structured_data() -> None:
    """``_drain`` returns the structured payload when the stream produced one."""

    async def gen():
        yield ToolCallStep(tool="echo", args={"text": "x"}, call_id="c1")
        yield StructuredFinal(data={"ok": True})

    agent = _EchoAgent()
    assert await agent._drain(gen()) == {"ok": True}


async def test_drain_returns_message_text_when_no_structured() -> None:
    """With only a ``MessageFinal``, ``_drain`` returns its text."""

    async def gen():
        yield MessageFinal(text="final answer")

    assert await _EchoAgent()._drain(gen()) == "final answer"


async def test_drain_raises_on_interrupt() -> None:
    """An ``InterruptFinal`` in the stream surfaces loudly to a non-streaming
    caller via ``AgentInterruptedError`` carrying the pending interrupts."""

    async def gen():
        yield InterruptFinal(interrupt_id="i1", payload={"q": "?"}, reason="needs input")

    with pytest.raises(AgentInterruptedError) as exc:
        await _EchoAgent()._drain(gen())
    assert exc.value.interrupts[0].interrupt_id == "i1"


async def test_drain_raises_when_response_format_unmet() -> None:
    """When a ``response_format`` is requested but no ``StructuredFinal`` is
    produced, ``_drain`` raises rather than silently falling back to text."""

    async def gen():
        yield MessageFinal(text="just text")

    with pytest.raises(RuntimeError):
        await _EchoAgent()._drain(gen(), response_format=_EchoInput)


def test_event_shapes_carry_stable_discriminators() -> None:
    """The typed events carry stable ``type`` discriminators and the right
    terminal flags; terminals set ``final=True``, steps do not."""
    step = ToolCallStep(tool="echo", args={"a": 1}, call_id="c1")
    assert isinstance(step, StreamEvent)
    assert step.type == "tool_call_step"
    assert step.final is False

    final = StructuredFinal(data=42)
    assert final.type == "structured_final"
    assert final.final is True
