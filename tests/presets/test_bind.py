"""The preset bind kernel: a hidden/fixed FastMCP transform of a base tool.

Pins the typed-schema rebuild — the remaining arguments keep the base tool's real
schema, each baked ``fixed_kwargs`` key is HIDDEN and FIXED (a caller that passes
it is rejected, never overriding the baked constant), and the preset's
``description`` + categorization ``tags`` are set on the transformed tool. The
kernel is the single point every preset builds through, so these guarantees
reach every preset.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.tools import Tool
from tai_contract.app import TaiApp
from tai_kit.utils.data.json_schema_util import JsonSchemaValidationError

from tai_skeleton.presets.bind import preset_bind


def _base_tool() -> Tool:
    def weather(city: str, units: str = "metric") -> str:
        """Report the weather."""
        return f"{city}:{units}"

    return Tool.from_function(weather, name="weather")


def _dict_base_tool() -> Tool:
    def report(city: str, units: str = "metric") -> dict:
        """Report the weather as a structured object."""
        return {"city": city, "units": units, "temp": 20}

    return Tool.from_function(report, name="report")


def _agent_base_tool() -> Tool:
    def agent_run(city: str, response_format: dict[str, Any] | None = None) -> dict:
        """A stand-in agent run tool that echoes the response_format it was baked with."""
        return {"city": city, "response_format": response_format}

    return Tool.from_function(agent_run, name="agent_run")


class _FakeTools:
    def __init__(self, tool: Tool) -> None:
        self._tool = tool

    async def get_tool(self, key: str) -> Tool:
        assert key == self._tool.name
        return self._tool


class _FakeAgents:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def all_agents(self) -> dict[str, Any]:
        # Only the KEYS are read by the bind kernel (agent-base detection).
        return {name: object() for name in self._names}


class _FakeApp:
    def __init__(self, tool: Tool, agent_names: tuple[str, ...] = ()) -> None:
        self.tools = _FakeTools(tool)
        self.agents = _FakeAgents(agent_names)


def _app(tool: Tool, agent_names: tuple[str, ...] = ()) -> TaiApp:
    return cast("TaiApp", _FakeApp(tool, agent_names))


async def test_bind_hides_baked_key_and_keeps_typed_schema():
    tool = await preset_bind(
        _app(_base_tool()),
        "weather",
        {"units": "imperial"},
        name="paris_weather",
        description="Weather in Paris",
        tags=["geo"],
    )
    assert tool.name == "paris_weather"
    schema = tool.to_mcp_tool().inputSchema
    props = schema.get("properties", {})
    # The baked key is removed from the exposed schema; the remaining argument
    # keeps its real typed schema (name + type), not one opaque ``params`` blob.
    assert "units" not in props
    assert props["city"]["type"] == "string"


async def test_bind_sets_description_and_native_tags():
    tool = await preset_bind(_app(_base_tool()), "weather", {}, name="p", description="Desc", tags=["a", "b"])
    assert tool.description == "Desc"
    assert tool.tags == {"a", "b"}
    # The native tags ride on the serialized wire tool's _meta.fastmcp.tags.
    meta = tool.to_mcp_tool().meta
    assert meta is not None
    assert set(meta["fastmcp"]["tags"]) == {"a", "b"}


async def test_baked_kwarg_is_fixed_and_rejected_when_passed():
    tool = await preset_bind(_app(_base_tool()), "weather", {"units": "imperial"}, name="p")
    # The baked value is used as a fixed constant...
    ok = await tool.run({"city": "paris"})
    assert ok.structured_content == {"result": "paris:imperial"}
    # ...and a caller that passes the baked key is REJECTED, never overriding it.
    with pytest.raises(TypeError):
        await tool.run({"city": "paris", "units": "metric"})


async def test_baked_partial_binds_positional_through_presented_signature():
    # A baked MIDDLE arg: base ``(a, b, c)`` with ``b`` baked presents ``(a, c)``.
    # A positional call must map each value to the param it names in the PRESENTED
    # signature (1->a, 3->c), never onto the hidden baked slot.
    from fastmcp.tools.tool_transform import TransformedTool

    from tai_skeleton.tools.binding import _baked_partial

    def base(a: int, b: int = 0, c: int = 0) -> dict:
        """base"""
        return {"a": a, "b": b, "c": c}

    tool = await preset_bind(_app(Tool.from_function(base, name="base")), "base", {"b": 99}, name="baked")
    # ``preset_bind`` returns a ``TransformedTool`` at runtime (typed as ``Tool``).
    partial = _baked_partial(cast(TransformedTool, tool))

    assert partial(1, 3) == {"a": 1, "b": 99, "c": 3}
    assert partial(a=1, c=3) == {"a": 1, "b": 99, "c": 3}
    # Passing the baked key is still rejected.
    with pytest.raises(TypeError):
        partial(1, b=5)


# -- output_schema front-door: plain-tool DECLARE + VALIDATE -----------------


async def test_plain_tool_output_schema_is_advertised_and_conforming_passes():
    schema = {
        "type": "object",
        "properties": {"temp": {"type": "integer", "minimum": 0}},
        "required": ["temp"],
    }
    tool = await preset_bind(_app(_dict_base_tool()), "report", {}, name="r", output_schema=schema)
    # The authored schema is advertised as the bound tool's output schema.
    assert tool.output_schema == schema
    # A conforming result passes through unchanged.
    ok = await tool.run({"city": "paris"})
    assert ok.structured_content is not None
    assert ok.structured_content["temp"] == 20


async def test_plain_tool_output_schema_violation_raises_on_constraint_keyword():
    # ``minimum`` is a constraint keyword the faithful validator enforces: the tool
    # returns ``temp=20``, which violates ``minimum: 100``.
    schema = {
        "type": "object",
        "properties": {"temp": {"type": "integer", "minimum": 100}},
        "required": ["temp"],
    }
    tool = await preset_bind(_app(_dict_base_tool()), "report", {}, name="r", output_schema=schema)
    with pytest.raises(JsonSchemaValidationError):
        await tool.run({"city": "paris"})


# -- output_schema front-door: agent-base FORCE (bake response_format) --------


async def test_agent_base_bakes_response_format_and_injects_preset_name_as_title():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}  # no title
    tool = await preset_bind(
        _app(_agent_base_tool(), agent_names=("agent_run",)),
        "agent_run",
        {},
        name="my_preset",
        output_schema=schema,
    )
    # Advertised output schema stays the authored, title-free value.
    assert tool.output_schema == schema
    assert "title" not in (tool.output_schema or {})
    # response_format is baked (hidden) — removed from the exposed input schema.
    assert "response_format" not in tool.to_mcp_tool().inputSchema.get("properties", {})
    # The baked response_format carries the preset name injected as its title.
    out = await tool.run({"city": "paris"})
    assert out.structured_content is not None
    baked = out.structured_content["response_format"]
    assert baked["title"] == "my_preset"
    assert baked["properties"] == schema["properties"]


async def test_agent_base_preserves_an_authored_title():
    schema = {"type": "object", "title": "MyAnswer", "properties": {"answer": {"type": "string"}}}
    tool = await preset_bind(
        _app(_agent_base_tool(), agent_names=("agent_run",)),
        "agent_run",
        {},
        name="my_preset",
        output_schema=schema,
    )
    out = await tool.run({"city": "paris"})
    # An authored title is preserved, never overwritten by the preset name.
    assert out.structured_content is not None
    assert out.structured_content["response_format"]["title"] == "MyAnswer"


async def test_agent_base_does_not_double_validate():
    # The agent path attaches NO validation wrapper (the agent's own drain is the
    # only validator). The echo result lacks the required ``answer`` field, so a
    # validation wrapper WOULD raise — the agent path must return it unchanged.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    tool = await preset_bind(
        _app(_agent_base_tool(), agent_names=("agent_run",)),
        "agent_run",
        {},
        name="p",
        output_schema=schema,
    )
    out = await tool.run({"city": "paris"})
    assert out.structured_content is not None
    assert out.structured_content["city"] == "paris"
