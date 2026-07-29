"""Agent fixtures for the run-tool synthesis tests.

``EchoFieldsAgent.run`` echoes the sorted names of the kwargs it actually
received, so a test can observe that the synthesized run tool forwards only the
caller-supplied fields (``from_tool_input``'s set-fields-only contract) rather
than every field materialized with its default.

``NestedToolsAgent.run`` resolves its own tools BY NAME from the process-global
tool facet mid-turn and invokes one — the resolution a real agent (and any
subagent it spawns) performs for itself, which no wrapping at the agent's own
call site can reach. It is how a test observes what the shared tool-dispatch seam
does to a tool an agent picked up on its own.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app


class EchoInput(BaseModel):
    text: str
    times: int = 1
    note: str = "unset"


@tai42_app.agents.agent("echo_fields")
class EchoFieldsAgent(Agent):
    tool_name = "echo_fields"
    tool_description = "Echo which fields were forwarded."
    ToolInput = EchoInput

    async def run(self, **kwargs) -> str:
        return ",".join(sorted(kwargs))


class PresetSpecLike(BaseModel):
    base_tool: str
    fixed_kwargs: dict[str, Any] = {}


class SubAgentSpecLike(BaseModel):
    name: str
    prompt: str = ""


class InlineSkillLike(BaseModel):
    name: str
    content: str


class NestedInput(BaseModel):
    """A ``ToolInput`` carrying nested pydantic-model fields, standing in for the
    real ``tools_agent`` / ``deep_agent`` inputs (the skeleton binds the ``Agent``
    contract only and never imports ``tai42_agents``).

    ``presets`` / ``subagents`` / ``inline_skills`` each nest a model, so
    ``model_json_schema`` emits them as ``$defs`` refs — the shape an extension
    branch must preserve when it composes over the synthesized run tool."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="A required scalar field.")
    times: int = 1
    presets: list[PresetSpecLike] | None = Field(default=None, description="Nested preset specs.")
    subagents: list[SubAgentSpecLike] | None = None
    inline_skills: list[InlineSkillLike] | None = None


@tai42_app.agents.agent("nested_fields")
class NestedFieldsAgent(Agent):
    tool_name = "nested_fields"
    tool_description = "Echo which nested fields were forwarded."
    ToolInput = NestedInput

    async def run(self, **kwargs) -> str:
        return ",".join(sorted(kwargs))


class NestedToolsInput(BaseModel):
    """The tool an agent resolves for itself mid-turn, plus the arguments it invokes
    it with."""

    tool_name: str
    arguments: dict[str, Any] = {}


@tai42_app.agents.agent("nested_tools")
class NestedToolsAgent(Agent):
    tool_name = "nested_tools"
    tool_description = "Resolve one tool by name mid-turn and invoke it."
    ToolInput = NestedToolsInput

    async def run(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Resolve ``tool_name`` from the process-global facet DURING the turn and
        invoke it. Nothing is captured up front, so the only place a decision can reach
        this call is the shared dispatch seam."""
        [tool] = await tai42_app.tools.get_client_tools([tool_name])
        return await tool.ainvoke(arguments or {})


class ConfigInput(BaseModel):
    """A ``ToolInput`` carrying a langgraph config mapping — the thread-scoping vector a
    caller reaches the run tool with, standing in for the real ``tools_agent`` input."""

    text: str
    langgraph_config: dict[str, Any] | None = None


@tai42_app.agents.agent("config_fields")
class ConfigFieldsAgent(Agent):
    tool_name = "config_fields"
    tool_description = "Echo the thread id its config carries."
    ToolInput = ConfigInput

    async def run(self, **kwargs) -> str:
        config = kwargs.get("langgraph_config") or {}
        return str(config.get("configurable", {}).get("thread_id", ""))
