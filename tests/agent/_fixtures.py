"""Agent fixtures for the run-tool synthesis tests.

``EchoFieldsAgent.run`` echoes the sorted names of the kwargs it actually
received, so a test can observe that the synthesized run tool forwards only the
caller-supplied fields (``from_tool_input``'s set-fields-only contract) rather
than every field materialized with its default.
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
