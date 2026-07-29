"""Fake agents for the authored-agent skeleton tests.

The skeleton surface binds to the ``Agent`` contract only, so these stand in for
a real generic tools-agent without depending on any concrete implementation:

* ``authorable_agent`` — ``spec_runnable = True``, its ``ToolInput`` carries the
  composable spec fields, and its ``from_tool_input`` RENAMES ``system_prompt`` ->
  ``system_message`` (so a raw-splat run path would fail to map it). Its ``astream``
  records the kwargs it received and echoes the mapped ``system_message`` back as a
  ``MessageFinal`` so a streamed run is observable.
* ``role_agent`` — a code role-agent: ``spec_runnable`` defaults ``False``, so it is
  runnable but NOT authorable.
* ``aliased_agent`` — registered under one decorator name with a DIFFERENT
  ``tool_name``, to pin the collision guard's registration-name/``tool_name`` union.
* ``locked_agent`` — ``spec_runnable`` left ``False`` but declaring
  ``preset_bakeable_fields = {"secret_config"}``: a non-UI-composable agent that still
  honors one baked field, so baking ``secret_config`` authors while baking any other
  field is rejected. Its ``astream`` records the kwargs it received.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field, model_validator
from tai42_contract.agent import Agent
from tai42_contract.agent.events import MessageFinal, StreamEvent
from tai42_contract.app import tai42_app


class _AuthorableInput(BaseModel):
    """The composable spec-field set an authored preset bakes, plus the run-time
    ``user_message`` an author deliberately leaves unbaked and ``count`` (a typed
    field used to pin ``fixed_kwargs`` type validation).

    The nested ``presets`` / ``subagents`` are typed as plain dicts so the auto-tool
    binding's ``model_json_schema()`` stays JSON-able (the contract ``PresetSpec`` /
    ``SubAgentSpec`` carry non-JSON fields); the authoring reference validation reads
    them as raw dicts regardless of this typing."""

    user_message: str = ""
    system_prompt: str = ""
    # Present alongside system_prompt (as on the real tools-agent): system_prompt maps
    # to this run kwarg, so setting both is the conflict from_tool_input rejects.
    system_message: str = ""
    tool_names: list[str] = []
    presets: list[dict[str, Any]] = []
    subagents: list[dict[str, Any]] = []
    strategy: str | None = None
    count: int = 1
    # A constrained field: its ``ge=0`` bound lives in the field metadata (not the
    # bare annotation), so it pins that ``fixed_kwargs`` validation honors declared
    # pydantic constraints, not just the field type.
    bounded: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _reject_sentinel(self) -> _AuthorableInput:
        """A cross-field (model-level) rejection: a model-level ``ValidationError``
        carries the WHOLE combined input as its ``input_value`` (including the baked
        ``fixed_kwargs``), so a run over this sentinel pins that the authored-run 400
        surfaces the failure WITHOUT echoing that input — otherwise a baked secret
        would leak to the runner. Fires only on a unique sentinel ``user_message``, so
        it never trips the other authoring/run tests."""
        if self.user_message == "boom-model-error":
            raise ValueError("model-level rejection for the sentinel user_message")
        return self


@tai42_app.agents.agent("authorable_agent")
class AuthorableAgent(Agent):
    tool_name = "authorable_agent"
    tool_description = "A fake authorable agent."
    ToolInput = _AuthorableInput
    spec_runnable = True

    def __init__(self) -> None:
        self.received_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        data = {name: getattr(validated, name) for name in validated.model_fields_set}
        if "system_prompt" in data:
            # Both map to the system_message run kwarg — a NON-EMPTY system_message
            # alongside system_prompt is a conflict, rejected loudly rather than
            # silently dropping one (mirrors the real tools_agent, so the authored-run
            # route surfaces it as a loud 400). An empty default is simply superseded.
            if data.get("system_message"):
                raise ValueError("set only one of system_prompt or system_message")
            data["system_message"] = data.pop("system_prompt")
        return data

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        self.received_kwargs = kwargs
        yield MessageFinal(text=f"system={kwargs.get('system_message', '')}")


class _RoleInput(BaseModel):
    text: str = ""


@tai42_app.agents.agent("role_agent")
class RoleAgent(Agent):
    tool_name = "role_agent"
    tool_description = "A fake code role-agent (not authorable)."
    ToolInput = _RoleInput

    async def run(self, *, text: str = "", **_: Any) -> str:
        return text


class _LockedInput(BaseModel):
    """A non-UI-composable agent's input: ``secret_config`` is the one field the
    runtime honors as a baked constant, ``user_message`` is the run-time field left
    unbaked (existing but NOT declared bakeable)."""

    secret_config: dict[str, Any] = {}
    user_message: str = ""


@tai42_app.agents.agent("locked_agent")
class LockedAgent(Agent):
    tool_name = "locked_agent"
    tool_description = "A fake non-spec-runnable agent that honors one baked field."
    ToolInput = _LockedInput
    preset_bakeable_fields = frozenset({"secret_config"})

    def __init__(self) -> None:
        self.received_kwargs: dict[str, Any] | None = None

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        self.received_kwargs = kwargs
        yield MessageFinal(text=f"secret={kwargs.get('secret_config')}")


class _ConfigInput(BaseModel):
    """Carries the LangGraph configs a run's ``thread_id``/``checkpoint_id`` ride in:
    the plain one plus a voting agent's judge/voter pair."""

    user_message: str = ""
    langgraph_config: dict[str, Any] = {}
    judge_langgraph_config: dict[str, Any] = {}
    voter_langgraph_config: dict[str, Any] = {}


@tai42_app.agents.agent("config_agent")
class ConfigAgent(Agent):
    """Maps every config field straight through to a run kwarg, which is what the
    bridge reservation guard scans."""

    tool_name = "config_agent"
    tool_description = "A fake agent whose run kwargs carry LangGraph configs."
    ToolInput = _ConfigInput

    def __init__(self) -> None:
        self.received_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        return {name: getattr(validated, name) for name in validated.model_fields_set}

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        self.received_kwargs = kwargs
        yield MessageFinal(text="ran")


@tai42_app.agents.agent("aliased_agent")
class AliasedAgent(Agent):
    # Registered under ``aliased_agent`` but declaring a DIFFERENT ``tool_name`` —
    # the two name sets the collision guard must union.
    tool_name = "aliased_tool_name"
    tool_description = "An agent whose tool_name differs from its registration name."
    ToolInput = _RoleInput

    async def run(self, *, text: str = "", **_: Any) -> str:
        return text
