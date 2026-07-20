"""Fake agents for the preset ``output_schema`` front-door tests.

The skeleton surface binds to the ``Agent`` contract only, so these stand in for
real agents without depending on any concrete implementation:

* ``structured_agent`` — its ``ToolInput`` advertises ``response_format`` (the
  forced-structured-output lever), so a preset ``output_schema`` bakes it. Its
  ``astream`` echoes the baked ``response_format``'s ``title`` back inside a
  ``StructuredFinal`` when a ``response_format`` is present, so the baked value
  (and the injected title) is observable through a real run.
* ``plain_agent`` — its ``ToolInput`` has NO ``response_format`` field (the
  voting-agent shape): a preset ``output_schema`` over it must be rejected at
  authoring, never baked onto a missing parameter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from tai_contract.agent import Agent
from tai_contract.agent.events import MessageFinal, StreamEvent, StructuredFinal
from tai_contract.app import tai_app


class _StructuredInput(BaseModel):
    user_message: str = ""
    response_format: dict[str, Any] | None = None


@tai_app.agents.agent("structured_agent")
class StructuredAgent(Agent):
    tool_name = "structured_agent"
    tool_description = "A fake agent that advertises response_format and forces structured output."
    ToolInput = _StructuredInput
    spec_runnable = True

    def __init__(self) -> None:
        self.received_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        return {name: getattr(validated, name) for name in validated.model_fields_set}

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        self.received_kwargs = kwargs
        response_format = kwargs.get("response_format")
        if response_format is not None:
            # Echo the baked response_format's title so the injected/preserved title
            # is observable, plus a field that conforms to the authored schema.
            yield StructuredFinal(
                data={"echoed_title": response_format.get("title"), "answer": kwargs.get("user_message", "")}
            )
        else:
            yield MessageFinal(text=kwargs.get("user_message", ""))


class _PlainInput(BaseModel):
    user_message: str = ""


@tai_app.agents.agent("plain_agent")
class PlainAgent(Agent):
    tool_name = "plain_agent"
    tool_description = "A fake agent with no response_format (the voting-agent shape)."
    ToolInput = _PlainInput

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        yield MessageFinal(text=kwargs.get("user_message", ""))
