"""Op-level oracles for the agents operations.

The list ops read the live agent binding through the ``_agents_registry`` seam and
render each agent via the contract-only ``_agent_view``. ``list_spec_runnable_agents``
filters on the agent's own ``spec_runnable`` marker, never on a hardcoded name.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tai42_skeleton.agent import Agent
from tai42_skeleton.operations import agents as agent_ops


class _Input(BaseModel):
    prompt: str


class _FakeAgent(Agent):
    tool_name = "faker"
    tool_description = "A fake agent."
    ToolInput = _Input

    async def run(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
        return None


class _RunnableAgent(_FakeAgent):
    spec_runnable = True


class _PlainAgent(_FakeAgent):
    spec_runnable = False


async def test_list_agents_renders_every_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = {"a": _RunnableAgent(), "b": _PlainAgent()}
    monkeypatch.setattr(agent_ops, "_agents_registry", lambda: agents)

    result = await agent_ops.list_agents()

    assert result["total"] == 2
    names = {item["name"] for item in result["items"]}
    assert names == {"a", "b"}
    view = next(item for item in result["items"] if item["name"] == "a")
    assert view["tool_name"] == "faker"
    assert view["description"] == "A fake agent."
    assert view["spec_runnable"] is True
    assert view["input_schema"] == _Input.model_json_schema()


async def test_list_spec_runnable_filters_on_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = {"a": _RunnableAgent(), "b": _PlainAgent()}
    monkeypatch.setattr(agent_ops, "_agents_registry", lambda: agents)

    result = await agent_ops.list_spec_runnable_agents()

    assert result["total"] == 1
    assert [item["name"] for item in result["items"]] == ["a"]


async def test_list_agents_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_ops, "_agents_registry", dict)
    assert await agent_ops.list_agents() == {"items": [], "total": 0}
