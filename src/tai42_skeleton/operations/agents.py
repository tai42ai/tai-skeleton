"""Agents operations — read the registered-agent surface.

* ``list_agents`` returns every registered agent as ``{name, description,
  tool_name, input_schema, spec_runnable}`` plus a total.
* ``list_spec_runnable_agents`` returns only the authorable agents
  (``spec_runnable`` True), the same shape.

Both bind to the :class:`~tai42_contract.agent.Agent` CONTRACT only (never a concrete
agent implementation) and read the live process agent binding. The agent RUN doors
(``/api/agents/{name}/runs`` and the authored variant) are SSE streams —
transport-shaped, so they stay handlers in the router and project no operation.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.agent import Agent

from tai42_skeleton.app import instance
from tai42_skeleton.operations import operation


def _agents_registry() -> dict[str, Agent]:
    """Every registered agent keyed by registration name — the process app's live
    agent binding."""
    return instance.app.agents.all_agents()


def _agent_view(name: str, agent: Agent) -> dict[str, Any]:
    return {
        "name": name,
        "description": agent.tool_description,
        "tool_name": agent.tool_name,
        # The one schema source: the binding builds the run tool from this exact
        # model, so the list schema equals the run-tool schema by construction.
        "input_schema": agent.ToolInput.model_json_schema(),
        # The implementation's own capability marker — read, never inferred and
        # never keyed off a hardcoded agent name.
        "spec_runnable": agent.spec_runnable,
    }


@operation(summary="List every registered agent", tags=["agents"])
async def list_agents() -> dict:
    items = [_agent_view(name, agent) for name, agent in _agents_registry().items()]
    return {"items": items, "total": len(items)}


@operation(summary="List the spec-runnable (authorable) agents", tags=["agents"])
async def list_spec_runnable_agents() -> dict:
    """Only the authorable agents (``spec_runnable`` True) — the compose UI's
    base-agent picker. Filters on the marker, never on a known agent name; an empty
    list means no authoring is possible."""
    items = [_agent_view(name, agent) for name, agent in _agents_registry().items() if agent.spec_runnable]
    return {"items": items, "total": len(items)}
