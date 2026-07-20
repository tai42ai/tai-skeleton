"""Agent seam: the uniform ``Agent`` contract, surfaced in the skeleton namespace.

The whole agent shape — the :class:`Agent` ABC with its default
``astream``/``_drain``/``from_tool_input`` bodies, ``AgentInterruptedError``, the
neutral ``PresetSpec``/``SubAgentSpec`` descriptors, and the typed event
vocabulary — is owned once by the contract (:mod:`tai_contract.agent`). The
event producers (the LangGraph ``updates``/``messages`` projection) and the
live-typed sub-agent construction live in the agents runtime; the skeleton adds
no agent impl of its own. These symbols are re-exported here so skeleton code and
its consumers (the app registry, agent test fixtures) import the agent contract
through one cohesive ``tai_skeleton.agent`` namespace that mirrors the contract
package layout.
"""

from tai_contract.agent.base import (
    Agent,
    AgentInterruptedError,
    PresetSpec,
    SubAgentSpec,
)
from tai_contract.agent.events import (
    InterruptFinal,
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    StreamEvent,
    StructuredFinal,
    ToolCallStep,
    ToolResultStep,
)

__all__ = [
    "Agent",
    "AgentInterruptedError",
    "InterruptFinal",
    "MessageDelta",
    "MessageFinal",
    "PresetSpec",
    "ReasoningStep",
    "RunUsage",
    "StreamEvent",
    "StructuredFinal",
    "SubAgentSpec",
    "ToolCallStep",
    "ToolResultStep",
]
