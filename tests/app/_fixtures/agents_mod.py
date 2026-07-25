"""Fixture agents module with two agents. A manifest that includes only one
exercises the ``@tai42_app.agent`` gate: the included agent registers + gets its
synthesized run tool; the excluded one (same module, not in ``include``) is
left unregistered."""

from pydantic import BaseModel
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app


class _In(BaseModel):
    text: str = ""


@tai42_app.agents.agent("kept_agent")
class KeptAgent(Agent):
    tool_name = "kept_agent"
    tool_description = "Kept agent."
    ToolInput = _In

    async def run(self, *, text: str = "", **_) -> str:
        return text


@tai42_app.agents.agent("dropped_agent")
class DroppedAgent(Agent):
    tool_name = "dropped_agent"
    tool_description = "Dropped agent."
    ToolInput = _In

    async def run(self, *, text: str = "", **_) -> str:
        return text
