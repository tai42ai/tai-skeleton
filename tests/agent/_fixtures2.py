"""A second agent module registering the SAME name as :mod:`tests.agent._fixtures`.

Listing both modules in one manifest fires the ``@tai_app.agents.agent`` decorator
twice for the name ``echo_fields`` within a single boot — a genuine collision the
agent registry must reject loudly.
"""

from __future__ import annotations

from pydantic import BaseModel
from tai_contract.agent import Agent
from tai_contract.app import tai_app


class OtherInput(BaseModel):
    text: str


@tai_app.agents.agent("echo_fields")
class OtherEchoAgent(Agent):
    tool_name = "echo_fields"
    tool_description = "A second agent claiming the same name."
    ToolInput = OtherInput

    async def run(self, **kwargs) -> str:
        return "other"
