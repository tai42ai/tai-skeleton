"""A dummy :class:`Agent` for the auto-tool-generation test.

Listed as an ``agents:`` manifest module so importing it fires
``@tai_app.agent`` and the skeleton synthesizes its JSON ``run`` tool. Its
``ToolInput`` deliberately carries a ``default_factory`` list and a nested
pydantic field to exercise the two auto-tool seam hazards (a factory field must
synthesize a real default, and a nested model must reach ``run`` as an instance).
"""

from pydantic import BaseModel, Field
from tai_contract.agent import Agent
from tai_contract.app import tai_app


class DummyItem(BaseModel):
    label: str


class DummyInput(BaseModel):
    text: str
    times: int = 1
    tags: list[str] = Field(default_factory=list)
    item: DummyItem | None = None


@tai_app.agents.agent("dummy_agent")
class DummyAgent(Agent):
    tool_name = "dummy_agent"
    tool_description = "Echo text a number of times."
    ToolInput = DummyInput

    async def run(self, *, text: str = "", times: int = 1, tags=(), item=None, **_):
        return f"{text * times}|tags={len(tags)}|item={type(item).__name__}"
