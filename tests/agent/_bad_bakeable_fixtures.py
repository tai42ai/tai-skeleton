"""A fake agent whose ``preset_bakeable_fields`` names a field that is NOT on its
``ToolInput`` — used to pin the registration-time subset check that rejects a
silently-dead declaration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tai_contract.agent import Agent
from tai_contract.app import tai_app


class _BadInput(BaseModel):
    known: str = ""


@tai_app.agents.agent("bad_bakeable_agent")
class BadBakeableAgent(Agent):
    tool_name = "bad_bakeable_agent"
    tool_description = "Declares a preset_bakeable field that is not on its ToolInput."
    ToolInput = _BadInput
    preset_bakeable_fields = frozenset({"ghost_field"})

    async def run(self, **_: Any) -> str:
        return ""
