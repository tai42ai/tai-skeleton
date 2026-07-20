"""The ``@tai_app.agent`` decorator auto-generates a JSON ``run`` tool.

An ``agents:`` manifest module is imported at startup; its decorated agent is
registered for in-process ``get_agent`` AND gets a synthesized tool whose
signature mirrors the agent's ``ToolInput`` and whose body drives ``run`` to its
final value. Gating happens via the ``agents:`` section (not the tools
namespace).
"""

import asyncio

from tai_skeleton.app.instance import app
from tai_skeleton.manifest import Manifest

_MANIFEST = {
    "agents": [
        {
            "title": "agents",
            "module": "tests.fixtures.dummy_agent",
            "include": ["dummy_agent"],
        }
    ],
}


def test_agent_decorator_registers_agent_and_autotool():
    async def run():
        async with app.app_context(Manifest.model_validate(_MANIFEST)):
            # The agent is registered for in-process use.
            agent = app.agents.get_agent("dummy_agent")
            assert agent.tool_name == "dummy_agent"

            # A JSON tool of the same name exists.
            tools = await app.tools.get_tools()
            assert "dummy_agent" in tools

            # Its input schema mirrors the agent's ToolInput.
            tool = await app.tools.get_tool("dummy_agent")
            props = set(tool.parameters.get("properties", {}))
            assert props == {"text", "times", "tags", "item"}

            # The tool body drives run() to its final value. Omitting the
            # default_factory field `tags` must NOT raise — it synthesizes to []
            # in the tool signature, not None.
            result = await app.tools.run_tool("dummy_agent", {"text": "ab", "times": 3})
            assert result == "ababab|tags=0|item=NoneType"

            # A nested-model field reaches run() as a model instance, not a dict.
            nested = await app.tools.run_tool("dummy_agent", {"text": "x", "item": {"label": "y"}})
            assert nested == "x|tags=0|item=DummyItem"

    asyncio.run(run())
