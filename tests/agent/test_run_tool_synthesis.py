"""The synthesized agent ``run`` tool.

The run tool advertises the agent's ``ToolInput`` schema out-of-band (the
concrete per-field signature exists for extension composition, so ``parameters``
is set explicitly) and honors ``from_tool_input``'s set-fields-only contract:
only the caller-supplied fields reach ``run``, not every field materialized with
its default (an omitted optional arrives as the ``_UNSET`` sentinel and is
stripped). Agent run tools ARE extension-eligible — a manifest extension combo
targeting an agent name binds a branch composed over that concrete signature.
"""

from __future__ import annotations

import asyncio
import copy
import warnings
from typing import cast

import pytest
from fastmcp.tools.base import Tool
from fastmcp.tools.function_tool import FunctionTool
from pydantic.json_schema import PydanticJsonSchemaWarning

from tai42_skeleton.agent.thread_reservation import ReservedThreadNamespaceError
from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.tools.binding import _derive_input_schema


@pytest.fixture(autouse=True)
def _clean_server():
    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {"agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["echo_fields"]}]}
    )


def _nested_manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "agents": [
                {
                    "title": "agents",
                    "module": "tests.agent._fixtures",
                    "include": ["echo_fields", "nested_fields"],
                }
            ]
        }
    )


def _cache_manifest(*agents: str) -> Manifest:
    # Attach the real toolbox ``cache`` WRAPPER to each agent name; ``shout`` is
    # only present because a manifest ``tools:`` entry needs a selected tool.
    return Manifest.model_validate(
        {
            "extensions_modules": ["tai42_toolbox.extensions.cache"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {agent: [["cache"]] for agent in agents},
                }
            ],
            "agents": [
                {"title": "agents", "module": "tests.agent._fixtures", "include": list(agents)},
            ],
        }
    )


def _derive_under_ignore(tool: Tool) -> dict:
    # An agent run tool's concrete signature carries the ``_UNSET`` sentinel as the
    # default of every optional parameter; deriving its JSON schema deliberately
    # excludes that non-serializable default (a benign PydanticJsonSchemaWarning).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PydanticJsonSchemaWarning)
        return _derive_input_schema(cast(FunctionTool, tool).fn)


def test_run_tool_advertises_typed_schema_and_forwards_only_set_fields():
    async def run() -> None:
        async with app.app_context(_manifest()):
            # list_tools advertises the ToolInput fields — NOT an empty **kwargs
            # schema (which would advertise no fields at all).
            tool = await app.tools.get_tool("echo_fields")
            assert set(tool.parameters.get("properties", {})) == {"text", "times", "note"}
            assert set(tool.parameters.get("required", [])) == {"text"}

            # Only the caller-supplied field is forwarded to run; the omitted
            # defaults (``times``, ``note``) are NOT — set-fields-only.
            assert await app.tools.run_tool("echo_fields", {"text": "hi"}) == "text"
            assert await app.tools.run_tool("echo_fields", {"text": "hi", "times": 2}) == "text,times"

    asyncio.run(run())


def test_client_tool_advertises_agent_tool_input_schema_and_runs():
    async def run() -> None:
        async with app.app_context(_manifest()):
            [client_tool] = await app.tools.get_client_tools(["echo_fields"])

            # Surfaced as a client subtool, the agent run tool advertises its real
            # ToolInput FIELDS via its explicit ``.parameters`` (the concrete
            # signature carries sentinel defaults langchain cannot round-trip).
            assert set(client_tool.args) == {"text", "times", "note"}

            # And it still executes end-to-end through the client tool: the
            # permissive runnable forwards set-fields only (``times`` omitted → not
            # forwarded), so the body reports only ``text``.
            assert await client_tool.ainvoke({"text": "hi"}) == "text"

    asyncio.run(run())


def test_set_fields_only_preserved_for_scalar_nested_and_unset_fields():
    async def run() -> None:
        async with app.app_context(_nested_manifest()):
            # Scalar-only subset.
            assert await app.tools.run_tool("nested_fields", {"text": "hi"}) == "text"
            # A nested-model field set alongside a scalar forwards exactly that
            # subset; the unset scalar/nested fields (``times``, ``subagents``,
            # ``inline_skills``) are NOT forwarded.
            forwarded = await app.tools.run_tool("nested_fields", {"text": "hi", "presets": [{"base_tool": "x"}]})
            assert forwarded == "presets,text"

    asyncio.run(run())


def test_agent_base_tool_advertises_exact_model_schema():
    async def run() -> None:
        async with app.app_context(_nested_manifest()):
            from tests.agent._fixtures import EchoInput, NestedInput

            echo = await app.tools.get_tool("echo_fields")
            nested = await app.tools.get_tool("nested_fields")
            # The base tool's advertised input schema DEEP-EQUALS the model schema
            # — exact by construction (set out-of-band on ``parameters``).
            assert echo.parameters == EchoInput.model_json_schema()
            assert nested.parameters == NestedInput.model_json_schema()

    asyncio.run(run())


def test_branch_schema_equals_base_plus_extension_params():
    async def run() -> None:
        async with app.app_context(_cache_manifest("echo_fields")):
            from tests.agent._fixtures import EchoInput

            base = await app.tools.get_tool("echo_fields")
            branch = await app.tools.get_tool("echo_fields_cache")

            base_derived = _derive_under_ignore(base)
            branch_derived = _derive_under_ignore(branch)

            # The branch derives base params PLUS the wrapper's own ``exp`` control.
            assert set(branch_derived["properties"]) == set(base_derived["properties"]) | {"exp"}

            # Branch MINUS the extension's param deep-equals the base's derived
            # schema — the branch carries the agent's exact input contract.
            without_exp = copy.deepcopy(branch_derived)
            without_exp["properties"].pop("exp")
            if "exp" in without_exp.get("required", []):
                without_exp["required"].remove("exp")
            assert without_exp == base_derived

            # The ONE documented allowance: the derived schema omits the ``default``
            # key the model schema carries for optional fields (the sentinel
            # suppresses it) — and NO agent field gains a garbage default. The only
            # property carrying a default is the wrapper's own ``exp``.
            model_schema = EchoInput.model_json_schema()
            assert "default" in model_schema["properties"]["times"]
            assert "default" not in base_derived["properties"]["times"]
            with_defaults = {name for name, prop in branch_derived["properties"].items() if "default" in prop}
            assert with_defaults == {"exp"}

    asyncio.run(run())


def test_nested_models_survive_as_defs_on_a_branch():
    async def run() -> None:
        async with app.app_context(_cache_manifest("nested_fields")):
            branch = await app.tools.get_tool("nested_fields_cache")
            derived = _derive_under_ignore(branch)

            # The nested pydantic-model fields survive as ``$defs`` refs through the
            # branch composition.
            assert set(derived.get("$defs", {})) == {"PresetSpecLike", "SubAgentSpecLike", "InlineSkillLike"}
            assert derived["properties"]["presets"]["anyOf"][0]["items"]["$ref"] == "#/$defs/PresetSpecLike"
            assert derived["properties"]["subagents"]["anyOf"][0]["items"]["$ref"] == "#/$defs/SubAgentSpecLike"
            assert derived["properties"]["inline_skills"]["anyOf"][0]["items"]["$ref"] == "#/$defs/InlineSkillLike"

    asyncio.run(run())


def test_agent_registration_rejects_stray_preset_bakeable_field():
    # A ``preset_bakeable_fields`` entry that is not a real ``ToolInput`` field could
    # never pass the preset route's unknown-field check, so registration rejects it
    # loudly (naming the stray field) rather than let the declaration lie dormant.
    manifest = Manifest.model_validate(
        {
            "agents": [
                {
                    "title": "bad",
                    "module": "tests.agent._bad_bakeable_fixtures",
                    "include": ["bad_bakeable_agent"],
                }
            ]
        }
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="ghost_field"):
            async with app.app_context(manifest):
                pass

    asyncio.run(run())


def test_agent_run_tool_accepts_extension_combo():
    # A manifest extension combo targeting an agent name now binds a branch composed
    # over the run tool's concrete signature: the base is untouched (advertising its
    # exact ToolInput schema) and the branch runs the agent behind the wrapper.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_mod"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"echo_fields": [["loud"]]},
                }
            ],
            "agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["echo_fields"]}],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            from tests.agent._fixtures import EchoInput

            tools = await app.tools.get_tools()
            assert {"echo_fields", "echo_fields_loud"} <= set(tools)

            # Base still advertises the exact model schema; the branch composed a
            # real signature (not an open bag).
            base = await app.tools.get_tool("echo_fields")
            assert base.parameters == EchoInput.model_json_schema()

            # Both run: the base forwards set-fields-only, and the branch runs the
            # agent behind the ``loud`` wrapper (whose sync body passes the agent's
            # awaited result through). End-to-end transformer/wrapper execution over
            # an agent is exercised by the cache+chain test below.
            assert await app.tools.run_tool("echo_fields", {"text": "hi"}) == "text"
            assert await app.tools.run_tool("echo_fields_loud", {"text": "hi"}) == "text"

    asyncio.run(run())


def test_cache_and_chain_compose_over_agent_run_tool_across_a_real_run():
    # A WRAPPER (cache) and a TRANSFORMER (chain) compose over the same agent run
    # tool and both execute end-to-end.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tai42_toolbox.extensions.cache", "tai42_toolbox.extensions.chain"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"echo_fields": [["cache"], ["chain"]]},
                }
            ],
            "agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["echo_fields"]}],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            from tests.agent._fixtures import EchoInput

            tools = await app.tools.get_tools()
            assert {"echo_fields", "echo_fields_cache", "echo_fields_chain"} <= set(tools)

            # cache (WRAPPER): runs the real agent behind the memoizing branch,
            # preserving set-fields-only; a repeat call serves the memoized value.
            assert await app.tools.run_tool("echo_fields_cache", {"text": "hi"}) == "text"
            assert await app.tools.run_tool("echo_fields_cache", {"text": "hi"}) == "text"

            # chain (TRANSFORMER): runs the agent, jq-transforms its output into
            # args for ``shout``, and returns that tool's result.
            chained = await app.tools.run_tool(
                "echo_fields_chain",
                {
                    "text": "hi",
                    "times": 2,
                    "note": "n",
                    "jq_expression": "{text: .}",
                    "next_tool_name": "shout",
                },
            )
            assert chained == "note,text,times"

            # The base's advertised schema is untouched by the composition — it
            # still byte-equals the model schema (the A↔B advertise seam).
            base = await app.tools.get_tool("echo_fields")
            assert base.parameters == EchoInput.model_json_schema()

    asyncio.run(run())


def test_chain_over_agent_run_tool_omitting_an_optional_preserves_set_fields_only():
    # A chain TRANSFORMER re-dispatches the agent run tool BY NAME; an optional the
    # caller omits must not reach the agent as the _UNSET sentinel (which would fail
    # validation) — the run_tool boundary strips it, so the agent sees only ``text``.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tai42_toolbox.extensions.chain"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"echo_fields": [["chain"]]},
                }
            ],
            "agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["echo_fields"]}],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            chained = await app.tools.run_tool(
                "echo_fields_chain",
                {"text": "hi", "jq_expression": "{text: .}", "next_tool_name": "shout"},
            )
            # Only ``text`` was supplied, so the agent echoed only ``text`` (no sentinel
            # for the omitted ``times``/``note``), and ``shout`` returned it unchanged.
            assert chained == "text"

    asyncio.run(run())


# -- the reserved bridge: thread namespace ------------------------------------


def _config_manifest() -> Manifest:
    return Manifest.model_validate(
        {"agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["config_fields"]}]}
    )


def test_the_run_tool_refuses_a_reserved_bridge_thread():
    # The HTTP run doors are not the only way to reach an agent: the auto-registered run
    # tool is a second door onto the same seam, and the reservation holds there too.
    async def run() -> None:
        async with app.app_context(_config_manifest()):
            with pytest.raises(ReservedThreadNamespaceError, match="thread_id"):
                await app.tools.run_tool(
                    "config_fields",
                    {"text": "hi", "langgraph_config": {"configurable": {"thread_id": "bridge:support:+15550001111"}}},
                )
            with pytest.raises(ReservedThreadNamespaceError, match="checkpoint_id"):
                await app.tools.run_tool(
                    "config_fields",
                    {"text": "hi", "langgraph_config": {"configurable": {"checkpoint_id": "bridge:support:x"}}},
                )

    asyncio.run(run())


def test_the_run_tool_allows_an_unreserved_thread():
    async def run() -> None:
        async with app.app_context(_config_manifest()):
            answer = await app.tools.run_tool(
                "config_fields",
                {"text": "hi", "langgraph_config": {"configurable": {"thread_id": "user-42"}}},
            )
            assert answer == "user-42"

    asyncio.run(run())
