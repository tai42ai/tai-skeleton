"""Behavior of the manifest ``extensions`` attachment map (decoupled from
``include``/``exclude`` selection).

Every case drives the REAL bind path (``app.app_context`` / a live
``ToolRegistry``): ``include``/``exclude`` stay pure selection, and the separate
``extensions`` map attaches a clip-on extension to a SELECTED tool, applied after
selection. A mapped tool that is not selected raises loudly via ``missing_tools``;
a ``"name:ext"`` colon is part of a literal tool name (not an extension
delimiter), so such a name never binds.
"""

from __future__ import annotations

import asyncio

import pytest
from tai_contract.manifest import MCPConfig, TaiMCPConfig

from tai_skeleton.app.instance import app
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.manifest import Manifest
from tai_skeleton.tools.registry import ToolRegistry

_TOOLS_B = "tests.app._fixtures.tools_b"
_EXT_MOD = "tests.app._fixtures.ext_mod"  # registers the WRAPPER ``loud``
_EXT_KINDS = "tests.app._fixtures.ext_kinds"  # marka/markb wrappers, backendx/backendy


async def _clear_server() -> None:
    provider = app._fast_mcp.local_provider
    for tool in list(await provider.list_tools()):
        provider.remove_tool(tool.name)


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior test bound would collide with this
    test's bind under ``on_duplicate="error"``."""
    asyncio.run(_clear_server())
    yield
    asyncio.run(_clear_server())


def _tools_manifest(
    *, include: list[str], extensions: dict[str, list[list[str]]], ext_module: str = _EXT_MOD
) -> Manifest:
    return Manifest.model_validate(
        {
            "extensions_modules": [ext_module],
            "tools": [
                {"title": "fxt", "module": _TOOLS_B, "include": include, "extensions": extensions},
            ],
        }
    )


# -- round-trip ---------------------------------------------------------------


def test_extensions_map_binds_branch_and_survives_dump_reload():
    manifest = _tools_manifest(include=["shout"], extensions={"shout": [["loud"]]})

    # The derived attachment map is available on the loaded model.
    assert manifest.tool_extensions == {"shout": [["loud"]]}

    async def run(m: Manifest) -> set[str]:
        # Clear first: this test enters app_context twice on the shared singleton
        # server, and the second bind would collide under on_duplicate="error".
        await _clear_server()
        async with app.app_context(m):
            return set(await app.tools.get_tools())

    tools = asyncio.run(run(manifest))
    assert {"shout", "shout_loud"} <= tools

    # Dump -> reload: the ``extensions`` config field round-trips and the derived
    # map rebuilds, so the branch still binds.
    reloaded = Manifest.model_validate(manifest.model_dump())
    assert reloaded.tools[0].extensions == {"shout": [["loud"]]}
    assert reloaded.tool_extensions == {"shout": [["loud"]]}
    assert {"shout", "shout_loud"} <= asyncio.run(run(reloaded))


# -- dynamic stays dynamic ----------------------------------------------------


def test_empty_include_with_extensions_stays_dynamic():
    # ``include: []`` is dynamic selection (every module tool binds). The mapped
    # ``shout`` gets its ``loud`` branch; the sibling ``ping`` (no map entry)
    # binds BARE — attaching an extension never froze selection to an allowlist.
    manifest = _tools_manifest(include=[], extensions={"shout": [["loud"]]})

    async def run(m: Manifest) -> set[str]:
        # Clear first: this test enters app_context twice on the shared singleton
        # server, and the second bind would collide under on_duplicate="error".
        await _clear_server()
        async with app.app_context(m):
            return set(await app.tools.get_tools())

    tools = asyncio.run(run(manifest))
    assert {"shout", "shout_loud", "ping"} <= tools
    # No branch grew on the unmapped sibling.
    assert "ping_loud" not in tools

    # Siblings survive a dump->reload (no allowlist flip).
    reloaded = Manifest.model_validate(manifest.model_dump())
    reloaded_tools = asyncio.run(run(reloaded))
    assert {"shout", "shout_loud", "ping"} <= reloaded_tools


# -- surgical MCP reload (replace_mcp) ----------------------------------------


def test_replace_mcp_rebuilds_tool_extensions():
    # The surgical MCP reload swaps the MCP rows and rebuilds the attachment map,
    # so a hot-added MCP row's ``extensions`` entry takes effect on reload.
    manifest = Manifest.model_validate(
        {"mcp": [TaiMCPConfig(title="old", include=[], config=MCPConfig(url="http://x/mcp")).model_dump()]}
    )
    assert manifest.tool_extensions == {}

    hot_added = [
        TaiMCPConfig(
            title="probed",
            include=[],
            extensions={"probed_ping": [["loud"]]},
            config=MCPConfig(type="http", url="http://x/mcp"),
        )
    ]
    manifest.replace_mcp(hot_added)

    # The rebuilt map picks up the hot-added entry, and a registry built from the
    # post-reload manifest carries the combo the MCP tool will bind through.
    assert manifest.tool_extensions == {"probed_ping": [["loud"]]}
    reg = ToolRegistry(manifest.tools_list, manifest.tool_extensions)
    assert list(reg.tool_extensions_iterator("probed_ping")) == [["loud"]]


def test_mcp_tool_binds_with_extension_branch(monkeypatch):
    # An MCP config carrying an ``extensions`` entry for its (normalized) tool
    # name binds the branch through the real probe -> bind path.
    from unittest.mock import AsyncMock

    cfg = TaiMCPConfig(
        title="probed",
        include=[],
        extensions={"probed_ping": [["loud"]]},
        config=MCPConfig(type="http", url="http://x/mcp"),
    )
    fake_tool = type(
        "_FakeMcpTool",
        (),
        {
            "name": "ping",
            "description": "ping",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {},
        },
    )()
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock(return_value=[fake_tool]))
    manifest = Manifest.model_validate({"extensions_modules": [_EXT_MOD], "mcp": [cfg.model_dump()]})

    async def run() -> set[str]:
        async with app.app_context(manifest):
            return set(await app.tools.get_tools())

    tools = asyncio.run(run())
    assert {"probed_ping", "probed_ping_loud"} <= tools


# -- multiple combos ----------------------------------------------------------


def test_multiple_combos_bind_independent_and_stacked_branches():
    # ``{shout: [[marka], [marka, markb]]}`` -> one single-ext branch and one
    # stacked branch (mirrors ``report: [[chain], [chain, batch]]``).
    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_EXT_KINDS],
            "tools": [
                {
                    "title": "fxt",
                    "module": _TOOLS_B,
                    "include": ["shout"],
                    "extensions": {"shout": [["marka"], ["marka", "markb"]]},
                }
            ],
        }
    )

    async def run() -> set[str]:
        async with app.app_context(manifest):
            return set(await app.tools.get_tools())

    tools = asyncio.run(run())
    assert {"shout", "shout_marka", "shout_marka_markb"} <= tools


# -- one-per-kind -------------------------------------------------------------


def test_two_backends_in_one_combo_rejected():
    # A single combo with two BACKEND extensions violates one-per-kind and raises
    # via ``ExtensionRegistry.validate`` at bind (independent combos would not).
    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_EXT_KINDS],
            "tools": [
                {
                    "title": "fxt",
                    "module": _TOOLS_B,
                    "include": ["shout"],
                    "extensions": {"shout": [["backendx", "backendy"]]},
                }
            ],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            pass

    with pytest.raises(TaiValidationError, match=r"Only one 'backend' extension is allowed"):
        asyncio.run(run())


# -- bind order: body locality vs execution relocation ------------------------


def test_locality_wrapper_outside_relocating_backend_rejected():
    # Extensions apply left-to-right, so the later element wraps (sits OUTSIDE)
    # the earlier ones. ``localwrap`` requires body locality; stacked AFTER the
    # relocating ``backendx`` it would wrap the worker-submitting stub in this
    # process and silently never apply — the bind must reject the combo loudly,
    # naming the tool, both extensions, and the required order.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_EXT_KINDS],
            "tools": [
                {
                    "title": "fxt",
                    "module": _TOOLS_B,
                    "include": ["shout"],
                    "extensions": {"shout": [["backendx", "localwrap"]]},
                }
            ],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            pass

    with pytest.raises(TaiValidationError) as exc_info:
        asyncio.run(run())
    message = str(exc_info.value)
    assert "tool 'shout'" in message
    assert "'localwrap'" in message
    assert "'backendx'" in message
    assert "INSIDE" in message


def test_locality_wrapper_inside_relocating_backend_binds():
    # The correct order — the locality-requiring wrapper BEFORE (inside) the
    # relocating backend — binds cleanly: the wrapper travels with the body.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_EXT_KINDS],
            "tools": [
                {
                    "title": "fxt",
                    "module": _TOOLS_B,
                    "include": ["shout"],
                    "extensions": {"shout": [["localwrap", "backendx"]]},
                }
            ],
        }
    )

    async def run() -> set[str]:
        async with app.app_context(manifest):
            return set(await app.tools.get_tools())

    tools = asyncio.run(run())
    assert {"shout", "shout_localwrap", "shout_localwrap_backendx"} <= tools


# -- extensions not selected --------------------------------------------------


def test_extension_on_unselected_tool_raises_missing():
    # ``include: [shout]`` selects only ``shout``; mapping an extension onto the
    # unselected ``ping`` never selects it, so ``ping`` stays unbound and the
    # ``missing_tools`` validation raises.
    manifest = _tools_manifest(include=["shout"], extensions={"ping": [["loud"]]})

    async def run() -> None:
        async with app.app_context(manifest):
            pass

    with pytest.raises(TaiValidationError, match="ping"):
        asyncio.run(run())


# -- suffix is dead -----------------------------------------------------------


def test_colon_suffix_include_is_a_literal_missing_tool():
    # ``include: ["shout:loud"]`` does not attach an extension — the colon is
    # part of a LITERAL tool name that never binds, so ``missing_tools`` raises
    # and no ``shout_loud`` branch appears.
    manifest = _tools_manifest(include=["shout:loud"], extensions={})
    assert manifest.tool_extensions == {}

    async def run() -> None:
        async with app.app_context(manifest):
            pass

    with pytest.raises(TaiValidationError, match="shout:loud"):
        asyncio.run(run())


# -- structured runtime attach (preset-engine path) ---------------------------


def test_register_tool_info_binds_runtime_branch():
    # The preset-engine path: attach an extension to a NOT-YET-bound tool through
    # the structured facade, then bind it — ``bind_tool_func`` reads the combo and
    # grows the branch, exactly as a reload would.
    manifest = _tools_manifest(include=["shout"], extensions={})

    async def run() -> set[str]:
        async with app.app_context(manifest):
            app.tools.register_tool_info("foo", [["loud"]])

            @app.tools.tool(force=True)
            def foo(text: str) -> str:
                """A runtime-attached tool."""
                return text

            return set(await app.tools.get_tools())

    tools = asyncio.run(run())
    assert {"foo", "foo_loud"} <= tools


def test_register_tool_two_independent_combos_not_stacked():
    # ``combos=[[e1], [e2]]`` -> TWO independent single-ext combos, never one
    # stacked ``[e1, e2]``.
    reg = ToolRegistry(set(), {})
    reg.register_tool("foo", [["e1"], ["e2"]])
    combos = list(reg.tool_extensions_iterator("foo"))
    assert combos == [[], ["e1"], ["e2"]]
    assert ["e1", "e2"] not in combos


def test_reregister_after_unregister_first_accrues_no_duplicate_combos():
    # Idempotency of the reload sequence: unregister-first THEN re-register with
    # the same combos leaves exactly one copy — no duplicate accrual.
    reg = ToolRegistry(set(), {})
    reg.register_tool("foo", [["loud"]])
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["loud"]]

    reg.unregister_tool_base("foo")
    reg.register_tool("foo", [["loud"]])
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["loud"]]
