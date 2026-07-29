"""Focused tests for the manifest impl: load from an in-memory dict, derived-map
building, and the include/exclude filter predicates."""

from tai42_skeleton.manifest import Manifest


def _manifest() -> Manifest:
    """A small in-memory manifest exercising include + exclude on tools, an
    agents module, and an MCP server keyed by title."""
    return Manifest.model_validate(
        {
            "tools": [
                {
                    "title": "Math",
                    "module": "pkg.math_tools",
                    "include": ["add", "sub"],
                },
                {
                    "title": "Text",
                    "module": "pkg.text_tools",
                    "exclude": ["danger"],
                },
            ],
            "agents": [
                {"title": "Agents", "module": "pkg.agents", "include": ["planner"]},
            ],
            "mcp": [
                {
                    "title": "remote",
                    "include": ["search"],
                    "config": {"url": "http://localhost:9000/mcp"},
                },
            ],
        }
    )


def test_post_init_builds_derived_maps():
    m = _manifest()

    # config maps keyed as documented: tools/agents by module, mcp by title.
    assert set(m.tools_map) == {"pkg.math_tools", "pkg.text_tools"}
    assert set(m.agents_map) == {"pkg.agents"}
    assert set(m.mcp_map) == {"remote"}

    # tools_list aggregates includes across tools + mcp.
    assert m.tools_list == {"add", "sub", "search"}

    # module -> title lookup.
    assert m.tools_module_title_map["pkg.math_tools"] == "Math"


def test_should_include_tool_respects_include_list():
    m = _manifest()
    # include-list present: only listed names pass.
    assert m.should_include_tool("add", "pkg.math_tools") is True
    assert m.should_include_tool("mul", "pkg.math_tools") is False


def test_should_include_tool_exclude_only_defaults_allow():
    m = _manifest()
    # no include list, exclude only: everything except the excluded name passes.
    assert m.should_include_tool("upper", "pkg.text_tools") is True
    assert m.should_include_tool("danger", "pkg.text_tools") is False


def test_should_include_agent_and_mcp_tool():
    m = _manifest()
    assert m.should_include_agent("planner", "pkg.agents") is True
    assert m.should_include_agent("rogue", "pkg.agents") is False
    # mcp predicate is keyed by title, not module.
    assert m.should_include_mcp_tool("search", "remote") is True
    assert m.should_include_mcp_tool("delete", "remote") is False


def test_should_include_records_decisions_without_polluting_config():
    m = _manifest()
    assert m.should_include_tool("upper", "pkg.text_tools") is True  # included
    assert m.should_include_tool("danger", "pkg.text_tools") is False  # excluded

    # The emitted config keeps the operator's ORIGINAL lists — resolved names are
    # not appended (which would pollute live_manifest's includes).
    text_cfg = m.tools_map["pkg.text_tools"]
    assert text_cfg.include == []
    assert text_cfg.exclude == ["danger"]
    live_text = next(t for t in m.live_manifest.tools if t.module == "pkg.text_tools")
    assert live_text.include == []

    # The decisions are recorded off to the side instead.
    assert "upper" in m.resolved_includes["pkg.text_tools"]
    assert "danger" in m.resolved_excludes["pkg.text_tools"]


def test_find_module_longest_prefix_via_submodule():
    m = _manifest()
    # a submodule of a registered module resolves through the prefix walk.
    assert m.should_include_tool("add", "pkg.math_tools.sub.deep") is True


def test_find_title_longest_prefix_and_fallback():
    m = _manifest()
    assert m.find_title("pkg.math_tools.sub") == "Math"
    # unknown module falls back to the module path itself.
    assert m.find_title("pkg.unknown") == "pkg.unknown"


def test_replace_mcp_rebuilds_maps_and_tools_list():
    from tai42_contract.manifest import TaiMCPConfig

    m = _manifest()
    new_mcp = TaiMCPConfig.model_validate(
        {
            "title": "other",
            "include": ["lookup"],
            "config": {"url": "http://localhost:9100/mcp"},
        }
    )
    m.replace_mcp([new_mcp])

    assert set(m.mcp_map) == {"other"}
    # tools_list rebuilt: tool includes survive, old mcp include gone, new in.
    assert m.tools_list == {"add", "sub", "lookup"}


def test_duplicate_tools_module_rejected():
    import pytest

    with pytest.raises(ValueError, match=r"duplicate module 'pkg.dup'"):
        Manifest.model_validate(
            {
                "tools": [
                    {"title": "One", "module": "pkg.dup", "include": ["a"]},
                    {"title": "Two", "module": "pkg.dup", "include": ["b"]},
                ],
            }
        )


def test_duplicate_mcp_title_rejected():
    import pytest

    with pytest.raises(ValueError, match=r"duplicate title 'remote'"):
        Manifest.model_validate(
            {
                "mcp": [
                    {"title": "remote", "include": ["a"], "config": {"url": "http://localhost:9000/mcp"}},
                    {"title": "remote", "include": ["b"], "config": {"url": "http://localhost:9001/mcp"}},
                ],
            }
        )


def test_derived_maps_are_frozensets():
    m = _manifest()
    assert isinstance(m.include_module_tools_map["pkg.math_tools"], frozenset)
    assert isinstance(m.exclude_module_tools_map["pkg.text_tools"], frozenset)
    assert isinstance(m.include_title_mcp_tools_map["remote"], frozenset)


def test_replace_mcp_clears_stale_resolved_entries():
    from tai42_contract.manifest import TaiMCPConfig

    m = _manifest()
    # Resolve both an include and an exclude decision under the current title.
    assert m.should_include_mcp_tool("search", "remote") is True
    assert m.should_include_mcp_tool("delete", "remote") is False
    assert "search" in m.resolved_includes["remote"]
    assert "delete" in m.resolved_excludes["remote"]

    # A surgical reload narrows the same title to a different include set.
    narrowed = TaiMCPConfig.model_validate(
        {"title": "remote", "include": ["lookup"], "config": {"url": "http://localhost:9000/mcp"}}
    )
    m.replace_mcp([narrowed])

    # Both stale decision maps under the title are cleared — no old name lingers.
    assert "remote" not in m.resolved_includes
    assert "remote" not in m.resolved_excludes
    assert m.should_include_mcp_tool("search", "remote") is False
    assert m.should_include_mcp_tool("lookup", "remote") is True


def test_replace_mcp_drops_resolved_for_vanished_title():
    from tai42_contract.manifest import TaiMCPConfig

    m = _manifest()
    # Resolve decisions under the current title, then replace with a DIFFERENT
    # title so the old one disappears from the mcp list entirely.
    assert m.should_include_mcp_tool("search", "remote") is True
    assert m.should_include_mcp_tool("delete", "remote") is False
    assert "remote" in m.resolved_includes
    assert "remote" in m.resolved_excludes

    other = TaiMCPConfig.model_validate(
        {"title": "other", "include": ["lookup"], "config": {"url": "http://localhost:9100/mcp"}}
    )
    m.replace_mcp([other])

    # The vanished title's resolved entries are gone.
    assert "remote" not in m.resolved_includes
    assert "remote" not in m.resolved_excludes
    assert set(m.mcp_map) == {"other"}
    # Derived-map keys remain plain strings after the rebuild.
    assert all(isinstance(k, str) for k in m.include_title_mcp_tools_map)
    assert all(isinstance(k, str) for k in m.mcp_map)


def test_live_manifest_serializes_user_facing_fields():
    m = _manifest()
    live = m.live_manifest
    dumped = live.model_dump()

    # excluded-from-dump derived maps must not leak into serialized output.
    assert "tools_map" not in dumped
    assert "tools_list" not in dumped
    # the config sections round-trip.
    assert {t["module"] for t in dumped["tools"]} == {"pkg.math_tools", "pkg.text_tools"}
