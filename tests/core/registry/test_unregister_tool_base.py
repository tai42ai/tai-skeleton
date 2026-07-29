"""``ToolRegistry.unregister_tool_base`` base-name teardown.

There is no ``name:ext`` store and no prefix scan: a base tool is torn down by
its BASE NAME. ``unregister_tool_base`` drops the base's
selection entry and combos, and clears the base's extension BRANCH tools from
``_extend_tools`` — which it enumerates and RETURNS so the caller can remove each
branch from the MCP provider. Every bound tool records itself in ``_extend_tools``
(including the base, where ``curr_name == orig_name``), so the base SELF-ENTRY is
cleared here but EXCLUDED from the returned branch list — the caller removes the
base separately and must never double-remove it.
"""

from tai42_skeleton.tools.registry import ToolRegistry


def _bind(reg: ToolRegistry, base: str, *branches: str) -> None:
    # Mirror ``ToolBinding.bind_tool``: every bound tool records itself in
    # ``_extend_tools`` as branch-name -> base-name, the base included (its
    # self-entry ``base -> base`` is load-bearing for ``missing_tools``).
    reg.register_extend_tool(base, base)
    for branch in branches:
        reg.register_extend_tool(base, branch)


def test_returns_branch_names_excluding_base_self_entry():
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"], ["ext2"]]})
    _bind(reg, "foo", "foo_ext1", "foo_ext2")

    removed = reg.unregister_tool_base("foo")

    # Both branch names are returned; the base self-entry is NOT.
    assert sorted(removed) == ["foo_ext1", "foo_ext2"]
    assert "foo" not in removed
    # Base + both branches gone from _extend_tools; base gone from _tools/_requested.
    assert reg._extend_tools == {}
    assert "foo" not in reg._tools
    assert "foo" not in reg._requested_tools


def test_zero_extension_returns_empty_and_clears_self_entry():
    reg = ToolRegistry({"foo"}, {})
    _bind(reg, "foo")  # base self-entry only, no branches

    removed = reg.unregister_tool_base("foo")

    assert removed == []
    # The base self-entry is still cleared.
    assert "foo" not in reg._extend_tools
    assert "foo" not in reg._tools
    assert "foo" not in reg._requested_tools


def test_does_not_over_match_other_base_tools():
    # Teardown removes only the named base; other bases (even name-prefix
    # siblings) survive — the match is by ``_extend_tools`` base VALUE, never a
    # string prefix.
    reg = ToolRegistry({"foo", "foo_bar", "foobar"}, {})
    _bind(reg, "foo")
    _bind(reg, "foo_bar", "foo_bar_ext")
    _bind(reg, "foobar")

    reg.unregister_tool_base("foo")

    assert "foo_bar" in reg._requested_tools
    assert "foobar" in reg._requested_tools
    # foo_bar's branch belongs to a different base and survives untouched.
    assert reg._extend_tools.get("foo_bar_ext") == "foo_bar"
    assert reg._extend_tools.get("foobar") == "foobar"
