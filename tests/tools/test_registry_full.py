"""Full ``ToolRegistry`` coverage: the structured ``(name, combos)`` register /
base-name unregister path, the derived ``used_extensions`` / ``missing_tools``
sets, and the ``validation`` raise/pass/ignore branches. ``_tools`` is keyed by
BASE tool name mapped to its extension combos — no ``name:ext`` string parsing.
"""

from __future__ import annotations

import pytest

from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.tools.registry import ToolRegistry


def test_requested_tools_is_copied_not_aliased():
    # The caller passes ``manifest.tools_list`` (the same set object); the
    # registry must copy it, or register/unregister would rewrite the manifest.
    source = {"a", "b"}
    reg = ToolRegistry(requested_tools=source, tool_extensions={})
    reg.register_tool("c")
    reg.unregister_tool("a")
    assert source == {"a", "b"}
    assert reg._requested_tools == {"b", "c"}


def test_register_tool_without_extension_tracks_empty_combo():
    # A bare ``name`` records the empty present-check combo — the same tracking
    # the constructor path does — so ``missing_tools``/``validation`` cover
    # runtime-registered tools too.
    reg = ToolRegistry(set(), {})
    reg.register_tool("foo")
    assert "foo" in reg._requested_tools
    assert list(reg.tool_extensions_iterator("foo")) == [[]]
    assert "foo" in reg.missing_tools


def test_register_tool_with_structured_combos_seeds_each_in_order():
    # The bare present-check combo PLUS each attachment combo, in order and with
    # no per-combo dedup — symmetric with the manifest seeding path.
    reg = ToolRegistry(set(), {})
    reg.register_tool("foo", [["ext1"], ["ext1", "ext2"]])
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["ext1"], ["ext1", "ext2"]]
    assert reg.used_extensions == frozenset({"ext1", "ext2"})


def test_used_extensions_extracts_name_from_dict_element():
    # A ``{"name", "config"}`` combo element contributes only its NAME to the
    # used-extension set (the element itself is unhashable), alongside bare names.
    reg = ToolRegistry(set(), {})
    reg.register_tool(
        "foo",
        [[{"name": "ask_external", "config": {"verifier": {"name": "gh"}}}, "monitor"]],
    )
    assert list(reg.tool_extensions_iterator("foo")) == [
        [],
        [{"name": "ask_external", "config": {"verifier": {"name": "gh"}}}, "monitor"],
    ]
    assert reg.used_extensions == frozenset({"ask_external", "monitor"})


def test_register_tool_already_requested_same_combos_is_idempotent():
    # Pre-seeding via the constructor already marks the name requested; a second
    # register_tool carrying the SAME combos is a true no-op — no duplicate accrues.
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["ext1"]]
    reg.register_tool("foo", [["ext1"]])
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["ext1"]]


def test_register_tool_conflicting_combos_raises():
    # A second registration on the same name carrying DIFFERENT combos is a
    # caller bug (the reload path must unregister first), so it raises rather than
    # silently discarding the new combos.
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})
    with pytest.raises(TaiValidationError, match="different extension combos"):
        reg.register_tool("foo", [["ext2"]])
    # The tracked combos are untouched by the rejected call.
    assert list(reg.tool_extensions_iterator("foo")) == [[], ["ext1"]]


def test_unregister_tool_not_requested_is_noop():
    reg = ToolRegistry(set(), {})
    reg.unregister_tool("ghost")  # nothing to remove -> early return
    assert reg._requested_tools == set()


def test_unregister_tool_drops_name_and_its_combos():
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})
    reg.unregister_tool("foo")
    assert "foo" not in reg._requested_tools
    # Base-name pop removes the whole ``_tools`` key, combos and all.
    assert "foo" not in reg._tools


def test_unregister_tool_requested_but_absent_from_tools():
    # A name marked requested but never tracked in ``_tools`` (no seed) is still
    # discarded from ``_requested_tools`` without touching ``_tools``.
    reg = ToolRegistry(set(), {})
    reg._requested_tools.add("foo")
    reg.unregister_tool("foo")
    assert "foo" not in reg._requested_tools
    assert "foo" not in reg._tools


def test_unregister_tool_base_drops_name_and_all_combos():
    reg = ToolRegistry({"foo", "foobar", "bar"}, {"foo": [["ext1"], ["ext2"]]})
    reg.unregister_tool_base("foo")
    # ``foo`` and its combos gone; the unrelated base names survive (base-name
    # teardown never over-matches a different tool).
    assert reg._requested_tools == {"foobar", "bar"}
    assert "foo" not in reg._tools


def test_missing_tools_excludes_backed_tools():
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})
    assert reg.missing_tools == frozenset({"foo"})
    reg.register_extend_tool(tool_name="foo", extend_tool_name="foo_ext1")
    assert reg.missing_tools == frozenset()


def test_validation_raises_lists_missing():
    with pytest.raises(TaiValidationError, match="bar, foo"):
        ToolRegistry({"foo", "bar"}, {}).validation()


def test_validation_passes_when_all_backed():
    reg = ToolRegistry({"foo"}, {})
    reg.register_extend_tool("foo", "foo")
    reg.validation()  # no raise


def test_validation_ignore_suppresses_named_missing():
    ToolRegistry({"foo"}, {}).validation(ignore=frozenset({"foo"}))
