"""Unit coverage for the shared command helpers in ``cli.commands._common``.

The input parsers and the offline manifest validator each own a usage-error
contract the command wrappers rely on; these exercise those contracts directly,
with no server involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import typer

from tai_skeleton.cli.commands import _common


def test_app_context_rejects_uninitialized_context() -> None:
    # The root callback stashes an ``AppContext`` on ``ctx.obj``; anything else means
    # the command ran without initialization, which must fail loudly.
    ctx = cast("typer.Context", SimpleNamespace(obj=object()))
    with pytest.raises(RuntimeError, match="not initialized"):
        _common.app_context(ctx)


def test_parse_json_object_rejects_invalid_json() -> None:
    with pytest.raises(typer.BadParameter, match="valid JSON"):
        _common.parse_json_object("{not json", param_hint="--x")


def test_parse_json_object_rejects_non_object() -> None:
    with pytest.raises(typer.BadParameter, match="JSON object"):
        _common.parse_json_object("[1, 2]", param_hint="--x")


def test_parse_json_object_accepts_object() -> None:
    assert _common.parse_json_object('{"a": 1}', param_hint="--x") == {"a": 1}


def test_parse_json_value_rejects_invalid_json() -> None:
    with pytest.raises(typer.BadParameter, match="valid JSON"):
        _common.parse_json_value("{not json", param_hint="--x")


def test_parse_json_value_accepts_any_json() -> None:
    assert _common.parse_json_value("[1, 2]", param_hint="--x") == [1, 2]


def test_parse_kwargs_merges_json_and_pairs_with_pair_override() -> None:
    # The base JSON object seeds the mapping; a ``--kw`` pair for the same key wins.
    result = _common.parse_kwargs('{"a": 1, "b": 2}', ["b=3", "c=hello"])
    # ``3`` parses as JSON (an int), while the unquoted ``hello`` falls back to the string.
    assert result == {"a": 1, "b": 3, "c": "hello"}


def test_parse_kwargs_rejects_pair_without_equals() -> None:
    with pytest.raises(typer.BadParameter, match="key=value"):
        _common.parse_kwargs(None, ["noequals"])


def test_echo_stderr_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    _common.echo_stderr("heads up")
    captured = capsys.readouterr()
    assert captured.err.strip() == "heads up"
    assert captured.out == ""


def test_validate_manifest_file_rejects_unreadable_yaml(tmp_path) -> None:
    bad = tmp_path / "manifest.yml"
    bad.write_text("key: [unterminated\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="could not read manifest YAML"):
        _common.validate_manifest_file(str(bad))


def test_validate_manifest_file_rejects_non_mapping(tmp_path) -> None:
    scalar = tmp_path / "manifest.yml"
    scalar.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="YAML mapping"):
        _common.validate_manifest_file(str(scalar))


def test_validate_manifest_file_rejects_invalid_manifest_shape(tmp_path) -> None:
    # A well-formed YAML mapping that violates the ``Manifest`` model surfaces the
    # model's validation error, not a raw traceback.
    invalid = tmp_path / "manifest.yml"
    invalid.write_text("mcp: not-a-list\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="invalid manifest"):
        _common.validate_manifest_file(str(invalid))
