"""Output rendering: JSON vs table, and control-character stripping.

The renderer treats every server string as untrusted — control characters
(which start terminal escape sequences) must never reach the terminal.
"""

from __future__ import annotations

import json

import pytest

from tai_skeleton.cli import render


def test_strip_control_removes_escape_and_control_chars() -> None:
    dirty = "a\x1b[31mred\x1b[0m\x07\x00b"
    cleaned = render.strip_control(dirty)

    assert "\x1b" not in cleaned  # the ESC that arms an ANSI sequence is gone
    assert "\x07" not in cleaned  # BEL
    assert "\x00" not in cleaned  # NUL
    # The inert printable remainder survives; only the control bytes were dropped.
    assert cleaned == "a[31mred[0mb"


def test_render_json_is_parseable_and_escapes_controls() -> None:
    payload = {"name": "tool\x1bx", "items": [1, 2]}
    text = render.render_json(payload)

    parsed = json.loads(text)
    assert parsed == payload
    # A raw ESC never appears in the JSON text — json.dumps escapes it.
    assert "\x1b" not in text


def test_render_table_aligns_columns_and_strips_controls() -> None:
    records = [{"name": "alpha\x1b", "count": 3}, {"name": "b", "count": 12}]
    table = render.render_table(records, ["name", "count"])

    lines = table.splitlines()
    assert lines[0].split() == ["name", "count"]
    assert set(lines[1]) <= {"-", " "}  # separator row
    assert "\x1b" not in table
    assert "alpha" in lines[2]
    assert "12" in lines[3]


def test_print_records_json_output_emits_json_list(capsys: pytest.CaptureFixture[str]) -> None:
    records = [{"name": "a"}, {"name": "b"}]
    render.print_records(records, ["name"], json_output=True)

    out = capsys.readouterr().out
    assert json.loads(out) == records


def test_print_records_table_output_emits_table(capsys: pytest.CaptureFixture[str]) -> None:
    render.print_records([{"name": "a"}], ["name"], json_output=False)

    out = capsys.readouterr().out
    assert "name" in out
    assert "a" in out
    # Not JSON — a bare table has no surrounding brackets.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_print_result_mapping_renders_key_value_table(capsys: pytest.CaptureFixture[str]) -> None:
    render.print_result({"mode": "file", "ready": True}, json_output=False)

    out = capsys.readouterr().out
    assert "field" in out
    assert "value" in out
    assert "mode" in out
    assert "file" in out


def test_print_result_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    render.print_result({"mode": "file"}, json_output=True)

    assert json.loads(capsys.readouterr().out) == {"mode": "file"}
