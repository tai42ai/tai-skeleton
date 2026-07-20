"""Output rendering for the ``tai`` CLI.

Human-readable tables by default; raw JSON under the global ``--json`` flag for
scripting and piping. Every server-supplied string is treated as untrusted:
control characters (which include terminal escape sequences) are stripped before
anything is written to a terminal, mirroring the Studio's escape rule so a
malicious tool/resource name cannot inject cursor moves or colour codes.
"""

import json
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import IO, Any

# Categories covering C0/C1 control characters (Cc — includes ESC, the start of
# every ANSI terminal escape) and zero-width/format characters (Cf).
_STRIP_CATEGORIES = {"Cc", "Cf"}


def strip_control(value: str) -> str:
    """Drop control and format characters from an untrusted string."""
    return "".join(ch for ch in value if unicodedata.category(ch) not in _STRIP_CATEGORIES)


def _cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, default=str, ensure_ascii=False)
    return strip_control(text)


def render_json(data: Any) -> str:
    """Pretty-printed JSON for machine/pipe consumption.

    ``json.dumps`` escapes the C0 control characters (``0x00``-``0x1F``, including
    the ESC that arms an ANSI sequence) inside string values, but NOT the C1 range
    (``0x80``-``0x9F``); the payload is left byte-faithful for downstream tools
    rather than stripped, so this output is for a pipe, not a raw terminal."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def render_table(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """A column-aligned table of ``records`` restricted to ``columns``."""
    headers = [strip_control(column) for column in columns]
    rows = [[_cell(record.get(column)) for column in columns] for record in records]

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    lines.extend("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)


def print_json(data: Any, *, file: IO[str] | None = None) -> None:
    print(render_json(data), file=file if file is not None else sys.stdout)


def print_records(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    json_output: bool,
    file: IO[str] | None = None,
) -> None:
    """Render a list of records as a table, or as raw JSON under ``--json``."""
    stream = file if file is not None else sys.stdout
    if json_output:
        print_json(list(records), file=stream)
    else:
        print(render_table(records, columns), file=stream)


def print_result(data: Any, *, json_output: bool, file: IO[str] | None = None) -> None:
    """Render a single result. Under ``--json`` the raw value is emitted; the
    human form is a two-column key/value table for a mapping, else the stripped
    string form."""
    stream = file if file is not None else sys.stdout
    if json_output:
        print_json(data, file=stream)
    elif isinstance(data, Mapping):
        rows = [{"field": str(key), "value": value} for key, value in data.items()]
        print(render_table(rows, ["field", "value"]), file=stream)
    else:
        print(_cell(data), file=stream)
