"""``tai catalog`` — dump the packaged ecosystem catalog."""

from __future__ import annotations

import importlib.resources
from typing import Any

import click
import typer
import yaml

from tai_skeleton.cli.commands._common import app_context
from tai_skeleton.cli.render import print_records

# Columns rendered in the human table (JSON output carries the raw records).
_COLUMNS = ["name", "kind", "group", "package", "repo", "module", "description"]


def load_catalog() -> list[dict[str, Any]]:
    """Read the packaged ecosystem catalog, joining each entry's ``package`` to
    its repo via the file's ``packages`` map.

    The repo lives in exactly one place — the ``packages`` map — so a package
    that appears on an entry but is absent from the map raises loudly rather than
    rendering a blank repo cell.
    """
    resource = importlib.resources.files("tai_skeleton").joinpath("data", "ecosystem.yml")
    document = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    entries = document.get("entries", [])
    packages = document.get("packages", {})

    records: list[dict[str, Any]] = []
    for entry in entries:
        package = entry["package"]
        if package not in packages:
            raise RuntimeError(
                f"ecosystem catalog entry '{entry['name']}' names package '{package}', which is "
                "missing from the 'packages' repo map in ecosystem.yml — add it, never leave the repo blank."
            )
        records.append({**entry, "repo": packages[package]})
    return records


def catalog(ctx: typer.Context) -> None:
    """Print the packaged ecosystem catalog.

    Reads the static catalog shipped inside the tai-skeleton package and renders
    it, working offline on a bare install with zero plugin imports. ``--json``
    (global) emits the raw records for scripting.
    """
    app_ctx = app_context(ctx)
    try:
        records = load_catalog()
    except RuntimeError as exc:
        # A packaging bug (an entry's package missing from the repo map) should read
        # as the uniform CLI error line, not a raw traceback.
        raise click.ClickException(str(exc)) from exc
    print_records(records, _COLUMNS, json_output=app_ctx.json_output)
