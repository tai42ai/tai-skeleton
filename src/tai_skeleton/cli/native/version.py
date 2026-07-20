"""``tai version`` — show tai-skeleton and key dependency versions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import typer

from tai_skeleton.cli.commands._common import app_context
from tai_skeleton.cli.render import print_records

# tai-skeleton first, then the foundation and the runtime deps whose versions
# most affect behavior in the field.
_PACKAGES = ["tai-skeleton", "tai-contract", "tai-kit", "fastmcp", "mcp", "typer", "click"]


def _versions() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name in _PACKAGES:
        try:
            installed = package_version(name)
        except PackageNotFoundError:
            installed = "not installed"
        records.append({"package": name, "version": installed})
    return records


def version(ctx: typer.Context) -> None:
    """Show the tai-skeleton version and the versions of its key dependencies."""
    app_ctx = app_context(ctx)
    print_records(_versions(), ["package", "version"], json_output=app_ctx.json_output)
