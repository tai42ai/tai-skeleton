"""``tai backup`` — export and import server state.

Thin wrappers over the ``/api/backup/*`` routes. ``export`` streams the raw backup
document to stdout (redirect it to a file); ``import`` consumes exactly that file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
    fetch_download,
)

app = typer.Typer(
    name="backup",
    help="Export and import server state.",
    no_args_is_help=True,
)


@app.command("sections")
@covers(("GET", "/api/backup/sections"))
def list_sections(ctx: typer.Context) -> None:
    """List the registered backup sections.

    Example: ``tai backup sections``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/backup/sections")
    emit_records(ctx_obj, data, ["name", "secret"])


@app.command("export")
@covers(("POST", "/api/backup/export"))
def export_backup(
    ctx: typer.Context,
    section: Annotated[list[str], typer.Option("--section", help="A section to export (repeatable).")],
) -> None:
    """Export the named sections as a backup document (printed to stdout).

    Redirect stdout to save it: ``tai backup export --section access_control > backup.json``
    """
    ctx_obj = app_context(ctx)
    document = fetch_download(ctx_obj, "POST", "/api/backup/export", json_body={"sections": list(section)})
    typer.echo(document)


@app.command("import")
@covers(("POST", "/api/backup/import"))
def import_backup(
    ctx: typer.Context,
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="A backup document file.")],
    section: Annotated[list[str], typer.Option("--section", help="A section to import (repeatable).")],
) -> None:
    """Import selected sections from a backup document file.

    Example: ``tai backup import backup.json --section access_control``
    """
    ctx_obj = app_context(ctx)
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"backup file must be valid JSON: {exc}", param_hint="FILE") from exc
    with ctx_obj.client() as client:
        data = client.post("/api/backup/import", json={"document": document, "sections": list(section)})
    emit_result(ctx_obj, data)
