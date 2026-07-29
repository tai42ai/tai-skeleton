"""``tai extensions`` — browse the global tool-extension catalog.

``list`` is the GLOBAL extension catalog only. Per-tool extension operations
(show/apply a tool's combos) live under the ``tools`` group.
"""

from __future__ import annotations

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_records

app = typer.Typer(
    name="extensions",
    help="Browse the global tool-extension catalog.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/extensions"))
def list_extensions(ctx: typer.Context) -> None:
    """List every registered extension in the global catalog.

    Example: ``tai extensions list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/extensions")
    emit_records(ctx_obj, data, ["name", "kind"])
