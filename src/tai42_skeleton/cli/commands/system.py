"""``tai system`` — inspect the server's own runtime surface.

Thin wrappers over the authed ``/api/system*`` routes. ``kinds`` reports every
pluggable kind's live active/default/off state — the same table the startup
``[kinds]`` summary logs — so an operator can read it from the terminal without
scraping boot logs.
"""

from __future__ import annotations

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
)

app = typer.Typer(
    name="system",
    help="Inspect the server's own runtime surface.",
    no_args_is_help=True,
)


@app.command("kinds")
@covers(("GET", "/api/system/kinds"))
def kinds(ctx: typer.Context) -> None:
    """List every pluggable kind's live state (active / default / off).

    Example: ``tai system kinds``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/system/kinds")
    emit_records(ctx_obj, data, ["kind", "state", "plugin", "detail"])
