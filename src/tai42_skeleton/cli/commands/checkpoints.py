"""``tai checkpoints`` — conversation checkpoint retention.

A thin wrapper over the ``/api/checkpoints/sweep`` route.
"""

from __future__ import annotations

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_result

app = typer.Typer(
    name="checkpoints",
    help="Manage conversation checkpoint retention.",
    no_args_is_help=True,
)


@app.command("sweep")
@covers(("POST", "/api/checkpoints/sweep"))
def sweep(ctx: typer.Context) -> None:
    """Delete conversation checkpoints idle longer than the configured lifetime.

    Example: ``tai checkpoints sweep``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/checkpoints/sweep")
    emit_result(ctx_obj, data)
