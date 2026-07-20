"""``tai channels`` — inspect the registered delivery channels.

A thin wrapper over the authed ``GET /api/channels`` catalog route: the channel
names ``ask_user(channel=...)`` can currently resolve. Registration itself is
import-only (a manifest ``channel_modules`` entry), so this group is read-only.
"""

from __future__ import annotations

import typer

from tai_skeleton.cli.commands._common import app_context, covers, emit_records

app = typer.Typer(
    name="channels",
    help="Inspect the registered delivery channels.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/channels"))
def list_channels(ctx: typer.Context) -> None:
    """List the registered channel names.

    Example: ``tai channels list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/channels")
    emit_records(ctx_obj, data, ["name"], items_key="channels")
