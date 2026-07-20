"""``tai fleet`` — inspect the backend identity and drive fleet ops.

Thin wrappers over the authed ``/api/backend`` (identity) and ``/api/fleet/*``
(census + soft-restart) routes. The group is named ``fleet`` (not ``backend``)
because the ``tai backend`` command is the re-homed runtime launcher; a
``backend`` group would clobber it. ``info`` reports whether a task backend is
installed; ``workers`` lists the whole bus fleet and ``reload-config`` soft-restarts
it — both work with or without a task backend.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
)

app = typer.Typer(
    name="fleet",
    help="Inspect the execution backend and drive fleet ops.",
    no_args_is_help=True,
)


@app.command("info")
@covers(("GET", "/api/backend"))
def fleet_info(ctx: typer.Context) -> None:
    """Show the registered backend's identity (or the empty state).

    Example: ``tai fleet info``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/backend")
    emit_result(ctx_obj, data)


@app.command("workers")
@covers(("GET", "/api/fleet/workers"))
def list_workers(ctx: typer.Context) -> None:
    """List the live worker fleet — every process on the bus.

    Example: ``tai fleet workers``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/fleet/workers")
    emit_records(ctx_obj, data, ["origin", "kind", "pid"], items_key="workers")


@app.command("reload-config")
@covers(("POST", "/api/fleet/reload-config"))
def reload_config(
    ctx: typer.Context,
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the reload to (repeatable).")
    ] = None,
) -> None:
    """Soft-restart the worker fleet — all of them, or only ``--target`` ones.

    The response is the per-origin fleet report (one row per worker with its
    reload outcome). Example: ``tai fleet reload-config --target serve-abc123``
    """
    ctx_obj = app_context(ctx)
    targets = list(target) if target else None
    with ctx_obj.client() as client:
        data = client.post("/api/fleet/reload-config", json={"targets": targets})
    emit_result(ctx_obj, data)
