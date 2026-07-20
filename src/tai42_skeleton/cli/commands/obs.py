"""``tai obs`` — query observability metrics.

Thin wrapper over ``GET /api/observability/metrics``. Distinct from the local
``tai metrics`` Prometheus server command. Run traces live under ``tai traces``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_result

app = typer.Typer(
    name="obs",
    help="Query observability metrics.",
    no_args_is_help=True,
)


@app.command("metrics")
@covers(("GET", "/api/observability/metrics"))
def query_metrics(
    ctx: typer.Context,
    from_: Annotated[
        str | None, typer.Option("--from", help="Range start: an ISO instant or a relative token (30d).")
    ] = None,
    to: Annotated[str | None, typer.Option("--to", help="Range end: an ISO instant or a relative token.")] = None,
    granularity: Annotated[
        str | None, typer.Option("--granularity", help="Series granularity: hour, day, or week.")
    ] = None,
) -> None:
    """Query aggregate observability metrics over a time range.

    Example: ``tai obs metrics --from 7d --granularity day``
    """
    ctx_obj = app_context(ctx)
    params: dict[str, str] = {}
    if from_ is not None:
        params["from"] = from_
    if to is not None:
        params["to"] = to
    if granularity is not None:
        params["granularity"] = granularity
    with ctx_obj.client() as client:
        data = client.get("/api/observability/metrics", params=params or None)
    emit_result(ctx_obj, data)
