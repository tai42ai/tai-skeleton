"""``tai traces`` — list and inspect run traces.

Thin wrappers over the ``/api/observability/runs*`` routes. ``--export`` switches a
command from the enveloped read to its download variant (CSV/JSON), printed raw.
"""

from __future__ import annotations

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
    name="traces",
    help="List and inspect run traces.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/observability/runs"), ("GET", "/api/observability/runs/export"))
def list_runs(
    ctx: typer.Context,
    from_: Annotated[
        str | None, typer.Option("--from", help="Range start: an ISO instant or a relative token (30d).")
    ] = None,
    to: Annotated[str | None, typer.Option("--to", help="Range end: an ISO instant or a relative token.")] = None,
    status: Annotated[str | None, typer.Option("--status", help="Filter: 'error' or 'success'.")] = None,
    sort: Annotated[
        str | None, typer.Option("--sort", help="Sort field: createdAt, cost, latencyMs, totalTokens.")
    ] = None,
    direction: Annotated[str | None, typer.Option("--dir", help="Sort direction: asc or desc.")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number (1-based).")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Rows per page.")] = None,
    export: Annotated[bool, typer.Option("--export", help="Download the filtered list instead of paging it.")] = False,
    fmt: Annotated[str, typer.Option("--format", help="Export format: csv or json (only with --export).")] = "csv",
) -> None:
    """List observability runs, or download the filtered list with ``--export``.

    Example: ``tai traces list --status error --sort cost``
    """
    ctx_obj = app_context(ctx)
    params: dict[str, str] = {}
    if from_ is not None:
        params["from"] = from_
    if to is not None:
        params["to"] = to
    if status is not None:
        params["status"] = status
    if sort is not None:
        params["sort"] = sort
    if direction is not None:
        params["dir"] = direction
    if export:
        params["format"] = fmt
        typer.echo(fetch_download(ctx_obj, "GET", "/api/observability/runs/export", params=params or None))
        return
    if page is not None:
        params["page"] = str(page)
    if page_size is not None:
        params["pageSize"] = str(page_size)
    with ctx_obj.client() as client:
        data = client.get("/api/observability/runs", params=params or None)
    emit_records(ctx_obj, data, ["traceId", "createdAt", "status", "cost", "latencyMs"], items_key="items")


@app.command("get")
@covers(
    ("GET", "/api/observability/runs/{trace_id}/trace"),
    ("GET", "/api/observability/runs/{trace_id}/trace/export"),
)
def get_trace(
    ctx: typer.Context,
    trace_id: Annotated[str, typer.Argument(help="Trace id.")],
    export: Annotated[bool, typer.Option("--export", help="Download the full trace as a JSON file instead.")] = False,
) -> None:
    """Get one run's full trace, or download it as JSON with ``--export``.

    Example: ``tai traces get trace_abc``
    """
    ctx_obj = app_context(ctx)
    if export:
        typer.echo(fetch_download(ctx_obj, "GET", f"/api/observability/runs/{trace_id}/trace/export"))
        return
    with ctx_obj.client() as client:
        data = client.get(f"/api/observability/runs/{trace_id}/trace")
    emit_result(ctx_obj, data)
