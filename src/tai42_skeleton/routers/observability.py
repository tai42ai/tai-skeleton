"""HTTP routes for the observability surface — ``/api/observability/*``.

Account-wide runs observability sourced exclusively from the vendor-neutral
monitoring READ contract (``tai42_contract.monitoring``): aggregate metrics, a
filterable run list, single-run trace detail, and CSV/JSON exports. A "run" is a
monitoring **trace** (keyed by ``trace_id``); there is no other source.

All five routes are AUTHED. A trace embeds the full input/output of every tool
and model call in a run — arguments, results, prompts, completions — so these
reads are data-bearing and must sit behind the Studio credential, never public.

The three enveloped reads (metrics, run list, single-trace detail) are thin
adapters over operations in ``tai42_skeleton.operations.observability``: the query
string is decoded into the neutral filter/paging types here at the HTTP edge (the
context extractors raise ``BadRequestError`` → 400), then the operation runs
request-free. The two DOWNLOAD routes (a single trace as a JSON file, the run
list as a CSV/JSON file) stay handlers: they answer a raw attachment with a
``Content-Disposition`` header, not the ``{"data": ...}`` envelope.

The reader is fetched per request from the process monitoring registry
(``get_monitoring().reader``) so a reload swaps the backend without stale
capture; with no backend registered the no-op reader answers with EMPTY data
(not an error). The reader's methods are async and awaited directly.

Errors are loud. Only two typed monitoring errors are mapped:
``MonitoringReadNotSupportedError`` → 501 with a ``code`` the UI keys its
dedicated state on, and ``TraceNotFoundError`` → 404 on the trace routes. Every
other exception propagates as a 500 — no blanket catch, no silent degrade. The
one deliberate exception is the OPTIONAL by-model metrics sub-query
(``_safe_query``), which is logged and omitted rather than allowed to break the
core tiles. Success bodies are ``{"data": ...}``; failures are
``{"error": "<message>"}``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app
from tai42_contract.monitoring import (
    MonitoringReadNotSupportedError,
    MonitoringTrace,
    TraceNotFoundError,
)

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.monitoring.registry import get_monitoring
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.observability import get_metrics as _get_metrics_op
from tai42_skeleton.operations.observability import get_run_trace as _get_run_trace_op
from tai42_skeleton.operations.observability import list_observability_runs as _list_observability_runs_op
from tai42_skeleton.routers.observability_support import (
    PAGE_CHUNK,
    RequestParseError,
    csv_safe,
    derive_run,
    map_trace,
    parse_paging,
    parse_run_filter,
    parse_time_range,
    select_granularity,
)

logger = logging.getLogger(__name__)

# Bulk-export guardrail: cap rows so a huge range cannot stream unbounded. When
# exceeded, the response flags truncation (no silent loss).
_EXPORT_CAP = 5000

_CSV_COLUMNS = [
    "traceId",
    "createdAt",
    "status",
    "fetchError",
    "cost",
    "latencyMs",
    "totalTokens",
    "model",
    "inputPreview",
    "outputPreview",
]


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _not_supported(exc: MonitoringReadNotSupportedError) -> JSONResponse:
    """The backend cannot serve reads — a distinct 501 carrying ``code`` so the
    UI can render its dedicated 'read not supported' state (not a generic error)."""
    return JSONResponse(
        {"error": str(exc), "code": "monitoring-read-not-supported"},
        status_code=501,
    )


# ---------------------------------------------------------------------------
# HTTP-edge query extractors for the read operations
# ---------------------------------------------------------------------------


async def _extract_metrics_query(request: Request) -> dict[str, Any]:
    """Decode the metrics query string into the operation's flat params, mapping the
    module-local ``RequestParseError`` to the door's explicit 400."""
    try:
        t0, t1 = parse_time_range(request)
        granularity = select_granularity(t0, t1, request.query_params.get("granularity"))
    except RequestParseError as exc:
        raise BadRequestError(str(exc)) from exc
    return {"t0": t0, "t1": t1, "granularity": granularity}


async def _extract_runs_query(request: Request) -> dict[str, Any]:
    """Decode the run-list query string into the operation's flat params, mapping the
    module-local ``RequestParseError`` to the door's explicit 400."""
    try:
        t0, t1 = parse_time_range(request)
        run_filter, order_by = parse_run_filter(request)
        page, page_size = parse_paging(request)
    except RequestParseError as exc:
        raise BadRequestError(str(exc)) from exc
    return {"t0": t0, "t1": t1, "run_filter": run_filter, "order_by": order_by, "page": page, "page_size": page_size}


get_metrics = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_metrics_op),
    path="/api/observability/metrics",
    method="GET",
    context_extractor=_extract_metrics_query,
    action="read",
)

list_runs = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_observability_runs_op),
    path="/api/observability/runs",
    method="GET",
    context_extractor=_extract_runs_query,
    action="read",
)

get_run_trace = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_run_trace_op),
    path="/api/observability/runs/{trace_id}/trace",
    method="GET",
    action="read",
)


# ---------------------------------------------------------------------------
# Exports (handler-native downloads — a raw attachment, not the JSON envelope)
# ---------------------------------------------------------------------------


@http_surface().custom_route(
    "/api/observability/runs/{trace_id}/trace/export",
    methods=["GET"],
    summary="Export a run's trace as a JSON download",
    tags=["observability"],
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(401, 404, 501),
        success_status=200,
    ),
    action="read",
)
async def export_run_trace(request: Request) -> Response:
    """Single run's full trace as a downloadable JSON file."""
    trace_id = request.path_params["trace_id"]
    reader = get_monitoring().reader
    try:
        trace = await reader.get_trace(trace_id)
    except MonitoringReadNotSupportedError as exc:
        return _not_supported(exc)
    except TraceNotFoundError:
        return _error("Run trace not found", 404)

    body = json.dumps(map_trace(trace), default=str, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="trace-{trace_id}.json"'},
    )


@http_surface().custom_route(
    "/api/observability/runs/export",
    methods=["GET"],
    summary="Export runs as a CSV download",
    tags=["observability"],
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(400, 401, 501),
        success_status=200,
    ),
    action="read",
)
async def export_runs(request: Request) -> Response:
    """Bulk export of the filtered run list as CSV (default) or JSON. Honors the
    same advanced filters as the run list. Capped at ``_EXPORT_CAP`` rows;
    truncation is flagged in-band (never a silent loss)."""
    try:
        t0, t1 = parse_time_range(request)
        run_filter, order_by = parse_run_filter(request)
    except RequestParseError as exc:
        return _error(str(exc), 400)
    fmt = request.query_params.get("format", "csv")
    if fmt not in ("csv", "json"):
        return _error("format must be 'csv' or 'json'", 400)

    reader = get_monitoring().reader
    # Drain pages of ``PAGE_CHUNK`` (the reader rejects oversized pages) until the
    # cap is exceeded or a short/empty page signals the end.
    traces: list[MonitoringTrace] = []
    page = 1
    try:
        while len(traces) <= _EXPORT_CAP:
            batch = await reader.list_traces(
                from_timestamp=t0,
                to_timestamp=t1,
                limit=PAGE_CHUNK,
                page=page,
                filter=run_filter,
                order_by=order_by,
            )
            traces.extend(batch)
            if len(batch) < PAGE_CHUNK:
                break
            page += 1
    except MonitoringReadNotSupportedError as exc:
        return _not_supported(exc)

    truncated = len(traces) > _EXPORT_CAP
    rows = [derive_run(t) for t in traces[:_EXPORT_CAP]]
    if truncated:
        logger.warning("runs export truncated at %d rows", _EXPORT_CAP)

    if fmt == "json":
        payload = {"items": rows, "truncated": truncated}
        return Response(
            content=json.dumps(payload, default=str, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="runs.json"'},
        )

    return _csv_response(rows, truncated=truncated)


def _csv_response(rows: list[dict], *, truncated: bool) -> Response:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # input/output previews are structured values — serialize for the CSV cell.
        flat = dict(row)
        for key in ("inputPreview", "outputPreview"):
            v = flat.get(key)
            if v is not None and not isinstance(v, str):
                flat[key] = json.dumps(v, ensure_ascii=False)
        writer.writerow({k: csv_safe(v) for k, v in flat.items()})
    if truncated:
        buf.write(f"# truncated at {_EXPORT_CAP} rows\n")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="runs.csv"'},
    )
