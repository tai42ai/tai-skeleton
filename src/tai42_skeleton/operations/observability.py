"""Observability read operations — ``/api/observability/*``.

Account-wide runs observability sourced exclusively from the vendor-neutral
monitoring READ contract (``tai42_contract.monitoring``): aggregate metrics, a
filterable run list, and single-run trace detail. A "run" is a monitoring
**trace** (keyed by ``trace_id``); there is no other source.

The reader is fetched per operation from the process monitoring registry
(``get_monitoring().reader``) so a reload swaps the backend without stale
capture; with no backend registered the no-op reader answers with EMPTY data
(not an error). Two typed monitoring errors are mapped: a read the backend
cannot serve becomes ``NotSupportedError`` (501, carrying the ``code`` the UI
keys its dedicated state on) and an absent trace becomes ``NotFoundError`` (404).
Every other reader error propagates loudly as a 500. The query string is decoded
into the neutral filter/paging types at the HTTP edge (the router's context
extractors, which raise ``BadRequestError`` → 400); these operations receive the
already-parsed flat params and stay request-free.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from tai42_contract.monitoring import (
    MetricsFilter,
    MetricsView,
    MonitoringFilter,
    MonitoringReadNotSupportedError,
    OrderBy,
    TraceNotFoundError,
)

from tai42_skeleton.monitoring.registry import get_monitoring
from tai42_skeleton.operations import BadRequestError, NotFoundError, NotSupportedError, operation
from tai42_skeleton.routers.observability_support import (
    _safe_query,
    derive_run,
    map_model_rows,
    map_trace,
    summary_from_rows,
    time_series_from_rows,
)

# Measures requested from the metrics API. Names are the backend's measure
# vocabulary (the contract passes them through unchanged).
_METRICS = ["count", "totalCost", "totalTokens", "latency"]

# The ``code`` a monitoring read-not-supported answer carries so the UI can
# render its dedicated 'read not supported' state (not a generic error).
_READ_NOT_SUPPORTED_CODE = "monitoring-read-not-supported"


@operation(
    summary="Query observability metrics",
    tags=["observability"],
    errors=[BadRequestError, NotSupportedError],
)
async def get_metrics(t0: datetime, t1: datetime, granularity: str) -> dict:
    """Aggregate metrics over the time range via the contract's ``query_metrics``.

    Three fixed queries: a summary total, a granularity series, and an OPTIONAL
    by-model breakdown. By-model runs through ``_safe_query`` so a backend
    rejecting the extra dimension omits that panel rather than breaking the core
    tiles; the summary and series queries are hard (they may raise).
    """
    reader = get_monitoring().reader

    summary_filter = MetricsFilter(view=MetricsView.TRACES, metrics=_METRICS, from_timestamp=t0, to_timestamp=t1)
    series_filter = MetricsFilter(
        view=MetricsView.TRACES,
        metrics=_METRICS,
        from_timestamp=t0,
        to_timestamp=t1,
        granularity=granularity,
    )
    model_filter = MetricsFilter(
        view=MetricsView.OBSERVATIONS,
        metrics=_METRICS,
        from_timestamp=t0,
        to_timestamp=t1,
        dimensions=["providedModelName"],
    )

    try:
        summary_res, series_res, model_res = await asyncio.gather(
            reader.query_metrics(summary_filter),
            reader.query_metrics(series_filter),
            _safe_query(reader, model_filter),
        )
    except MonitoringReadNotSupportedError as exc:
        raise NotSupportedError(str(exc), extra={"code": _READ_NOT_SUPPORTED_CODE}) from exc

    return {
        "summary": summary_from_rows(summary_res.rows),
        "timeSeries": time_series_from_rows(series_res.rows),
        "byModel": map_model_rows(model_res),
        "granularity": granularity,
    }


@operation(
    summary="List observability runs",
    tags=["observability"],
    errors=[BadRequestError, NotSupportedError],
)
async def list_observability_runs(
    t0: datetime,
    t1: datetime,
    run_filter: MonitoringFilter | None,
    order_by: OrderBy | None,
    page: int,
    page_size: int,
) -> dict:
    """Filterable run list via the contract's ``list_traces``, paged with the
    reader's ``limit`` / ``page``. Time range plus the neutral advanced filters
    (tags / status / cost / token / latency ranges) and sort.

    List- or dict-typed query params are JSON-encoded in the query string
    (``tags`` may be a JSON list ``["a","b"]`` or a comma-separated string)."""
    reader = get_monitoring().reader
    try:
        traces = await reader.list_traces(
            from_timestamp=t0,
            to_timestamp=t1,
            limit=page_size,
            page=page,
            filter=run_filter,
            order_by=order_by,
        )
    except MonitoringReadNotSupportedError as exc:
        raise NotSupportedError(str(exc), extra={"code": _READ_NOT_SUPPORTED_CODE}) from exc

    items = [derive_run(t) for t in traces]
    next_page = page + 1 if len(traces) >= page_size else None
    return {"items": items, "page": page, "nextPage": next_page}


@operation(
    summary="Get a run's trace",
    tags=["observability"],
    errors=[NotFoundError, NotSupportedError],
)
async def get_run_trace(trace_id: str) -> dict:
    """Full detailed trace for one run via the contract's ``get_trace``.

    The contract's ``get_trace`` always returns a trace or raises: an absent trace
    is ``TraceNotFoundError`` → 404, and every other backend error propagates as a
    loud 500.
    """
    reader = get_monitoring().reader
    try:
        trace = await reader.get_trace(trace_id)
    except MonitoringReadNotSupportedError as exc:
        raise NotSupportedError(str(exc), extra={"code": _READ_NOT_SUPPORTED_CODE}) from exc
    except TraceNotFoundError as exc:
        raise NotFoundError("Run trace not found") from exc
    return map_trace(trace)
