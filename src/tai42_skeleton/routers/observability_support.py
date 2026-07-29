"""Pure helpers for the observability router, kept out of ``observability.py``
so the router holds only route registration, reader glue, and handlers.

Two groups, neither of which owns a reader instance:

* **Request → params** — decode the query string into the contract's neutral
  types (``parse_time_range``, ``parse_run_filter``, ``parse_paging``,
  ``select_granularity``). These take a Starlette ``Request`` but do no I/O and
  raise ``RequestParseError`` (→ 400) on malformed input.
* **Output transforms** — pure functions over plain values / contract models:
  the metrics summary + series + by-model row readers, the trace → run / trace
  mapping, the structurally-bounded input/output preview, and the CSV-injection
  guard.

``_safe_query`` is the one helper that touches a reader: it runs an OPTIONAL
metrics sub-query and returns ``None`` (logged, never raised) when the backend
rejects it, so an unsupported extra dimension cannot break the core tiles.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from tai42_contract.monitoring import (
    MetricsFilter,
    MetricsResult,
    MonitoringFilter,
    MonitoringLevel,
    MonitoringTrace,
    OrderBy,
)

if TYPE_CHECKING:
    from starlette.requests import Request
    from tai42_contract.monitoring import MonitoringObservation, MonitoringReader

logger = logging.getLogger(__name__)


class RequestParseError(Exception):
    """A query parameter is missing or malformed — the handler maps it to a
    loud 400. Distinct from a reader/backend failure, which propagates as a 500
    (or the two typed monitoring errors, which map to 501/404)."""


# ---------------------------------------------------------------------------
# Request → params parsing (time range, advanced run filter, paging, granularity)
# ---------------------------------------------------------------------------

_RELATIVE_RE = re.compile(r"^(\d+)([hdw])$")
_RELATIVE_UNIT = {"h": "hours", "d": "days", "w": "weeks"}
_DEFAULT_FROM = "30d"

# Run-list sort key (frontend token) → neutral ``OrderBy.field`` on list_traces.
# ``timestamp`` sorts natively; ``total_cost`` / ``latency`` / ``total_tokens``
# are metric-ranked globally by the contract.
_SORT_FIELDS = {
    "createdAt": "timestamp",
    "cost": "total_cost",
    "latencyMs": "latency",
    "totalTokens": "total_tokens",
}

# Max page size the reader accepts per ``list_traces`` call; the run list caps
# ``pageSize`` here and the export drains pages of this size up to ``_EXPORT_CAP``.
PAGE_CHUNK = 100


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_instant(value: str, *, now: datetime, field: str) -> datetime:
    """Resolve an absolute ISO instant or a relative token (e.g. ``7d`` = 7 days
    ago), normalized to UTC."""
    match = _RELATIVE_RE.match(value)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        return now - timedelta(**{_RELATIVE_UNIT[unit]: amount})
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RequestParseError(f"Invalid {field}: {value}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_time_range(request: Request) -> tuple[datetime, datetime]:
    """The ``from`` / ``to`` window: each is an ISO instant or a relative token
    (``\\d+[hdw]``). ``from`` defaults to ``30d`` ago, ``to`` to now. ``from``
    at or after ``to`` is a 400."""
    q = request.query_params
    now = _now_utc()
    t0 = _parse_instant(q.get("from") or _DEFAULT_FROM, now=now, field="from")
    t1 = _parse_instant(q["to"], now=now, field="to") if q.get("to") else now
    if t0 >= t1:
        raise RequestParseError("`from` must be before `to`")
    return t0, t1


def select_granularity(t0: datetime, t1: datetime, override: str | None) -> str:
    """≤2 days → hourly, ≤90 days → daily, else weekly — unless pinned to one of
    hour/day/week. An explicit granularity outside that set is malformed and raises
    ``RequestParseError`` (→ 400, same as a bad from/to); auto-selection applies
    only when it is absent."""
    if override:
        if override not in ("hour", "day", "week"):
            raise RequestParseError(f"granularity must be one of hour, day, week: {override!r}")
        return override
    span = (t1 - t0).total_seconds()
    if span <= 2 * 86400:
        return "hour"
    if span <= 90 * 86400:
        return "day"
    return "week"


def _q_float(q: Any, key: str) -> float | None:
    raw = q.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RequestParseError(f"{key} must be a number") from exc


def _parse_tags(raw: str | None) -> list[str]:
    """Tags as either a JSON list (``["a","b"]``) or a comma-separated string.

    A value opening with ``[`` is decoded as JSON (the list/dict query-param
    encoding); anything else is split on commas. Empty entries are dropped."""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except ValueError as exc:
            raise RequestParseError("tags must be a JSON list or comma-separated") from exc
        if not isinstance(decoded, list) or not all(isinstance(t, str) for t in decoded):
            raise RequestParseError("tags JSON must be a list of strings")
        return [t.strip() for t in decoded if t.strip()]
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_run_filter(request: Request) -> tuple[MonitoringFilter | None, OrderBy | None]:
    """Build the neutral ``MonitoringFilter`` + ``OrderBy`` for the run list from
    the query string. Every clause is optional; an all-empty query yields
    ``(None, None)`` (a plain newest-first listing).

    List- or dict-typed params are JSON-encoded in the query string (``tags`` may
    be a JSON list or a comma-separated string). ``status=error`` maps to a
    level==ERROR clause; ``status=success`` is unfiltered (the absence of errors
    is not a trace-level clause). The ``minCost`` / ``maxCost`` / ``minTokens`` /
    ``maxTokens`` / ``minLatencyMs`` / ``maxLatencyMs`` ranges are inclusive;
    latency is exposed in ms and converted to the contract's seconds. An inverted
    range is rejected by the contract (→ 400). Sort: ``sort`` ∈ {createdAt, cost,
    latencyMs, totalTokens} with ``dir`` ∈ {asc, desc} (default desc)."""
    q = request.query_params

    tags = _parse_tags(q.get("tags"))
    status = q.get("status")
    if status not in (None, "", "error", "success"):
        raise RequestParseError("status must be 'error' or 'success'")
    level = MonitoringLevel.ERROR if status == "error" else None

    min_latency_ms = _q_float(q, "minLatencyMs")
    max_latency_ms = _q_float(q, "maxLatencyMs")
    min_cost = _q_float(q, "minCost")
    max_cost = _q_float(q, "maxCost")
    min_tokens = _q_float(q, "minTokens")
    max_tokens = _q_float(q, "maxTokens")

    has_clause = any(
        v is not None and v != []
        for v in (tags or None, level, min_cost, max_cost, min_tokens, max_tokens, min_latency_ms, max_latency_ms)
    )
    filter_: MonitoringFilter | None = None
    if has_clause:
        try:
            filter_ = MonitoringFilter(
                tags=tags,
                level=level,
                min_cost=min_cost,
                max_cost=max_cost,
                min_tokens=int(min_tokens) if min_tokens is not None else None,
                max_tokens=int(max_tokens) if max_tokens is not None else None,
                min_latency=min_latency_ms / 1000 if min_latency_ms is not None else None,
                max_latency=max_latency_ms / 1000 if max_latency_ms is not None else None,
            )
        except ValueError as exc:  # inverted range → contract rejects at construction
            raise RequestParseError(str(exc)) from exc

    order_by: OrderBy | None = None
    sort = q.get("sort")
    if sort:
        if sort not in _SORT_FIELDS:
            raise RequestParseError(f"sort must be one of {sorted(_SORT_FIELDS)}")
        direction = q.get("dir", "desc")
        if direction not in ("asc", "desc"):
            raise RequestParseError("dir must be 'asc' or 'desc'")
        order_by = OrderBy(field=_SORT_FIELDS[sort], direction=direction)

    return filter_, order_by


def parse_paging(request: Request) -> tuple[int, int]:
    """``page`` (default 1) and ``pageSize`` (default 50). A ``page`` or
    ``pageSize`` below 1 is malformed and raises ``RequestParseError`` (→ 400),
    consistent with the from/to and granularity checks — never silently clamped.
    ``pageSize`` above ``PAGE_CHUNK`` is capped to that documented server limit
    (valid data, not an error). A non-integer value is a 400."""
    q = request.query_params
    try:
        page = int(q.get("page", "1"))
        page_size = int(q.get("pageSize", "50"))
    except ValueError as exc:
        raise RequestParseError("page and pageSize must be integers") from exc
    if page < 1 or page_size < 1:
        raise RequestParseError("page and pageSize must be >= 1")
    page_size = min(page_size, PAGE_CHUNK)
    return page, page_size


# ---------------------------------------------------------------------------
# CSV-injection guard
# ---------------------------------------------------------------------------

# Leading characters a spreadsheet treats as a formula. User/LLM-controlled cell
# text starting with one is prefixed with ``'`` so Excel/Sheets render it as
# text (CSV-injection guard).
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# Metrics-row readers (Dashboard tab)
# ---------------------------------------------------------------------------

_ISO_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_bucket(row: Any) -> str | None:
    """Find the time-bucket value in a granularity row. The contract does not
    standardize the time-field key, so search dimensions then metrics for the
    first ISO-date-like string."""
    for source in (row.dimensions, row.metrics):
        for value in source.values():
            if isinstance(value, str) and _ISO_LIKE.match(value):
                return value
    return None


def metric_value(metrics: dict[str, Any], measure: str) -> float:
    """Read a measure's value from a metrics row.

    The backend names output columns by measure + aggregation (e.g.
    ``count_count``, ``totalCost_sum``, ``latency_avg``), and the order is not
    standardized across versions — so match by measure-name substring, with an
    exact-key fast path when present."""
    if measure in metrics:
        return _num(metrics[measure])
    needle = measure.lower()
    for key, value in metrics.items():
        if isinstance(key, str) and needle in key.lower() and not _ISO_LIKE.match(str(value)):
            return _num(value)
    return 0.0


def summary_from_rows(rows: list[Any]) -> dict[str, Any]:
    """The Dashboard summary tile derived from the first (ungrouped) metrics row.

    Empty rows → all zeros. ``avgCostPerRun`` / ``avgTokensPerRun`` are derived
    from the totals; ``timeToFirstTokenMs`` is always ``None`` (no neutral
    measure for it — the field is kept for a stable response shape)."""
    m = rows[0].metrics if rows else {}
    total_runs = int(metric_value(m, "count"))
    total_cost = metric_value(m, "totalCost")
    total_tokens = int(metric_value(m, "totalTokens"))
    return {
        "totalRuns": total_runs,
        "totalCost": total_cost,
        "totalTokens": total_tokens,
        "averageLatencyMs": int(metric_value(m, "latency")),
        "avgCostPerRun": (total_cost / total_runs) if total_runs else 0.0,
        "avgTokensPerRun": int(total_tokens / total_runs) if total_runs else 0,
        "timeToFirstTokenMs": None,
    }


def time_series_from_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Per-bucket series points for the Dashboard chart, one per granularity row."""
    return [
        {
            "bucket": extract_bucket(row),
            "runs": int(metric_value(row.metrics, "count")),
            "cost": metric_value(row.metrics, "totalCost"),
            "avgLatencyMs": int(metric_value(row.metrics, "latency")),
            "totalTokens": int(metric_value(row.metrics, "totalTokens")),
        }
        for row in rows
    ]


def map_model_rows(res: MetricsResult | None) -> list[dict[str, Any]]:
    """Per-model breakdown rows, top 8 by cost. Empty when unavailable (the
    by-model sub-query is optional — see ``_safe_query``)."""
    if res is None:
        return []
    rows = [
        {
            "model": row.dimensions.get("providedModelName") or "unknown",
            "calls": int(metric_value(row.metrics, "count")),
            "cost": metric_value(row.metrics, "totalCost"),
            "totalTokens": int(metric_value(row.metrics, "totalTokens")),
            "avgLatencyMs": int(metric_value(row.metrics, "latency")),
        }
        for row in res.rows
    ]
    rows.sort(key=lambda r: r["cost"], reverse=True)
    return rows[:8]


async def _safe_query(reader: MonitoringReader, f: MetricsFilter) -> MetricsResult | None:
    """Run an OPTIONAL metrics sub-query, returning ``None`` (logged, not raised)
    if the backend rejects it — so an unsupported measure/dimension cannot break
    the core tiles. This is the single deliberate, visible degrade path; every
    other reader error propagates."""
    try:
        return await reader.query_metrics(f)
    except Exception as exc:
        # Optional enrichment: any backend error → omit this panel + log (the one
        # deliberate, visible degrade path; every other reader error propagates).
        logger.warning("optional metrics sub-query failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Run-list input/output previews
# ---------------------------------------------------------------------------

# Structural bounds for the run-list previews: keep the row payload small while
# staying VALID JSON (truncate at the structure level, not mid-string), so the
# UI can always render the preview as a parsed tree. The full value lives in the
# trace drill-in.
_PREVIEW_STR = 240  # max chars per string leaf
_PREVIEW_ITEMS = 20  # max entries per object / array
_PREVIEW_DEPTH = 6  # max nesting depth
_PREVIEW_PARSE_MAX = 20_000  # do not structurally parse strings larger than this


def _maybe_json(text: str) -> Any:
    """Parse a string that is itself an object/array — JSON first, then a Python
    ``repr`` (single quotes, ``True``/``False``/``None``) via ``literal_eval`` —
    else ``None``. Lets a stringified structure be bounded structurally (and
    rendered as a tree) instead of char-clipped into garbage."""
    s = text.strip()
    # Size gate: parsing happens before structural bounding, so a multi-MB
    # stringified payload would be fully materialized just to keep 20 items.
    # Over the cap, skip parsing — the caller char-clips the raw string instead.
    if len(s) < 2 or len(s) > _PREVIEW_PARSE_MAX or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except ValueError:
        pass
    try:
        # literal_eval is safe — only Python literals, no code execution.
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _preview(value: Any, depth: int = 0) -> Any:
    """A structurally-bounded copy of a (possibly nested) value for the run list:
    long strings clipped, objects/arrays capped, deep nesting elided — small but
    still valid JSON the client can render as a tree."""
    if isinstance(value, str):
        parsed = _maybe_json(value)
        if isinstance(parsed, (dict, list)):
            return _preview(parsed, depth)
        return value if len(value) <= _PREVIEW_STR else value[:_PREVIEW_STR] + "…"
    if isinstance(value, dict):
        if depth >= _PREVIEW_DEPTH:
            return {"…": "…"}
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _PREVIEW_ITEMS:
                out["…"] = f"+{len(value) - _PREVIEW_ITEMS} more"
                break
            out[str(k)] = _preview(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if depth >= _PREVIEW_DEPTH:
            return ["…"]
        items = [_preview(v, depth + 1) for v in list(value)[:_PREVIEW_ITEMS]]
        if len(value) > _PREVIEW_ITEMS:
            items.append(f"+{len(value) - _PREVIEW_ITEMS} more")
        return items
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)  # dates, etc. — JSON-safe fallback


# ---------------------------------------------------------------------------
# Trace → run / trace mapping
# ---------------------------------------------------------------------------


def derive_run(trace: MonitoringTrace) -> dict[str, Any]:
    """A run-list row derived from a complete trace.

    ``total_cost`` is first-class on the trace; status / latency / tokens / model
    are derived best-effort from the observations (the contract has no
    first-class per-run latency/token/status). A trace the backend kept but could
    not load (``fetch_error`` set) carries no observations — surfaced via
    ``fetchError`` so it does not read as an empty *success*."""
    obs = trace.observations or []
    starts = [o.start for o in obs if o.start]
    ends = [o.end for o in obs if o.end]
    latency_ms: int | None = None
    if starts and ends:
        latency_ms = int((max(ends) - min(starts)).total_seconds() * 1000)

    has_error = any((o.level or "").upper() == "ERROR" for o in obs)
    tokens = 0
    for o in obs:
        if isinstance(o.usage, dict):
            for key in ("input", "output"):
                tokens += int(_num(o.usage.get(key)))
    model = next((o.model for o in obs if o.model), None)

    return {
        "id": trace.id,
        "traceId": trace.id,
        "createdAt": trace.timestamp.isoformat() if trace.timestamp else None,
        "tags": list(trace.tags or []),
        "status": "error" if has_error else "success",
        "cost": trace.total_cost,
        "latencyMs": latency_ms,
        "totalTokens": tokens or None,
        "model": model,
        "inputPreview": _preview(trace.input),
        "outputPreview": _preview(trace.output),
        "fetchError": trace.fetch_error,
    }


def _map_span(o: MonitoringObservation) -> dict[str, Any]:
    node_id = None
    if isinstance(o.metadata, dict):
        node_id = o.metadata.get("node_id") or o.metadata.get("nodeId")
    return {
        "id": o.id,
        "parentId": o.parent_id,
        "traceId": o.trace_id,
        "name": o.name,
        "type": o.type,
        "level": o.level,
        "statusMessage": o.status_message,
        "start": o.start.isoformat() if o.start else None,
        "end": o.end.isoformat() if o.end else None,
        "model": o.model,
        "usage": o.usage,
        "metadata": o.metadata,
        "input": o.input,
        "output": o.output,
        "nodeId": node_id,
    }


def map_trace(trace: MonitoringTrace) -> dict[str, Any]:
    """The single-run detail view: trace attributes plus every span.

    ``availability`` is ``unavailable`` when the backend kept the trace but could
    not load it (``fetch_error``), ``full`` when spans loaded, else ``partial``."""
    spans = [_map_span(o) for o in (trace.observations or [])]
    availability = "unavailable" if trace.fetch_error else "full" if spans else "partial"
    return {
        "traceId": trace.id,
        "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
        "tags": list(trace.tags or []),
        "totalCost": trace.total_cost,
        "input": trace.input,
        "output": trace.output,
        "metadata": trace.metadata,
        "availability": availability,
        "fetchError": trace.fetch_error,
        "spans": spans,
    }
