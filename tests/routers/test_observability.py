"""Observability router: metrics composition, run list filter/sort/paging
parsing, single-trace detail + 404, the two typed monitoring-error mappings
(501 read-not-supported / 404 not-found), loud propagation of any other reader
error, per-request reader fetch, and the CSV/JSON exports.

Handlers are driven directly as coroutines with real ``Request`` objects
carrying a query string. The monitoring backend is a fake in-memory reader
implementing the contract's async ``MonitoringReader`` protocol, installed
through the process monitoring registry (``register_monitoring`` /
``reset_monitoring``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import urlencode

import pytest
from starlette.requests import Request
from tai_contract.monitoring import (
    MetricsResult,
    MetricsRow,
    MetricsView,
    MonitoringLevel,
    MonitoringObservation,
    MonitoringReadNotSupportedError,
    MonitoringTrace,
    TraceNotFoundError,
)

from tai_skeleton.monitoring.registry import register_monitoring, reset_monitoring
from tai_skeleton.routers import observability as router

_T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
_T1 = datetime(2026, 7, 1, 12, 0, 2, tzinfo=UTC)


# -- request + response helpers ----------------------------------------------


def _req(query: str = "", **path_params) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/observability",
        "headers": [],
        "query_string": query.encode(),
        "path_params": path_params,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _q(**params) -> str:
    return urlencode(params)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


# -- fake monitoring backend (async contract reader) -------------------------


class _FakeWriter:
    def shutdown(self) -> None:
        pass


class _FakeMonitoring:
    def __init__(self, reader: _FakeReader) -> None:
        self._reader = reader
        self._writer = _FakeWriter()

    @property
    def reader(self) -> _FakeReader:
        return self._reader

    @property
    def writer(self) -> _FakeWriter:
        return self._writer

    def add_project(self, project) -> None:
        pass


class _FakeReader:
    def __init__(self) -> None:
        self.summary_rows: list[MetricsRow] = []
        self.series_rows: list[MetricsRow] = []
        self.model_rows: list[MetricsRow] = []
        self.traces: list[MonitoringTrace] = []
        self.traces_by_id: dict[str, MonitoringTrace] = {}
        self.query_error: Exception | None = None
        self.list_error: Exception | None = None
        self.get_trace_error: Exception | None = None
        self.list_calls: list[dict] = []

    async def query_metrics(self, filter) -> MetricsResult:
        if self.query_error is not None:
            raise self.query_error
        if filter.view == MetricsView.OBSERVATIONS:
            return MetricsResult(rows=self.model_rows)
        if filter.granularity:
            return MetricsResult(rows=self.series_rows)
        return MetricsResult(rows=self.summary_rows)

    async def list_traces(
        self, *, from_timestamp=None, to_timestamp=None, limit=None, page=None, filter=None, order_by=None
    ) -> list[MonitoringTrace]:
        self.list_calls.append({"limit": limit, "page": page, "filter": filter, "order_by": order_by})
        if self.list_error is not None:
            raise self.list_error
        return self.traces

    async def get_trace(self, trace_id: str) -> MonitoringTrace:
        if self.get_trace_error is not None:
            raise self.get_trace_error
        trace = self.traces_by_id.get(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"trace {trace_id!r} not found")
        return trace

    async def list_spans_in_window(self, t0, t1, *, run=None, kind=None, filter=None, order_by=None):
        return []


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_monitoring()
    yield
    reset_monitoring()


def _install(reader: _FakeReader) -> _FakeReader:
    register_monitoring(lambda: _FakeMonitoring(reader))
    return reader


def _obs(**kw) -> MonitoringObservation:
    return MonitoringObservation(**kw)


def _trace(**kw) -> MonitoringTrace:
    return MonitoringTrace(**kw)


# -- GET /api/observability/metrics ------------------------------------------


async def test_metrics_composed_shape():
    reader = _install(_FakeReader())
    reader.summary_rows = [MetricsRow(metrics={"count": 10, "totalCost": 2.5, "totalTokens": 1000, "latency": 1500})]
    reader.series_rows = [
        MetricsRow(
            dimensions={"time": "2026-07-01"},
            metrics={"count": 4, "totalCost": 1.0, "totalTokens": 400, "latency": 1200},
        ),
        MetricsRow(
            dimensions={"time": "2026-07-02"},
            metrics={"count": 6, "totalCost": 1.5, "totalTokens": 600, "latency": 1800},
        ),
    ]
    reader.model_rows = [
        MetricsRow(
            dimensions={"providedModelName": "gpt-4o"},
            metrics={"count": 7, "totalCost": 2.0, "totalTokens": 700, "latency": 1600},
        ),
        MetricsRow(
            dimensions={"providedModelName": "claude"},
            metrics={"count": 3, "totalCost": 0.5, "totalTokens": 300, "latency": 900},
        ),
    ]

    resp = await router.get_metrics(_req("from=7d&granularity=day"))
    assert resp.status_code == 200
    data = _json(resp)["data"]

    assert data["granularity"] == "day"
    assert data["summary"] == {
        "totalRuns": 10,
        "totalCost": 2.5,
        "totalTokens": 1000,
        "averageLatencyMs": 1500,
        "avgCostPerRun": 0.25,
        "avgTokensPerRun": 100,
        "timeToFirstTokenMs": None,
    }
    assert data["timeSeries"] == [
        {"bucket": "2026-07-01", "runs": 4, "cost": 1.0, "avgLatencyMs": 1200, "totalTokens": 400},
        {"bucket": "2026-07-02", "runs": 6, "cost": 1.5, "avgLatencyMs": 1800, "totalTokens": 600},
    ]
    # Sorted by cost desc, top 8.
    assert data["byModel"] == [
        {"model": "gpt-4o", "calls": 7, "cost": 2.0, "totalTokens": 700, "avgLatencyMs": 1600},
        {"model": "claude", "calls": 3, "cost": 0.5, "totalTokens": 300, "avgLatencyMs": 900},
    ]


async def test_metrics_empty_rows_zero_summary():
    _install(_FakeReader())  # all row sets empty
    resp = await router.get_metrics(_req(""))
    data = _json(resp)["data"]
    assert data["summary"]["totalRuns"] == 0
    assert data["summary"]["avgCostPerRun"] == 0.0
    assert data["timeSeries"] == []
    assert data["byModel"] == []


async def test_metrics_bymodel_omitted_when_subquery_rejected():
    reader = _install(_FakeReader())
    reader.summary_rows = [MetricsRow(metrics={"count": 1})]

    # The by-model view is the only OBSERVATIONS query; reject just that one.
    async def query(filter):
        if filter.view == MetricsView.OBSERVATIONS:
            raise MonitoringReadNotSupportedError("no observations metrics")
        return MetricsResult(rows=reader.summary_rows if not filter.granularity else [])

    reader.query_metrics = query  # type: ignore[method-assign]
    resp = await router.get_metrics(_req(""))
    assert resp.status_code == 200
    assert _json(resp)["data"]["byModel"] == []  # omitted, tiles intact


async def test_metrics_bad_token_400():
    _install(_FakeReader())
    resp = await router.get_metrics(_req("from=nonsense"))
    assert resp.status_code == 400
    assert "Invalid from" in _json(resp)["error"]


async def test_metrics_inverted_range_400():
    _install(_FakeReader())
    resp = await router.get_metrics(_req("from=1d&to=10d"))
    assert resp.status_code == 400
    assert "before" in _json(resp)["error"]


async def test_metrics_bad_granularity_400():
    # An explicit granularity outside {hour, day, week} is malformed — it must be a
    # loud 400, not silently fall through to auto-selection.
    _install(_FakeReader())
    resp = await router.get_metrics(_req("granularity=bogus"))
    assert resp.status_code == 400
    assert "granularity" in _json(resp)["error"]


# -- GET /api/observability/runs ---------------------------------------------


async def test_runs_list_and_derive_shape():
    reader = _install(_FakeReader())
    obs = [
        _obs(
            id="o1",
            trace_id="t1",
            type="LLM",
            name="call",
            level="DEFAULT",
            model="gpt-4o",
            usage={"input": 100, "output": 50},
            start=_T0,
            end=_T1,
        )
    ]
    reader.traces = [
        _trace(id="t1", timestamp=_T0, tags=["a"], total_cost=0.3, observations=obs, input={"q": "hi"}, output="done")
    ]

    resp = await router.list_runs(_req(""))
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert data["page"] == 1
    assert data["nextPage"] is None
    assert data["items"][0] == {
        "id": "t1",
        "traceId": "t1",
        "createdAt": _T0.isoformat(),
        "tags": ["a"],
        "status": "success",
        "cost": 0.3,
        "latencyMs": 2000,
        "totalTokens": 150,
        "model": "gpt-4o",
        "inputPreview": {"q": "hi"},
        "outputPreview": "done",
        "fetchError": None,
    }


async def test_runs_error_status_derived_from_observation_level():
    reader = _install(_FakeReader())
    obs = [_obs(id="o1", trace_id="t1", level="ERROR", start=_T0, end=_T1)]
    reader.traces = [_trace(id="t1", timestamp=_T0, total_cost=0.0, observations=obs)]
    resp = await router.list_runs(_req(""))
    assert _json(resp)["data"]["items"][0]["status"] == "error"


async def test_runs_tags_json_list_decoded():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(tags='["a","b"]')))
    assert reader.list_calls[0]["filter"].tags == ["a", "b"]


async def test_runs_tags_comma_decoded():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(tags="x, y ,z")))
    assert reader.list_calls[0]["filter"].tags == ["x", "y", "z"]


async def test_runs_status_error_maps_to_error_level():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(status="error")))
    assert reader.list_calls[0]["filter"].level == MonitoringLevel.ERROR


async def test_runs_status_success_is_unfiltered():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(status="success")))
    assert reader.list_calls[0]["filter"] is None


async def test_runs_bad_status_400():
    _install(_FakeReader())
    resp = await router.list_runs(_req(_q(status="weird")))
    assert resp.status_code == 400


async def test_runs_sort_and_dir_map_to_order_by():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(sort="cost", dir="asc")))
    order = reader.list_calls[0]["order_by"]
    assert order.field == "total_cost"
    assert order.direction == "asc"


async def test_runs_bad_sort_400():
    _install(_FakeReader())
    resp = await router.list_runs(_req(_q(sort="nope")))
    assert resp.status_code == 400


async def test_runs_latency_ms_converted_to_seconds():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(minLatencyMs="2000")))
    assert reader.list_calls[0]["filter"].min_latency == 2.0


async def test_runs_inverted_cost_range_400():
    _install(_FakeReader())
    resp = await router.list_runs(_req(_q(minCost="5", maxCost="1")))
    assert resp.status_code == 400


async def test_runs_pagesize_non_int_400():
    _install(_FakeReader())
    resp = await router.list_runs(_req(_q(pageSize="abc")))
    assert resp.status_code == 400


async def test_runs_pagesize_capped_at_100():
    reader = _install(_FakeReader())
    await router.list_runs(_req(_q(pageSize="500")))
    assert reader.list_calls[0]["limit"] == 100


async def test_runs_paging_sub_one_is_400():
    # A 0/negative page or pageSize is malformed → rejected as a 400, never
    # silently clamped to a floor of 1.
    _install(_FakeReader())
    resp = await router.list_runs(_req(_q(page="0", pageSize="0")))
    assert resp.status_code == 400


async def test_runs_next_page_when_full():
    reader = _install(_FakeReader())
    reader.traces = [_trace(id="t1"), _trace(id="t2")]
    resp = await router.list_runs(_req(_q(pageSize="1", page="1")))
    assert _json(resp)["data"]["nextPage"] == 2


# -- GET /api/observability/runs/{trace_id}/trace ----------------------------


async def test_get_trace_maps_full_detail():
    reader = _install(_FakeReader())
    obs = _obs(
        id="o1",
        trace_id="t1",
        parent_id=None,
        type="LLM",
        name="call",
        level="DEFAULT",
        status_message=None,
        model="gpt-4o",
        usage={"input": 1},
        start=_T0,
        end=_T1,
        metadata={"node_id": "n1"},
        input={"p": 1},
        output="ok",
    )
    reader.traces_by_id["t1"] = _trace(
        id="t1",
        timestamp=_T0,
        tags=["a"],
        total_cost=0.5,
        input={"q": "hi"},
        output="done",
        metadata={"m": 1},
        observations=[obs],
    )

    resp = await router.get_run_trace(_req("", trace_id="t1"))
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert data["availability"] == "full"
    assert data == {
        "traceId": "t1",
        "timestamp": _T0.isoformat(),
        "tags": ["a"],
        "totalCost": 0.5,
        "input": {"q": "hi"},
        "output": "done",
        "metadata": {"m": 1},
        "availability": "full",
        "fetchError": None,
        "spans": [
            {
                "id": "o1",
                "parentId": None,
                "traceId": "t1",
                "name": "call",
                "type": "LLM",
                "level": "DEFAULT",
                "statusMessage": None,
                "start": _T0.isoformat(),
                "end": _T1.isoformat(),
                "model": "gpt-4o",
                "usage": {"input": 1},
                "metadata": {"node_id": "n1"},
                "input": {"p": 1},
                "output": "ok",
                "nodeId": "n1",
            }
        ],
    }


async def test_get_trace_unknown_404():
    _install(_FakeReader())  # empty store → get_trace raises TraceNotFoundError
    resp = await router.get_run_trace(_req("", trace_id="missing"))
    assert resp.status_code == 404
    assert _json(resp)["error"] == "Run trace not found"


# -- typed error mappings + loud propagation ---------------------------------


async def test_metrics_read_not_supported_501():
    reader = _install(_FakeReader())
    reader.query_error = MonitoringReadNotSupportedError("read disabled")
    resp = await router.get_metrics(_req(""))
    assert resp.status_code == 501
    body = _json(resp)
    assert body["code"] == "monitoring-read-not-supported"
    assert body["error"] == "read disabled"


async def test_runs_read_not_supported_501():
    reader = _install(_FakeReader())
    reader.list_error = MonitoringReadNotSupportedError("read disabled")
    resp = await router.list_runs(_req(""))
    assert resp.status_code == 501
    assert _json(resp)["code"] == "monitoring-read-not-supported"


async def test_trace_read_not_supported_501():
    reader = _install(_FakeReader())
    reader.get_trace_error = MonitoringReadNotSupportedError("read disabled")
    resp = await router.get_run_trace(_req("", trace_id="t1"))
    assert resp.status_code == 501
    assert _json(resp)["code"] == "monitoring-read-not-supported"


async def test_noop_reader_yields_empty_not_501():
    # No backend installed → the process no-op reader answers with empty data.
    resp = await router.get_metrics(_req(""))
    assert resp.status_code == 200
    assert _json(resp)["data"]["summary"]["totalRuns"] == 0
    resp_runs = await router.list_runs(_req(""))
    assert resp_runs.status_code == 200
    assert _json(resp_runs)["data"]["items"] == []


async def test_plain_reader_error_propagates_loudly():
    reader = _install(_FakeReader())
    reader.query_error = RuntimeError("backend exploded")
    with pytest.raises(RuntimeError):
        await router.get_metrics(_req(""))


async def test_plain_list_error_propagates_loudly():
    reader = _install(_FakeReader())
    reader.list_error = RuntimeError("backend exploded")
    with pytest.raises(RuntimeError):
        await router.list_runs(_req(""))


async def test_swapping_reader_changes_response():
    r1 = _install(_FakeReader())
    r1.summary_rows = [MetricsRow(metrics={"count": 5})]
    resp1 = await router.get_metrics(_req(""))
    assert _json(resp1)["data"]["summary"]["totalRuns"] == 5

    r2 = _install(_FakeReader())  # swap the registered backend
    r2.summary_rows = [MetricsRow(metrics={"count": 9})]
    resp2 = await router.get_metrics(_req(""))
    assert _json(resp2)["data"]["summary"]["totalRuns"] == 9


# -- exports -----------------------------------------------------------------


async def test_export_trace_json_download():
    reader = _install(_FakeReader())
    reader.traces_by_id["t1"] = _trace(id="t1", timestamp=_T0, total_cost=0.1, observations=[])

    resp = await router.export_run_trace(_req("", trace_id="t1"))
    assert resp.status_code == 200
    assert resp.media_type == "application/json"
    assert resp.headers["Content-Disposition"] == 'attachment; filename="trace-t1.json"'
    body = json.loads(bytes(resp.body))
    assert body["traceId"] == "t1"
    assert body["availability"] == "partial"  # no spans


async def test_export_trace_unknown_404():
    _install(_FakeReader())
    resp = await router.export_run_trace(_req("", trace_id="missing"))
    assert resp.status_code == 404


async def test_export_runs_json_download():
    reader = _install(_FakeReader())
    reader.traces = [_trace(id="t1", timestamp=_T0, total_cost=0.2, observations=[])]

    resp = await router.export_runs(_req(_q(format="json")))
    assert resp.media_type == "application/json"
    assert resp.headers["Content-Disposition"] == 'attachment; filename="runs.json"'
    body = json.loads(bytes(resp.body))
    assert body["truncated"] is False
    assert body["items"][0]["traceId"] == "t1"


async def test_export_runs_csv_download_and_injection_guard():
    reader = _install(_FakeReader())
    # An output preview whose leading char a spreadsheet reads as a formula.
    reader.traces = [_trace(id="t1", timestamp=_T0, total_cost=0.2, output="=SUM(A1)", observations=[])]

    resp = await router.export_runs(_req(""))  # csv is the default
    assert resp.media_type == "text/csv"
    assert resp.headers["Content-Disposition"] == 'attachment; filename="runs.csv"'
    text = bytes(resp.body).decode()
    header = text.splitlines()[0]
    expected_header = "traceId,createdAt,status,fetchError,cost,latencyMs,totalTokens,model,inputPreview,outputPreview"
    assert header.startswith(expected_header)
    # The formula-leading preview is prefixed with a quote (CSV-injection guard).
    assert "'=SUM(A1)" in text


async def test_export_runs_bad_format_400():
    _install(_FakeReader())
    resp = await router.export_runs(_req(_q(format="xml")))
    assert resp.status_code == 400


async def test_export_trace_read_not_supported_501():
    # The native trace-download handler maps a read-not-supported backend to the same
    # 501 + ``code`` the enveloped reads answer (via the module-local ``_not_supported``).
    reader = _install(_FakeReader())
    reader.get_trace_error = MonitoringReadNotSupportedError("read disabled")
    resp = await router.export_run_trace(_req("", trace_id="t1"))
    assert resp.status_code == 501
    assert _json(resp)["code"] == "monitoring-read-not-supported"


async def test_export_runs_read_not_supported_501():
    reader = _install(_FakeReader())
    reader.list_error = MonitoringReadNotSupportedError("read disabled")
    resp = await router.export_runs(_req(""))
    assert resp.status_code == 501
    assert _json(resp)["code"] == "monitoring-read-not-supported"
