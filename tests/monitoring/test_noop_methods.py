"""Every NoOp* method is exercised: writes accept their args and do nothing,
context managers yield, the reader returns empty results, and the span handle
records nothing. This is the explicit, named no-op backend used when monitoring
is disabled — not a real backend silently degrading."""

from __future__ import annotations

from datetime import datetime

from tai_contract.monitoring import (
    MetricsFilter,
    ProjectConfig,
    SpanKind,
    TraceContext,
)

from tai_skeleton.monitoring import NoOpMonitoring, NoOpReader, NoOpSpan, NoOpWriter


def _ctx() -> TraceContext:
    return TraceContext(trace_id="t-1")


# --- span -------------------------------------------------------------------


def test_noop_span_id_and_updates_are_inert() -> None:
    span = NoOpSpan()
    assert span.id == ""
    assert span.update(output="x", model="m", usage_details={}, metadata={}, status_message="s") is None
    assert span.set_trace_metadata(name="n", tags=["a"]) is None


# --- writer -----------------------------------------------------------------


def test_noop_writer_start_span_yields_a_span() -> None:
    writer = NoOpWriter()
    with writer.start_span(name="s", kind=SpanKind.TOOL, trace_context=_ctx(), input={"a": 1}) as span:
        assert isinstance(span, NoOpSpan)


def test_noop_writer_record_and_event_are_inert() -> None:
    writer = NoOpWriter()
    now = datetime.now()
    assert writer.record_span(name="s", kind=SpanKind.LLM, start=now, end=now, trace_context=_ctx()) is None
    assert writer.create_event(name="e", trace_context=_ctx(), input="i", output="o") is None
    assert writer.update_current_span(status_message="m", metadata={}, output="o") is None


def test_noop_writer_context_managers_yield() -> None:
    writer = NoOpWriter()
    with writer.trace_attributes(name="n", tags=["t"], metadata={}):
        pass
    with writer.scope("pk"):
        pass
    with writer.disable():
        pass


def test_noop_writer_query_helpers_return_neutral_values() -> None:
    writer = NoOpWriter()
    assert writer.current_trace_id() is None
    assert writer.inject_context(_ctx()) == {}
    assert writer.get_monitoring_callbacks(_ctx()) == []


def test_noop_writer_lifecycle_calls_are_inert() -> None:
    writer = NoOpWriter()
    assert writer.flush() is None
    assert writer.shutdown() is None


# --- reader -----------------------------------------------------------------


async def test_noop_reader_returns_empty_results() -> None:
    reader = NoOpReader()
    now = datetime.now()
    from tai_contract.monitoring import MetricsResult

    result = await reader.query_metrics(MetricsFilter(metrics=[], from_timestamp=now, to_timestamp=now))
    assert isinstance(result, MetricsResult)
    assert await reader.list_spans_in_window(now, now) == []
    assert await reader.list_traces() == []


# --- monitoring composite ---------------------------------------------------


def test_noop_monitoring_add_project_is_inert() -> None:
    monitoring = NoOpMonitoring()
    project = ProjectConfig(public_key="pk", secret_key="sk", host="http://localhost")
    assert monitoring.add_project(project) is None
