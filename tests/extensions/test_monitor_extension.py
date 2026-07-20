"""Tests for the ``monitor`` builtin tool extension.

Driven through the REAL apply site (``app.app_context`` with a manifest naming
the builtin as an ``extensions_modules`` entry and attaching it via
``extensions: {shout: [monitor]}``):
the WRAPPER schema rule is enforced at bind time, so a clean bind proves the
extension re-presents the tool's input schema unchanged. A recording monitoring
backend — built on the contract ``Monitoring`` protocol — captures the emitted
span so its name / ``SpanKind`` / level are asserted.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pytest
from fastmcp.utilities.types import get_cached_typeadapter
from tai42_contract.monitoring import (
    Monitoring,
    MonitoringLevel,
    Span,
    SpanKind,
    TraceContext,
)

from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.monitoring import (
    NoOpMonitoring,
    NoOpReader,
    NoOpSpan,
    NoOpWriter,
    init_monitoring,
    reset_monitoring,
)

_BUILTIN_MODULE = "tai42_skeleton.extensions.builtin.monitor"


# --- recording monitoring fake ------------------------------------------------


class _RecordingSpan(NoOpSpan):
    """Span handle that records every ``update`` call for later assertions."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage_details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        level: MonitoringLevel | None = None,
        status_message: str | None = None,
    ) -> None:
        self.updates.append(
            {
                "output": output,
                "model": model,
                "usage_details": usage_details,
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
            }
        )


class _RecordingWriter(NoOpWriter):
    """Writer that records each opened span. ``active_trace_id`` seeds
    ``current_trace_id`` so the monitor's suppress-under-active-trace branch can
    be exercised."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.active_trace_id: str | None = None

    def current_trace_id(self) -> str | None:
        return self.active_trace_id

    @contextmanager
    def start_span(
        self,
        *,
        name: str,
        kind: SpanKind,
        trace_context: TraceContext | None = None,
        input: Any = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        span = _RecordingSpan()
        self.spans.append({"name": name, "kind": kind, "input": input, "span": span})
        yield span


class _RecordingMonitoring(NoOpMonitoring):
    """Backend whose writer records spans; reader stays the no-op."""

    def __init__(self) -> None:
        super().__init__()
        self._recording_writer = _RecordingWriter()

    @property
    def writer(self) -> _RecordingWriter:
        return self._recording_writer

    @property
    def reader(self) -> NoOpReader:
        return self._reader


@pytest.fixture(autouse=True)
def _reset_monitoring() -> Iterator[None]:
    """Each test starts and ends with no backend registered, so the process-global
    monitoring registry cannot leak a backend across tests."""
    reset_monitoring()
    yield
    reset_monitoring()


@pytest.fixture(autouse=True)
def _clean_server() -> Iterator[None]:
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior test bound would collide with this
    test's bind under ``on_duplicate="error"``."""

    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


def _manifest(tool: str, *extensions: str, tool_module: str = "tests.app._fixtures.tools_b") -> Manifest:
    # ``include`` selects the tool; the ``extensions`` map attaches the combo
    # formed by ``extensions`` — no colon parsing.
    return Manifest.model_validate(
        {
            "extensions_modules": [_BUILTIN_MODULE],
            "tools": [
                {
                    "title": "fxt",
                    "module": tool_module,
                    "include": [tool],
                    "extensions": {tool: [list(extensions)]},
                }
            ],
        }
    )


# --- schema identity ----------------------------------------------------------


def test_monitor_presents_identical_input_schema():
    # The composed/presented signature must equal the original's, so the wrapper
    # is schema-transparent (what the apply-site kind enforcement compares).
    manifest = Manifest.model_validate({"extensions_modules": [_BUILTIN_MODULE]})

    def base(text: str, count: int = 2) -> str:
        return text * count

    async def run() -> None:
        async with app.app_context(manifest):
            factory = app._extension_registry.get_extension("monitor")
            monitored = factory(base, "base", "desc")
            assert monitored.__name__ == "base_monitor"
            # Presented signature is byte-identical to the original's.
            assert inspect.signature(monitored) == inspect.signature(base)
            # And so is the derived input JSON schema (the apply-site comparison).
            assert get_cached_typeadapter(monitored).json_schema() == get_cached_typeadapter(base).json_schema()

    asyncio.run(run())


def test_monitor_is_config_agnostic_and_rejects_config():
    # monitor takes no author config: it is a config-agnostic three-argument factory,
    # so ``factory_accepts_config`` is False and binding a non-empty config to it
    # raises loudly at the apply site (no silent drop).
    from tai42_skeleton.extensions.builtin.monitor import monitor
    from tai42_skeleton.extensions.registry import factory_accepts_config

    assert factory_accepts_config(monitor) is False

    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_BUILTIN_MODULE],
            "tools": [
                {
                    "title": "fxm",
                    "module": "tests.extensions._fixtures.tools_external",
                    "include": ["make_signature"],
                    "extensions": {"make_signature": [[{"name": "monitor", "config": {"anything": 1}}]]},
                }
            ],
        }
    )

    async def run() -> None:
        with pytest.raises(ValueError, match="does not accept config"):
            async with app.app_context(manifest):
                await app.tools.get_tools()

    asyncio.run(run())


def test_monitor_binds_as_wrapper_branch_at_apply_site():
    # A clean bind through the real apply site proves the WRAPPER schema rule is
    # satisfied — a schema change would raise ``ValidationError`` here.
    async def run() -> None:
        async with app.app_context(_manifest("shout", "monitor")):
            tools = await app.tools.get_tools()
            assert {"shout", "shout_monitor"} <= set(tools)

    asyncio.run(run())


# --- span emission ------------------------------------------------------------


def test_monitor_emits_span_with_name_kind_and_output():
    backend = _RecordingMonitoring()
    init_monitoring(backend)

    async def run() -> str:
        async with app.app_context(_manifest("shout", "monitor")):
            return await app.tools.run_tool("shout_monitor", {"text": "hi"})

    result = asyncio.run(run())

    assert result == "hi"
    assert len(backend.writer.spans) == 1
    span = backend.writer.spans[0]
    assert span["name"] == "shout"
    assert span["kind"] is SpanKind.TOOL
    captured = span["input"]
    assert "hi" in captured["args"] or captured["kwargs"].get("text") == "hi"
    updates = span["span"].updates
    assert any(u["output"] == "hi" for u in updates)
    # The happy path never escalates the level.
    assert not any(u["level"] is MonitoringLevel.ERROR for u in updates)


def test_monitor_suppresses_span_when_a_trace_is_active():
    backend = _RecordingMonitoring()
    backend.writer.active_trace_id = "trace-1"
    init_monitoring(backend)

    async def run() -> str:
        async with app.app_context(_manifest("shout", "monitor")):
            return await app.tools.run_tool("shout_monitor", {"text": "hi"})

    result = asyncio.run(run())

    # Tool still runs; no standalone span is opened while a trace owns the call.
    assert result == "hi"
    assert backend.writer.spans == []


def test_monitor_marks_error_level_and_reraises():
    backend = _RecordingMonitoring()
    init_monitoring(backend)

    async def run() -> None:
        async with app.app_context(_manifest("boom", "monitor", tool_module="tests.extensions._fixtures.tools_boom")):
            with pytest.raises(RuntimeError, match="kaboom"):
                await app.tools.run_tool("boom_monitor", {})

    asyncio.run(run())

    assert len(backend.writer.spans) == 1
    updates = backend.writer.spans[0]["span"].updates
    assert any(u["level"] is MonitoringLevel.ERROR for u in updates)
    assert any(u["status_message"] == "kaboom" for u in updates)


# --- fake conformance ---------------------------------------------------------


def test_recording_fake_conforms_to_monitoring_protocol():
    # The recording fake satisfies the contract ``Monitoring`` protocol
    # (``runtime_checkable`` checks member presence, not signatures).
    backend = _RecordingMonitoring()
    assert isinstance(backend, Monitoring)
    # ``record_span`` needs explicit times; the fake inherits the no-op form.
    now = datetime.now()
    backend.writer.record_span(
        name="x",
        kind=SpanKind.TOOL,
        start=now,
        end=now,
        trace_context=TraceContext(trace_id="t"),
    )
