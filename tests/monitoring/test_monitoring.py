"""Tests for the skeleton-owned monitoring impl: the no-op default conforms to
the contract protocols, and the registry installs/returns backends correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tai_contract.monitoring import (
    Monitoring,
    MonitoringReader,
    MonitoringWriter,
    Span,
    TraceNotFoundError,
)

from tai_skeleton.monitoring import (
    NoOpMonitoring,
    NoOpReader,
    NoOpSpan,
    NoOpWriter,
    get_monitoring,
    init_monitoring,
    reset_monitoring,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts and ends with no backend registered, so the process-global
    registry cannot leak a backend across tests."""
    reset_monitoring()
    yield
    reset_monitoring()


# --- conformance: the no-op default satisfies the contract protocols ---------


def test_noop_monitoring_conforms_to_contract_protocol():
    monitoring = NoOpMonitoring()
    assert isinstance(monitoring, Monitoring)


def test_noop_writer_conforms_to_contract_protocol():
    assert isinstance(NoOpWriter(), MonitoringWriter)


def test_noop_reader_conforms_to_contract_protocol():
    assert isinstance(NoOpReader(), MonitoringReader)


def test_noop_span_conforms_to_contract_protocol():
    assert isinstance(NoOpSpan(), Span)


def test_noop_monitoring_exposes_writer_and_reader_faces():
    monitoring = NoOpMonitoring()
    assert isinstance(monitoring.writer, MonitoringWriter)
    assert isinstance(monitoring.reader, MonitoringReader)


# --- registry behavior -------------------------------------------------------


def test_default_backend_is_the_noop():
    assert isinstance(get_monitoring(), NoOpMonitoring)


def test_default_backend_is_shared_across_calls():
    assert get_monitoring() is get_monitoring()


def test_register_a_backend_and_registry_returns_it():
    backend = NoOpMonitoring()
    init_monitoring(backend)
    assert get_monitoring() is backend


def test_reregister_shuts_down_replaced_backend_writer():
    # A reload re-fires the monitoring decorator. Installing a new backend must
    # shut down the previously-installed one's writer, so its background flush
    # thread / vendor client is not leaked when it is replaced.
    first = MagicMock()
    second = MagicMock()
    init_monitoring(first)
    init_monitoring(second)

    first.writer.shutdown.assert_called_once_with()
    second.writer.shutdown.assert_not_called()
    assert get_monitoring() is second


def test_reset_falls_back_to_a_fresh_noop_default():
    custom = NoOpMonitoring()
    init_monitoring(custom)
    assert get_monitoring() is custom

    reset_monitoring()
    fallback = get_monitoring()
    assert isinstance(fallback, NoOpMonitoring)
    assert fallback is not custom


# --- the no-op reader raises (does not silently return) on a missing trace ----


async def test_noop_reader_get_trace_raises_trace_not_found():
    with pytest.raises(TraceNotFoundError):
        await NoOpReader().get_trace("does-not-exist")
