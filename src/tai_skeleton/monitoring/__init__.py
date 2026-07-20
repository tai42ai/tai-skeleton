"""Monitoring impl owned by the skeleton: the process-global registry and the
no-op default backend.

The vendor-neutral monitoring *contract* (protocols, models, errors) lives in
``tai_contract.monitoring`` — import interfaces from there. This package owns
only the concrete pieces the framework ships: the registry that holds the active
backend and the explicit ``NoOp*`` default used when monitoring is disabled. Real
backends (e.g. Langfuse) are external plugins installed via
``@tai_app.monitoring.register_monitoring``.
"""

from __future__ import annotations

from tai_skeleton.monitoring.noop import (
    NoOpMonitoring,
    NoOpReader,
    NoOpSpan,
    NoOpWriter,
)
from tai_skeleton.monitoring.registry import (
    get_monitoring,
    init_monitoring,
    register_monitoring,
    reset_monitoring,
)

__all__ = [
    "NoOpMonitoring",
    "NoOpReader",
    "NoOpSpan",
    "NoOpWriter",
    "get_monitoring",
    "init_monitoring",
    "register_monitoring",
    "reset_monitoring",
]
