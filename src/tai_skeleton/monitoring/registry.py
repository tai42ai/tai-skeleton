"""Process-global registry for the monitoring backend.

Access is a free global (``get_monitoring()``), not hung off the app:
monitoring is foundational infra below the app, most emit sites have no app
handle, and the framework's own internals emit (so they cannot import the app
to reach it).

Monitoring is optional: with no plugin registered, ``get_monitoring()`` returns
a shared ``NoOpMonitoring`` default — there is exactly one place a real backend
is installed, the ``@tai_app.monitoring.register_monitoring`` plugin, mirroring how
backend/template register. Callers never seed the no-op themselves.

The registered backend is plain process memory: a forked worker child inherits
it across ``fork()``, so ``get_monitoring()`` keeps working post-fork —
``init_monitoring()`` is not re-called per child. Only the inner vendor client
dies on fork; the writer's ``shutdown()`` evicts and rebuilds it.
"""

from __future__ import annotations

from tai_contract.monitoring import Monitoring

from tai_skeleton.monitoring.noop import NoOpMonitoring

_backend: Monitoring | None = None


def init_monitoring(backend: Monitoring) -> None:
    """Register the monitoring backend, installed by a monitoring plugin
    (``@tai_app.monitoring.register_monitoring``).

    A reload re-imports the monitoring module and re-fires the decorator, so this
    can run again with a fresh backend. Shut down the previously-installed
    backend's writer first, so its background flush thread / vendor client is not
    leaked when it is replaced. The no-op default's ``shutdown`` is a no-op."""
    global _backend
    if _backend is not None:
        _backend.writer.shutdown()
    _backend = backend


def register_monitoring(builder=None):
    """Decorator installing the process monitoring backend (manifest
    ``monitoring_module``) — the ``app.monitoring`` facet body.

    A monitoring plugin (e.g. the Langfuse impl) decorates a zero-arg callable
    that returns a ``Monitoring``; it is built and installed via
    ``init_monitoring``, replacing the no-op default. One provider per process,
    last registration wins. The skeleton never names a concrete vendor — the
    plugin is selected purely by the manifest.
    """
    if builder:
        return register_monitoring()(builder)

    def decorator(fn):
        init_monitoring(fn())
        return fn

    return decorator


def reset_monitoring() -> None:
    """Clear any registered backend so ``get_monitoring()`` falls back to the
    no-op default. For test isolation: a test that installs its own recording
    backend resets here (typically via an autouse fixture) so it cannot leak into
    the next test. Not a production path — a real backend is registered once via
    the monitoring plugin."""
    global _backend
    _backend = None


def get_monitoring() -> Monitoring:
    """Return the registered backend, or a shared no-op default if none is set.

    Monitoring being absent is a valid 'disabled' state, not a failure: the first
    call with nothing registered installs a process-wide ``NoOpMonitoring`` (writes
    do nothing, reads return empty) and returns it. A real backend registered via
    ``init_monitoring`` replaces it. Callers never have to initialize monitoring.
    """
    global _backend
    if _backend is None:
        _backend = NoOpMonitoring()
    return _backend
