"""Boot-time worker-bus invariants — refuse to start a deployment that needs the
bus but has none configured.

The worker bus is what keeps sibling workers from serving stale config after a
reload, so three shapes require ``TAI_BUS_REDIS_URL``:

* more than one server worker (siblings would diverge on a reload),
* a task backend registered in the manifest (the backend-runtime and server
  processes must converge), and
* ``TAI_CONFIG_MODE=k8s`` (multi-pod shared config — the replica count is
  undetectable from inside one pod).

A single-worker, file-mode process with no backend is the supported busless shape
and runs on :meth:`WorkerBus.local`.

The k8s and workers checks read only boot-fixed env / args, so they run at the CLI
BEFORE the config manager is constructed — a busless k8s boot then refuses naming
``TAI_BUS_REDIS_URL`` rather than failing first on a kubeconfig connection. The
backend check needs the manifest, so it also runs at the ``app_context`` seam that
both ``tai serve`` and ``tai backend`` cross, and re-runs there on every reload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tai_skeleton.app.bus_settings import bus_settings
from tai_skeleton.config.config_mode import config_mode

if TYPE_CHECKING:
    from tai_skeleton.manifest import Manifest

_BUS_VAR = "TAI_BUS_REDIS_URL"


class BackendNeedsBusError(RuntimeError):
    """A resolved config registers a task backend while no worker bus is configured.

    The backend-runtime and server processes must converge on config reloads, which
    needs the bus. This is the one backend-needs-bus predicate, evaluated against the
    resolved config at boot, at every reload, and at mutation time — so the same
    invariant rejects a manifest/env change that ADDS a backend with no bus and one
    that REMOVES the bus while a backend still needs it."""


def _bus_configured() -> bool:
    return bus_settings().enabled


def check_backend_needs_bus(*, backend_module: str | None, bus_configured: bool) -> None:
    """Raise when a resolved config registers a task backend with no bus configured.

    The reusable core of the backend-needs-bus invariant: ``backend_module`` is the
    resolved manifest's backend module (empty / ``None`` when none is registered) and
    ``bus_configured`` is whether the effective env configures the bus. The mutation
    pipeline evaluates both against the POST-change resolved config, so this one
    predicate covers both directions — a change that adds a backend without a bus and
    a change that drops the bus while a backend remains."""
    if backend_module and not bus_configured:
        raise BackendNeedsBusError(
            f"Refusing a config that registers a task backend ({backend_module!r}) with no worker bus: "
            f"the backend-runtime and server processes must converge on config reloads. Set {_BUS_VAR}."
        )


def require_bus_for_workers(workers: int) -> None:
    """Refuse a multi-worker server with no bus: sibling workers would serve stale
    state after a reload with no channel to converge on."""
    if workers > 1 and not _bus_configured():
        raise RuntimeError(
            f"Refusing to start {workers} workers without the worker bus: sibling workers would serve "
            f"stale config after a reload, with no channel to converge on. Set {_BUS_VAR} to enable the "
            "bus, or run a single worker."
        )


def require_bus_for_k8s() -> None:
    """Refuse a k8s-mode boot with no bus: k8s config mode exists for multi-pod
    shared config, and a pod cannot see its own replica count."""
    if config_mode() == "k8s" and not _bus_configured():
        raise RuntimeError(
            f"Refusing to start in k8s config mode without the worker bus: k8s mode exists for multi-pod "
            f"shared config and the replica count is undetectable from inside one pod. Set {_BUS_VAR}."
        )


def require_bus_for_backend(manifest: Manifest) -> None:
    """Refuse a boot / reload that registers a task backend with no bus: the
    backend-runtime and server processes must converge on config reloads.

    The boot-time and reload-time face of :func:`check_backend_needs_bus`, reading
    the process-wide bus configuration — so the mutate-time pipeline check and this
    one agree on the same predicate."""
    check_backend_needs_bus(backend_module=manifest.backend_module, bus_configured=_bus_configured())
