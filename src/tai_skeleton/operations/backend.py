"""Backend identity + the fleet census and the fleet soft-restart.

``backend_info`` reports whether a task :class:`~tai_contract.backend.Backend`
provider is registered (identity only). The other two operations are FLEET
operations over the app's worker bus, and need no backend at all:

* ``list_workers`` — the live worker fleet from the bus presence census (every
  subscribed process, HTTP server and backend runtime alike). No try/except: a
  presence-store read that cannot read must surface as a ``500``, never an empty
  ``[]``.
* ``fleet_reload_config`` — soft-restart the fleet. The serving worker applies its
  own reload FIRST, then broadcasts the reload; the response embeds the per-origin
  fleet report. A failed local apply does NOT abort
  the broadcast — the door's whole purpose is converging siblings onto persisted
  state — so it publishes anyway and re-raises with the report attached.

``fleet_reload_config`` is named apart from the local-reload ``reload_config`` op
(a different door, ``/api/config/reload``): the fleet soft-restart and the
in-process config reload share no surface.
"""

from __future__ import annotations

from pydantic import BaseModel
from tai_contract.app import tai_app

from tai_skeleton.app import instance
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.operations import BadRequestError, operation
from tai_skeleton.operations._broadcast import broadcast


class ReloadConfig(BaseModel):
    """A fleet reload-config request — an optional ``targets`` list restricting the
    soft-restart to named workers (all workers when omitted)."""

    targets: list[str] | None = None


@operation(summary="Get the backend identity", tags=["backend"])
async def backend_info() -> dict:
    backend = tai_app.backends.backend
    if backend is None:
        return {"present": False, "backend": None, "module": None}
    return {"present": True, "backend": type(backend).__name__, "module": type(backend).__module__}


@operation(summary="List the worker fleet", tags=["backend"])
async def list_workers() -> dict:
    # The census IS the fleet listing now — every process on the bus, via its
    # presence key. No try/except: a presence-store read that cannot read must fail
    # loudly (500), never return an empty fleet.
    return {"workers": [origin.model_dump(mode="json") for origin in await instance.app.bus.census()]}


@operation(
    name="fleet_reload_config",
    summary="Soft-restart the worker fleet",
    tags=["backend"],
    destructive=True,
    errors=[BadRequestError],
    request_model=ReloadConfig,
)
async def fleet_reload_config(targets: list[str] | None) -> dict:
    # ``targets`` is validated at the HTTP edge (the route's ``_reload_targets``
    # extractor raises ``BadRequestError`` for a malformed body / non-string-list
    # targets), so the route answers a loud 400 there — the operation receives an
    # already-typed ``targets`` and declares ``BadRequestError`` so the surface
    # documents that 400. The bus then validates the target NAMES against the census.
    #
    # Convergence semantics: the serving worker applies its reload locally first,
    # but a failed local apply must not abort the broadcast — the recovery door
    # exists to heal the exact stale siblings a failed local reload would strand —
    # so it publishes anyway and re-raises with the fleet report attached.
    return await broadcast(
        {"op": "reload_config"},
        targets,
        lambda: reload_gate.run(tai_app.admin.reload_config),
        publish_on_local_failure=True,
    )
