"""The one fleet-broadcast primitive for runtime-op publishers.

Every operation that changes runtime state on this worker AND must reach the rest
of the fleet applies its change locally, then broadcasts it over the app's worker
bus (``instance.app.bus``). :func:`broadcast` is that shared step: it enforces the
two invariants every publisher shares — targets are validated against the presence
census BEFORE any local side effect, and the worker applies locally only when it is
itself a target — then awaits the confirmed broadcast and returns the per-origin
fleet report.

Failure discipline splits the callers into two disciplines, selected by
``publish_on_local_failure``:

* Pure registry/query ops (``reload_mcp``, ``deregister_mcp``, ``reload_tool``,
  ``remove_tool``, ``reload_failed_mcps``, ``list_failed_mcps``) leave it ``False``:
  a failed local apply raises before anything is broadcast, so the siblings stay
  genuinely untouched.
* Convergence and store-backed publishers (both ``reload_config`` doors, the
  connector catalog write) pass ``True``: their persist/convergence intent has
  already committed, so a failed local apply still broadcasts (siblings converge
  on the persisted state) and then re-raises with the fleet report attached — a
  stranded fleet never hides behind a local error.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from tai_skeleton.app import instance
from tai_skeleton.app.bus import FleetResult, LocalApplyResult, OpOutcome, UnknownFleetTargetsError
from tai_skeleton.operations.errors import BadRequestError

if TYPE_CHECKING:
    from tai_skeleton.config.service import ApplyResult

logger = logging.getLogger(__name__)


def fleet_fanout(fleet: FleetResult) -> dict[str, Any]:
    """Summarize a pipeline broadcast for a mutation response's ``fanout`` field.

    A single-worker deployment reaches no sibling, so the fan-out collapses to a
    human note; a reachable multi-worker broadcast returns the per-origin report, and
    an unreachable bus returns its error shape. The publisher's own origin is excluded
    when deciding local-only, so a lone worker never reports a fan-out.
    """
    if not fleet.reachable:
        return {"mode": "unreachable", **fleet.model_dump(mode="json")}
    remotes = [result for result in fleet.results if result.origin != instance.app.bus.origin.origin]
    if not remotes:
        return {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"}
    return {"mode": "fleet", **fleet.model_dump(mode="json")}


def apply_response(result: ApplyResult) -> dict[str, Any]:
    """The standard mutation-op response: the local reload result merged with the
    fan-out summary — the one shape every ConfigService writer returns from an
    :class:`~tai_skeleton.config.service.ApplyResult`."""
    return {**result.local, "fanout": result.fanout}


class FleetBroadcastError(RuntimeError):
    """Raised after a mutation has PERSISTED when the post-persist propagation does
    not fully complete — the local reload raised, the fleet broadcast raised, or both.
    The fleet report is attached (``reachable=False`` when the broadcast itself
    raised) so the caller can surface it and knows the change already landed on disk;
    the specific failure is the ``cause`` and the report's ``error``."""

    def __init__(self, op_name: str, report: FleetResult, cause: BaseException) -> None:
        super().__init__(f"{op_name}: change persisted but fleet propagation failed — {cause}")
        self.report = report


def log_non_convergence(report: FleetResult) -> None:
    """Loudly ERROR-log a reachable-but-non-converged fleet report: an unconfirmed
    origin is a visible failure, never a silently stale sibling. A bus-unreachable
    report is already logged inside :meth:`WorkerBus.publish`, and a fully converged
    report logs nothing — so this is a no-op unless the bus was reached yet some
    origin did not confirm ``applied``. Shared by every publisher (``broadcast`` and
    the store-backed helpers that publish directly) so the message stays identical."""
    if not (report.reachable and not report.ok):
        return
    unconfirmed = [(r.origin, r.outcome.value) for r in report.results if r.outcome != OpOutcome.applied]
    logger.error("worker bus: op %r did not fully converge — unconfirmed origins: %s", report.op, unconfirmed)


async def broadcast(
    op: dict[str, Any],
    targets: list[str] | None,
    apply: Callable[[], Awaitable[Any]],
    *,
    publish_on_local_failure: bool = False,
) -> dict[str, Any]:
    """Validate targets, apply locally per the self-targeting rule, then broadcast.

    ``op`` is the wire op dict (``{"op": <name>, ...}``); ``targets`` is ``None`` for
    the whole fleet or the explicit worker list; ``apply`` runs this worker's own
    apply and returns its result (rides the self entry as the payload). Returns the
    :class:`~tai_skeleton.app.bus.FleetResult` as a JSON-ready dict.
    """
    bus = instance.app.bus
    self_targeted = targets is None or bus.origin.origin in targets
    # Validate the whole target set against the census BEFORE any local side effect,
    # so a typo'd worker name is a loud error and never a silent narrowing. A name
    # absent from the census is a caller mistake, surfaced as a 400 (never a bare 500).
    if targets is not None:
        try:
            await bus.validate_targets(targets)
        except UnknownFleetTargetsError as exc:
            raise BadRequestError(str(exc)) from exc

    local: LocalApplyResult | None = None
    local_failure: Exception | None = None
    if self_targeted:
        try:
            result = await apply()
        except Exception as exc:
            if not publish_on_local_failure:
                # Abort-before-publish: nothing is broadcast, siblings untouched.
                raise
            local_failure = exc
            local = LocalApplyResult(outcome=OpOutcome.failed, error=f"{type(exc).__name__}: {exc}")
        else:
            local = LocalApplyResult(outcome=OpOutcome.applied, payload=result)

    report = await bus.publish(op, targets, local)
    # The report is embedded in the response, but an unconfirmed origin is also a
    # loud, visible failure — never a silently stale sibling.
    log_non_convergence(report)
    if local_failure is not None:
        raise FleetBroadcastError(report.op, report, local_failure) from local_failure
    return report.model_dump(mode="json")
