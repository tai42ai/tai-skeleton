"""A recording stand-in for :class:`~tai_skeleton.app.bus.WorkerBus`.

Op-level oracle tests drive the operations directly with a faked ``tai_app`` impl.
:class:`FakeBus` is the bus half of that fake: it records every ``publish`` /
``validate_targets`` call and returns a :class:`FleetResult` built from the caller's
own ``local`` self entry plus one ``applied`` entry per configured remote origin —
so a publisher's per-origin response can be asserted without a real Redis.
``validate_targets`` reproduces the census-membership raise so the
validate-before-apply order is exercised.
"""

from __future__ import annotations

from typing import Any

from tai_skeleton.app.bus import (
    FleetOrigin,
    FleetResult,
    LocalApplyResult,
    OpOutcome,
    OriginKind,
    OriginResult,
    UnknownFleetTargetsError,
)


class FakeBus:
    def __init__(self, *, origin: str = "serve-test", remotes: list[str] | None = None) -> None:
        self._origin = FleetOrigin(origin=origin, kind=OriginKind.serve, pid=1)
        self._remotes = [FleetOrigin(origin=o, kind=OriginKind.serve, pid=2) for o in (remotes or [])]
        self.publish_calls: list[tuple[dict[str, Any], list[str] | None, LocalApplyResult | None]] = []
        self.validate_calls: list[list[str] | None] = []

    @property
    def origin(self) -> FleetOrigin:
        return self._origin

    async def census(self) -> list[FleetOrigin]:
        return [self._origin, *self._remotes]

    async def validate_targets(self, targets: list[str] | None) -> None:
        self.validate_calls.append(targets)
        if targets is None:
            return
        known = {self._origin.origin, *(o.origin for o in self._remotes)}
        unknown = sorted(set(targets) - known)
        if unknown:
            raise UnknownFleetTargetsError(f"worker bus: unknown fleet targets (not on the census): {unknown}")

    async def publish(
        self,
        op: dict[str, Any],
        targets: list[str] | None,
        local: LocalApplyResult | None,
    ) -> FleetResult:
        self.publish_calls.append((op, targets, local))
        results: list[OriginResult] = []
        if local is not None:
            results.append(
                OriginResult(
                    origin=self._origin.origin, outcome=local.outcome, payload=local.payload, error=local.error
                )
            )
        reached = (
            [o.origin for o in self._remotes] if targets is None else [t for t in targets if t != self._origin.origin]
        )
        for origin in reached:
            results.append(OriginResult(origin=origin, outcome=OpOutcome.applied))
        return FleetResult(op=op["op"], results=results)
