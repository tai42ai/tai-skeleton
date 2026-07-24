"""The conversation recovery lifecycle wiring: the ``on_startup`` hook that re-drives every
unfinished record and starts the sweep, and the ``on_shutdown`` hook that stops it.

The recovery passes themselves are tested in ``test_turn`` and ``test_delivery_sweep``;
here only the wiring is asserted — that both hooks run their passes, in the order the
durability story depends on, and no-op without a backend.
"""

from __future__ import annotations

import pytest

import tai42_skeleton.conversations as conversations_package


def _router():
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import app as skeleton_app

    with tai42_app.bound(skeleton_app):
        from tai42_skeleton.routers import conversations as router

    return router


@pytest.fixture
def order(monkeypatch) -> list[str]:
    """Record the recovery passes as they fire, so their presence AND order are asserted."""
    ran: list[str] = []

    async def _redrive_accepted() -> None:
        ran.append("redrive_accepted")

    async def _redrive_pending() -> None:
        ran.append("redrive_pending")

    def _start_delivery_sweep() -> None:
        ran.append("start_delivery_sweep")

    async def _stop_delivery_sweep() -> None:
        ran.append("stop_delivery_sweep")

    monkeypatch.setattr(conversations_package, "redrive_accepted", _redrive_accepted)
    monkeypatch.setattr(conversations_package, "redrive_pending", _redrive_pending)
    monkeypatch.setattr(conversations_package, "start_delivery_sweep", _start_delivery_sweep)
    monkeypatch.setattr(conversations_package, "stop_delivery_sweep", _stop_delivery_sweep)
    return ran


async def test_startup_redrives_intake_then_delivery_then_starts_the_sweep(monkeypatch, order):
    # Intake re-drive FIRST: it gives every stranded ``accepted`` record a terminal outcome
    # the delivery re-drive then picks up. The sweep is what recovers a live-lease death.
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")
    await _router()._redrive_pending_conversations()
    assert order == ["redrive_accepted", "redrive_pending", "start_delivery_sweep"]


async def test_startup_is_a_no_op_without_a_backend(monkeypatch, order):
    monkeypatch.delenv("CONVERSATIONS_REDIS_URL", raising=False)
    await _router()._redrive_pending_conversations()
    assert order == []


async def test_shutdown_stops_the_sweep_with_a_backend(monkeypatch, order):
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")
    await _router()._stop_conversations_delivery_sweep()
    assert order == ["stop_delivery_sweep"]


async def test_shutdown_is_a_no_op_without_a_backend(monkeypatch, order):
    monkeypatch.delenv("CONVERSATIONS_REDIS_URL", raising=False)
    await _router()._stop_conversations_delivery_sweep()
    assert order == []
