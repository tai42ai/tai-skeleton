"""The hooks firing path evaluates its authored jq off the event loop via the
shared ``run_jq_first`` helper. Both the condition and the
expr->tool-input mapping route through it, and a fire-time jq error still
propagates loudly — no site swallows it."""

from __future__ import annotations

import pytest
from tai_contract.hooks import HookParams

import tai_skeleton.hooks.managers.base_hooks_manager as bhm
from tai_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai_skeleton.hooks.settings import HooksSettings


async def test_condition_and_expr_evaluate_through_off_loop_helper(make_app, monkeypatch):
    calls: list[tuple[str, dict]] = []
    real = bhm.run_jq_first

    async def _spy(expression, payload):
        calls.append((expression, payload))
        return await real(expression, payload)

    monkeypatch.setattr(bhm, "run_jq_first", _spy)

    app = make_app()
    manager = InMemoryHooksManager(HooksSettings())
    await manager.register(
        HookParams(name="c", topic="t", tool="noop", condition='.status == "paid"', expr="{id: .id}")
    )

    await manager.on_event("t", {"id": 3, "status": "paid"})

    # Same result the inline eval produced, now via the off-loop helper.
    assert app.tools.runs == [("noop", {"id": 3})]
    assert ('.status == "paid"', {"id": 3, "status": "paid"}) in calls
    assert ("{id: .id}", {"id": 3, "status": "paid"}) in calls


async def test_fire_time_jq_error_is_not_swallowed(make_app):
    app = make_app()  # noqa: F841 - registers the app context the manager reads
    manager = InMemoryHooksManager(HooksSettings())
    await manager.register(
        # Compiles at register time, raises at evaluation (string -> number).
        HookParams(name="bad", topic="t", tool="noop", condition=".x | tonumber")
    )

    with pytest.raises(ValueError, match="cannot be parsed as a number"):
        await manager.on_event("t", {"x": "abc"})
