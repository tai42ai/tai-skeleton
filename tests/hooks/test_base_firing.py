"""The hook firing path in ``BaseHooksManager``: condition evaluation (rendered
via the bound template manager), expr -> tool-input mapping + tool dispatch,
worker-limited fan-out, per-hook failure isolation, and jq validation at
registration. Plus the abstract-method contract that blocks constructing the base
directly.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from tai42_contract.hooks import HookParams

from tai42_skeleton.hooks.managers.base_hooks_manager import BaseHooksManager
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.settings import HooksSettings


def _settings(**kw) -> HooksSettings:
    return HooksSettings(**kw)


async def test_on_event_fires_tool_with_merged_expr_and_kwargs(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="ship",
            topic="orders",
            tool="ship_tool",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            condition='.status == "paid"',
            expr="{id: .id}",
            tool_kwargs={"extra": 1},
        )
    )

    await manager.on_event("orders", {"id": 7, "status": "paid"})

    assert app.tools.runs == [("ship_tool", {"id": 7, "extra": 1})]


async def test_on_event_without_expr_uses_tool_kwargs_only(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="noexpr",
            topic="t",
            tool="noop",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            tool_kwargs={"k": "v"},
        )
    )

    await manager.on_event("t", {"anything": True})

    assert app.tools.runs == [("noop", {"k": "v"})]


async def test_on_event_skips_hook_when_condition_false(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="cond",
            topic="t",
            tool="noop",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            condition='.status == "paid"',
        )
    )

    await manager.on_event("t", {"status": "unpaid"})

    assert app.tools.runs == []


async def test_on_event_raises_when_condition_errors_at_runtime(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    # Compiles at register time, raises at evaluation (string -> number).
    await manager.register(
        HookParams(
            name="bad-runtime",
            topic="t",
            tool="noop",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            condition=".x | tonumber",
        )
    )

    # A fire-time jq evaluation error must surface loudly, not silently skip the
    # hook (a skip is indistinguishable from a cleanly-false condition).
    with pytest.raises(ValueError, match="cannot be parsed as a number"):
        await manager.on_event("t", {"x": "not-a-number"})

    assert app.tools.runs == []


async def test_on_event_no_hooks_for_topic_is_noop(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.on_event("unknown-topic", {"a": 1})
    assert app.tools.runs == []


def _track_peak_concurrency(app):
    """Wrap the fake tool so it records in-flight concurrency: it yields once
    (so concurrently-scheduled hooks overlap) and tracks the peak number running
    at the same time. Returns the mutable ``{"peak": ...}`` record."""
    state = {"in_flight": 0, "peak": 0}
    original_run_tool = app.tools.run_tool

    async def tracking_run_tool(name, tool_input):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(0)  # yield so co-scheduled hooks can overlap
            return await original_run_tool(name, tool_input)
        finally:
            state["in_flight"] -= 1

    app.tools.run_tool = tracking_run_tool
    return state


async def test_on_event_bounds_fanout_at_max_workers(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings(max_workers=2))
    for name in ("a", "b", "c", "d"):
        await manager.register(
            HookParams(
                name=name, topic="t", tool=f"tool_{name}", execution_key="k-fire", execution_key_fingerprint="fp-fire"
            )
        )

    state = _track_peak_concurrency(app)

    await manager.on_event("t", {})

    assert {name for name, _ in app.tools.runs} == {"tool_a", "tool_b", "tool_c", "tool_d"}
    # The manager-wide semaphore held concurrency to max_workers; the peak never
    # exceeded 2.
    assert state["peak"] <= 2


async def test_semaphore_bounds_total_across_concurrent_events(make_app):
    # ONE semaphore per manager, created at construction, bounds the TOTAL
    # in-flight hook executions across ALL events: two topics firing at once
    # share the max_workers=2 bound instead of each fanning out its own 2.
    app = make_app()
    manager = InMemoryHooksManager(_settings(max_workers=2))
    for name in ("a", "b"):
        await manager.register(
            HookParams(
                name=f"t1-{name}",
                topic="t1",
                tool=f"tool_t1_{name}",
                execution_key="k-fire",
                execution_key_fingerprint="fp-fire",
            )
        )
        await manager.register(
            HookParams(
                name=f"t2-{name}",
                topic="t2",
                tool=f"tool_t2_{name}",
                execution_key="k-fire",
                execution_key_fingerprint="fp-fire",
            )
        )

    state = _track_peak_concurrency(app)

    await asyncio.gather(manager.on_event("t1", {}), manager.on_event("t2", {}))

    assert {name for name, _ in app.tools.runs} == {"tool_t1_a", "tool_t1_b", "tool_t2_a", "tool_t2_b"}
    # A per-event semaphore would allow a peak of 4 (2 per event); the global
    # bound caps the whole manager at 2.
    assert state["peak"] <= 2


async def test_on_event_isolates_a_failing_hook(make_app, caplog):
    # One hook's tool raises; the other still runs and the gather does not crash.
    app = make_app(raise_tools={"boom"})
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(name="bad", topic="t", tool="boom", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )
    await manager.register(
        HookParams(name="good", topic="t", tool="ok", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )

    with caplog.at_level(logging.ERROR):
        await manager.on_event("t", {})

    ran = {name for name, _ in app.tools.runs}
    assert ran == {"boom", "ok"}
    # Isolation must not be silent: the failing hook's error is logged loudly.
    assert any(rec.levelno == logging.ERROR and "bad" in rec.getMessage() for rec in caplog.records)


async def test_condition_rendered_via_template_id(make_app):
    # condition_id resolves through the template manager to a real jq expression.
    app = make_app(by_id={"cond-tmpl": ".ok == true"})
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="byid",
            topic="t",
            tool="noop",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            condition_id="cond-tmpl",
        )
    )

    await manager.on_event("t", {"ok": True})
    assert app.tools.runs == [("noop", {})]

    app.tools.runs.clear()
    await manager.on_event("t", {"ok": False})
    assert app.tools.runs == []


def test_validate_jq_accepts_valid_expr_and_condition():
    BaseHooksManager.validate_jq_fields(
        HookParams(
            name="ok",
            topic="t",
            tool="noop",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            condition=".a",
            expr=".b",
        )
    )
    # Nothing to validate when both inline fields are absent.
    BaseHooksManager.validate_jq_fields(
        HookParams(name="ok2", topic="t", tool="noop", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )


def test_validate_jq_rejects_bad_expr():
    bad = HookParams(
        name="bad",
        topic="t",
        tool="noop",
        execution_key="k-fire",
        execution_key_fingerprint="fp-fire",
        expr="this is ( not jq",
    )
    with pytest.raises(ValueError, match="expr is not valid jq"):
        BaseHooksManager.validate_jq_fields(bad)


# -- tool_kwargs_override ----------------------------------------------


async def test_override_merges_over_event_input_but_under_hook_kwargs(make_app):
    # Three colliding layers on key ``k``: expr-derived event_input (weakest),
    # the link override (middle), the hook's static tool_kwargs (strongest).
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="h",
            topic="t",
            tool="tool",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            expr="{k: .k, m: .m, e: 1}",
            tool_kwargs={"k": "hook", "h": 2},
        )
    )

    await manager.on_event("t", {"k": "event", "m": "event"}, tool_kwargs_override={"k": "link", "m": "link", "l": 3})

    # ``k``: hook beats link beats event. ``m``: link beats event. ``l``: link only.
    assert app.tools.runs == [("tool", {"k": "hook", "e": 1, "h": 2, "l": 3, "m": "link"})]


async def test_hook_pinned_tool_kwargs_are_not_overridable_by_a_trigger_link(make_app):
    # A hook's static tool_kwargs pin an argument no link can replace or clear; a key
    # the author never pinned is the link's to supply.
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="h",
            topic="t",
            tool="tool",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            tool_kwargs={"recipient": "ops@example.com"},
        )
    )

    await manager.on_event("t", {}, tool_kwargs_override={"recipient": "attacker@example.com", "subject": "hi"})

    assert app.tools.runs == [("tool", {"recipient": "ops@example.com", "subject": "hi"})]


async def test_none_override_is_byte_identical(make_app):
    # No override ⇒ exactly today's merge (regression guard for universal_webhook).
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="h",
            topic="t",
            tool="tool",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            expr="{id: .id}",
            tool_kwargs={"x": 1},
        )
    )

    await manager.on_event("t", {"id": 9}, tool_kwargs_override=None)
    assert app.tools.runs == [("tool", {"id": 9, "x": 1})]

    app.tools.runs.clear()
    await manager.on_event("t", {"id": 9})  # the universal_webhook call site (no arg)
    assert app.tools.runs == [("tool", {"id": 9, "x": 1})]


async def test_override_applies_to_every_hook_on_topic(make_app):
    app = make_app()
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(name="a", topic="t", tool="tool_a", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )
    await manager.register(
        HookParams(name="b", topic="t", tool="tool_b", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )

    await manager.on_event("t", {}, tool_kwargs_override={"shared": True})

    runs = dict(app.tools.runs)
    assert runs == {"tool_a": {"shared": True}, "tool_b": {"shared": True}}


async def test_span_writer_records_override_values(make_app, monkeypatch):
    from tai42_skeleton.hooks.managers import base_hooks_manager as base

    recorded: list[dict] = []

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Writer:
        def start_span(self, **kw):
            return _Span()

        def trace_attributes(self, **kw):
            return _Span()

        def update_current_span(self, **kw):
            if "metadata" in kw:
                recorded.append(kw["metadata"])

    monkeypatch.setattr(base, "get_monitoring", lambda: SimpleNamespace(writer=_Writer()))

    make_app()  # bind the fake tai42_app so the firing path resolves its tool runner
    manager = InMemoryHooksManager(_settings())
    await manager.register(
        HookParams(
            name="h",
            topic="t",
            tool="tool",
            execution_key="k-fire",
            execution_key_fingerprint="fp-fire",
            tool_kwargs={"a": 1},
        )
    )

    await manager.on_event("t", {}, tool_kwargs_override={"o": 2})

    run_span = next(m for m in recorded if m.get("tool") == "tool")
    assert run_span["tool_kwargs"] == {"a": 1}
    assert run_span["tool_kwargs_override"] == {"o": 2}


def test_base_manager_cannot_be_constructed():
    # ``BaseHooksManager`` is an ABC: a backend that forgets one of the eight
    # abstract ops fails at CONSTRUCTION (TypeError), not deferred to call time.
    with pytest.raises(TypeError):
        BaseHooksManager(_settings())  # type: ignore[abstract]


def test_partial_subclass_is_abstract():
    # A subclass implementing only some ops is still abstract and cannot be built.
    class _Partial(BaseHooksManager):
        async def register(self, params):  # type: ignore[override]
            return True

    with pytest.raises(TypeError):
        _Partial(_settings())  # type: ignore[abstract]
