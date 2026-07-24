"""A hook fires AS its bound execution key, or it does not fire.

* A record naming no execution key is refused before any work — never run unbounded.
* Each hook of a fan-out binds its OWN key inside its own task, never a sibling's.
* The binding is released when the fire ends, including on a refusal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from pydantic import ValidationError
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM
from tai42_contract.hooks import HookParams

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz import execution as execution_module
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.operations.errors import PermissionDenied
from tests.access_control.conftest import FakeAccessControlPg, make_pg_ctx
from tests.access_control.conftest import FakeRedis as AcFakeRedis
from tests.access_control.conftest import make_client_ctx as make_ac_client_ctx


def _manager() -> InMemoryHooksManager:
    return InMemoryHooksManager(HooksSettings())


def _record_bound_keys(app) -> list[tuple[str, str | None]]:
    """Wrap the fake tool runner so each dispatch records ``(tool, the execution key
    bound at that moment)`` — what a fire actually runs as, observed from inside it."""
    seen: list[tuple[str, str | None]] = []
    original = app.tools.run_tool

    async def recording_run_tool(name, tool_input):
        identity = get_execution_identity()
        seen.append((name, identity.user_id if identity is not None else None))
        await asyncio.sleep(0)  # yield so co-scheduled hooks interleave
        return await original(name, tool_input)

    app.tools.run_tool = recording_run_tool
    return seen


# -- a record with no bound key ------------------------------------------------


@pytest.mark.parametrize("execution_key", [None, ""])
def test_a_hook_record_cannot_be_built_without_an_execution_key(execution_key) -> None:
    # The contract refuses it at the model: a hook that names no identity (or a blank
    # one) cannot be stored at all, so no door can create the keyless record.
    fields: dict[str, Any] = {"name": "h", "topic": "t", "tool": "noop"}
    if execution_key is not None:
        fields["execution_key"] = execution_key
    with pytest.raises(ValidationError, match="execution_key"):
        HookParams(**fields)


async def test_a_keyless_record_that_reaches_the_fire_is_refused_and_runs_nothing(make_app) -> None:
    # A record that reached the manager without a key is refused BEFORE any work; there
    # is no fallback to the server's own authority.
    app = make_app()
    keyless = HookParams.model_construct(
        name="h", topic="t", tool="noop", execution_key="", tool_kwargs={}, expr_kwargs={}, condition_kwargs={}
    )
    with pytest.raises(PermissionDenied, match="binds no execution key"):
        await InMemoryHooksManager._run_hook(keyless, {})
    assert app.tools.runs == []


async def test_a_keyless_record_in_a_fan_out_fails_only_its_own_hook(make_app, caplog) -> None:
    # The refusal surfaces on the existing per-hook error-outcome path: the rest of the
    # topic still fires, and the failure is logged rather than swallowed.
    app = make_app()
    manager = _manager()
    await manager.register(
        HookParams(name="good", topic="t", tool="ok", execution_key="k-fire", execution_key_fingerprint="fp-k-fire")
    )
    # Seeded into the topic bucket directly: the model forbids a keyless record, so the
    # only way this state exists is a store that already holds one.
    keyless = HookParams.model_construct(
        name="bad", topic="t", tool="boom", execution_key="", tool_kwargs={}, expr_kwargs={}, condition_kwargs={}
    )
    manager._hooks[manager.settings.get_hook_key("t")]["bad"] = keyless

    with caplog.at_level(logging.ERROR):
        await manager.on_event("t", {})

    assert [name for name, _ in app.tools.runs] == ["ok"]
    assert any("bad" in record.getMessage() for record in caplog.records if record.levelno == logging.ERROR)


# -- one key per hook, per task ------------------------------------------------


async def test_each_fired_hook_runs_under_its_own_key(make_app) -> None:
    app = make_app()
    manager = _manager()
    await manager.register(
        HookParams(name="a", topic="t", tool="tool_a", execution_key="k-a", execution_key_fingerprint="fp-k-a")
    )
    await manager.register(
        HookParams(name="b", topic="t", tool="tool_b", execution_key="k-b", execution_key_fingerprint="fp-k-b")
    )
    await manager.register(
        HookParams(name="c", topic="t", tool="tool_c", execution_key="k-c", execution_key_fingerprint="fp-k-c")
    )
    seen = _record_bound_keys(app)

    await manager.on_event("t", {})

    # Each hook ran under exactly the key its own record names: a contextvar set inside
    # a task is invisible to its siblings, so no key bleeds across.
    assert dict(seen) == {"tool_a": "k-a", "tool_b": "k-b", "tool_c": "k-c"}


async def test_a_hooks_key_is_bound_only_for_its_own_dispatch(make_app) -> None:
    app = make_app()
    manager = _manager()
    await manager.register(
        HookParams(name="a", topic="t", tool="tool_a", execution_key="k-a", execution_key_fingerprint="fp-k-a")
    )
    seen = _record_bound_keys(app)

    assert get_execution_identity() is None
    await manager.on_event("t", {})
    assert seen == [("tool_a", "k-a")]
    # Released with the fire: the fan-out leaves nothing bound behind it.
    assert get_execution_identity() is None


async def test_the_binding_is_released_after_a_failing_fire(make_app) -> None:
    app = make_app(raise_tools={"boom"})
    manager = _manager()
    await manager.register(
        HookParams(name="bad", topic="t", tool="boom", execution_key="k-a", execution_key_fingerprint="fp-k-a")
    )

    await manager.on_event("t", {})

    assert [name for name, _ in app.tools.runs] == ["boom"]
    assert get_execution_identity() is None


async def test_an_outer_binding_does_not_reach_the_hooks(make_app) -> None:
    # The door binds the LINK's key around the fan-out, but each hook re-binds its own
    # inside its task — a hook is never fired under the door's key.
    from tai42_skeleton.authz.execution import bind_execution_identity

    app = make_app()
    manager = _manager()
    await manager.register(
        HookParams(name="a", topic="t", tool="tool_a", execution_key="k-hook", execution_key_fingerprint="fp-k-hook")
    )
    seen = _record_bound_keys(app)

    async with bind_execution_identity("k-link", bound_fingerprint="fp-link"):
        await manager.on_event("t", {})
        # The door's own binding is intact once the fan-out returns.
        outer = get_execution_identity()
        assert outer is not None
        assert outer.user_id == "k-link"

    assert seen == [("tool_a", "k-hook")]


# -- a key that can no longer carry authority ---------------------------------


async def test_a_fire_whose_key_no_longer_exists_is_refused_and_isolated(
    monkeypatch: pytest.MonkeyPatch, make_app, caplog
) -> None:
    # The identity is built from the key's live grants, so deleting a key denies its
    # hook's next fire with no cascade, while the sibling with a live key still runs.
    app = make_app()
    pg = FakeAccessControlPg()
    pg.add_policy("k-live", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-live"})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_ac_client_ctx(AcFakeRedis()))
    monkeypatch.setattr(verifier_module, "client_ctx", make_ac_client_ctx(AcFakeRedis()))
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))

    manager = _manager()
    await manager.register(
        HookParams(
            name="live", topic="t", tool="tool_live", execution_key="k-live", execution_key_fingerprint="fp-k-live"
        )
    )
    await manager.register(
        HookParams(
            name="gone",
            topic="t",
            tool="tool_gone",
            execution_key="k-deleted",
            execution_key_fingerprint="fp-k-deleted",
        )
    )

    with caplog.at_level(logging.ERROR):
        await manager.on_event("t", {})

    assert [name for name, _ in app.tools.runs] == ["tool_live"]
    assert any("gone" in record.getMessage() for record in caplog.records if record.levelno == logging.ERROR)
    assert get_execution_identity() is None


async def test_a_revoke_remint_of_the_execution_key_is_denied_not_run_as_admin(
    monkeypatch: pytest.MonkeyPatch, make_app, caplog
) -> None:
    # ``svc`` (fingerprint F1) is revoked and the SAME user_id reminted admin-shaped
    # with a fresh F2: the record's F1 no longer matches, so the fire is DENIED at the
    # bind and the reminted key never inherits the old record's fire.
    app = make_app()
    pg = FakeAccessControlPg()
    pg.add_policy("svc", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "F1"})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_ac_client_ctx(AcFakeRedis()))
    monkeypatch.setattr(verifier_module, "client_ctx", make_ac_client_ctx(AcFakeRedis()))
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))

    manager = _manager()
    await manager.register(
        HookParams(name="h", topic="t", tool="tool_svc", execution_key="svc", execution_key_fingerprint="F1")
    )
    # A normal fire under the unrevoked key still runs — the fingerprint matches.
    await manager.on_event("t", {})
    assert [name for name, _ in app.tools.runs] == ["tool_svc"]

    # Revoke + remint the same user_id as the admin-shaped key with a fresh fingerprint.
    pg.policies = [p for p in pg.policies if p["user_id"] != "svc"]
    pg.add_policy("svc", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "F2"})

    with caplog.at_level(logging.ERROR):
        await manager.on_event("t", {})

    # The tool did NOT run again: the fire was denied, not executed as the reminted admin.
    assert [name for name, _ in app.tools.runs] == ["tool_svc"]
    assert any(
        "svc" in record.getMessage() and "no longer matches the bound key identity" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    )
    assert get_execution_identity() is None
