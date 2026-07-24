"""The synthetic identity a background fire runs AS, and the contextvar it is bound into.

Everything the identity carries is read live from the policy store at the moment of the
fire — never from a token — so disabling or deleting the key or its owner is refused by the
next build. The binding lives in a contextvar distinct from the three request-scope identity
vars, so a background execution is never mistaken for an authenticated caller.
"""

from __future__ import annotations

import asyncio

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import get_current_user_id

from tai42_skeleton.access_control.request_scopes import get_request_effective_scopes, get_request_identity_claims
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz.execution import bind_execution_identity, build_execution_identity
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity, resolve_caller_identity
from tai42_skeleton.operations.errors import PermissionDenied


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Access control ENABLED for this module, over whatever store the test seeds — the
    identity build only reads a policy store when the gate is on."""
    from tai42_skeleton.authz import execution as execution_module

    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))


# -- what the identity carries ------------------------------------------------


def test_an_unowned_key_carries_no_owner_claim(ac_env, bound_app) -> None:
    # The owner claim is OMITTED, never ``None``, so the owner second-pass is skipped.
    ac_env.add_policy("k-fire", scopes=["hooks", "tools"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    identity = asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))

    assert identity.user_id == "k-fire"
    assert identity.claims == {}
    assert OWNER_USER_ID_CLAIM not in (identity.claims or {})
    # A fire is never the platform-internal principal.
    assert identity.is_internal is False


def test_an_owned_key_carries_its_owner_and_no_decided_scope_set(ac_env, bound_app) -> None:
    # NO scope set rides with the identity: the owner attenuation is re-derived from both
    # live policies at every dispatch, so a snapshot here would be stale on a mid-fire de-scope.
    ac_env.add_policy(
        "k-fire",
        scopes=["hooks", "tools"],
        policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-fire"},
    )
    ac_env.add_policy("alice", scopes=["hooks"])

    identity = asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))

    assert identity.claims == {OWNER_USER_ID_CLAIM: "alice"}
    assert identity.effective_scopes is None


def test_the_owner_is_read_from_the_stored_policy_not_from_a_token(ac_env, bound_app) -> None:
    # A fire presents no token, so the owner reference can only come from stored policy_data.
    ac_env.add_policy(
        "k-fire",
        scopes=["hooks"],
        policy_data={OWNER_USER_ID_CLAIM: "alice", "tier": "gold", KEY_FINGERPRINT_CLAIM: "fp-k-fire"},
    )
    ac_env.add_policy("alice", scopes=["hooks"])

    identity = asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))

    # ONLY the owner reference is carried; no other stored field becomes an identity claim.
    assert identity.claims == {OWNER_USER_ID_CLAIM: "alice"}


def test_access_control_disabled_yields_the_bare_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai42_skeleton.authz import execution as execution_module

    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=False))
    identity = asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))
    # No policy store to read: the identity names the key and every decision against it allows.
    assert identity == CallerIdentity(user_id="k-fire")


# -- the key that cannot carry authority at all -------------------------------


def test_an_unknown_key_is_refused(ac_env, bound_app) -> None:
    # Refusing here, rather than returning an authority-less identity, is what stops a
    # capability-tool fire (never scope-checked) under a key that no longer exists.
    with pytest.raises(PermissionDenied, match="has no policy"):
        asyncio.run(build_execution_identity("ghost", bound_fingerprint="fp-ghost"))


def test_a_disabled_key_is_refused(ac_env, bound_app) -> None:
    ac_env.add_policy("k-fire", scopes=["hooks"], policy_data={"disabled": True, KEY_FINGERPRINT_CLAIM: "fp-k-fire"})
    with pytest.raises(PermissionDenied, match="is disabled"):
        asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))


def test_a_key_whose_owner_is_disabled_is_refused(ac_env, bound_app) -> None:
    ac_env.add_policy(
        "k-fire", scopes=["hooks"], policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-fire"}
    )
    ac_env.add_policy("alice", scopes=["hooks"], policy_data={"disabled": True})
    with pytest.raises(PermissionDenied, match="owner 'alice' of execution key 'k-fire' is disabled"):
        asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))


def test_a_key_whose_owner_has_no_policy_is_refused(ac_env, bound_app) -> None:
    ac_env.add_policy(
        "k-fire", scopes=["hooks"], policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-fire"}
    )
    with pytest.raises(PermissionDenied, match="owner 'alice' of execution key 'k-fire' has no policy"):
        asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))


# -- every authority reduction lands on the NEXT build ------------------------


def test_deleting_the_key_denies_the_next_fire(ac_env, bound_app) -> None:
    ac_env.add_policy("k-fire", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})
    assert asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire")).user_id == "k-fire"

    ac_env.policies = [p for p in ac_env.policies if p["user_id"] != "k-fire"]

    with pytest.raises(PermissionDenied, match="has no policy"):
        asyncio.run(build_execution_identity("k-fire", bound_fingerprint="fp-k-fire"))


# -- the immutable per-mint fingerprint: a revoke+remint of the same user_id ---


def test_a_revoke_remint_of_the_same_user_id_denies_the_old_binding(ac_env, bound_app) -> None:
    # A remint of the same user_id writes a fresh fingerprint (here admin-shaped), so the
    # record's bound F1 no longer matches and the fire is denied fail-closed.
    ac_env.add_policy("svc", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "F1"})
    assert asyncio.run(build_execution_identity("svc", bound_fingerprint="F1")).user_id == "svc"

    # Revoke + remint the same user_id: a brand-new fingerprint, admin-shaped.
    ac_env.policies = [p for p in ac_env.policies if p["user_id"] != "svc"]
    ac_env.add_policy("svc", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "F2"})

    with pytest.raises(PermissionDenied, match="execution key 'svc' no longer matches the bound key identity"):
        asyncio.run(build_execution_identity("svc", bound_fingerprint="F1"))


def test_a_live_key_carrying_the_bound_fingerprint_still_fires(ac_env, bound_app) -> None:
    # Non-vacuity: the deny above is the mismatch, not the check refusing every key.
    ac_env.add_policy("svc", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "F1"})
    assert asyncio.run(build_execution_identity("svc", bound_fingerprint="F1")).user_id == "svc"


def test_a_live_key_with_no_fingerprint_is_refused(ac_env, bound_app) -> None:
    # Absent is not a wildcard: no stored fingerprint fails the same equality.
    ac_env.add_policy("svc", scopes=["hooks"])
    with pytest.raises(PermissionDenied, match="execution key 'svc' no longer matches the bound key identity"):
        asyncio.run(build_execution_identity("svc", bound_fingerprint="F1"))


# -- the binding: set + finally-reset, per task --------------------------------


def test_bind_sets_for_the_body_and_releases_afterwards(ac_env, bound_app) -> None:
    ac_env.add_policy("k-fire", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    async def run() -> None:
        assert get_execution_identity() is None
        async with bind_execution_identity("k-fire", bound_fingerprint="fp-k-fire") as identity:
            assert get_execution_identity() is identity
            assert identity.user_id == "k-fire"
        assert get_execution_identity() is None

    asyncio.run(run())


def test_bind_releases_even_when_the_body_raises(ac_env, bound_app) -> None:
    ac_env.add_policy("k-fire", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    async def run() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with bind_execution_identity("k-fire", bound_fingerprint="fp-k-fire"):
                raise RuntimeError("boom")
        # The binding cannot outlive the dispatch it was opened for.
        assert get_execution_identity() is None

    asyncio.run(run())


def test_a_refused_key_never_enters_the_body_and_binds_nothing(ac_env, bound_app) -> None:
    async def run() -> None:
        entered = False
        with pytest.raises(PermissionDenied, match="execution key 'ghost' has no policy"):
            async with bind_execution_identity("ghost", bound_fingerprint="fp-ghost"):
                entered = True  # pragma: no cover - the body must not run
        assert entered is False
        assert get_execution_identity() is None

    asyncio.run(run())


def test_concurrent_tasks_each_carry_their_own_key(ac_env, bound_app) -> None:
    # A contextvar set inside a task is invisible to siblings and to the gathering parent,
    # which is what lets a fanned-out event fire each hook under its own key.
    for key in ("k-a", "k-b", "k-c"):
        ac_env.add_policy(key, scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: f"fp-{key}"})

    async def run() -> list[str]:
        async def fire(key: str) -> str:
            async with bind_execution_identity(key, bound_fingerprint=f"fp-{key}"):
                await asyncio.sleep(0)  # yield so the tasks interleave
                bound = get_execution_identity()
                assert bound is not None
                return bound.user_id or ""

        seen = await asyncio.gather(fire("k-a"), fire("k-b"), fire("k-c"))
        # Nothing leaked out of the fan-out.
        assert get_execution_identity() is None
        return list(seen)

    assert asyncio.run(run()) == ["k-a", "k-b", "k-c"]


def test_an_inner_binding_shadows_an_outer_one_for_its_task_only(ac_env, bound_app) -> None:
    # A trigger link binds its key around the fan-out while each hook re-binds its own.
    ac_env.add_policy("k-link", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-link"})
    ac_env.add_policy("k-hook", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-hook"})

    async def run() -> None:
        async with bind_execution_identity("k-link", bound_fingerprint="fp-k-link"):

            async def inner() -> str:
                async with bind_execution_identity("k-hook", bound_fingerprint="fp-k-hook"):
                    bound = get_execution_identity()
                    assert bound is not None
                    return bound.user_id or ""

            assert await asyncio.gather(inner()) == ["k-hook"]
            outer = get_execution_identity()
            assert outer is not None
            assert outer.user_id == "k-link"

    asyncio.run(run())


# -- distinctness from the request-scope identity ------------------------------


def test_the_execution_identity_is_not_a_request_caller(ac_env, bound_app) -> None:
    # ``resolve_caller_identity`` reads only the three request-scope vars, never this one.
    ac_env.add_policy(
        "k-fire", scopes=["hooks"], policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-fire"}
    )
    ac_env.add_policy("alice", scopes=["hooks"])

    async def run() -> None:
        async with bind_execution_identity("k-fire", bound_fingerprint="fp-k-fire"):
            assert get_execution_identity() is not None
            assert get_current_user_id() is None
            assert get_request_effective_scopes() is None
            assert get_request_identity_claims() is None
            assert resolve_caller_identity() == CallerIdentity(user_id=None, effective_scopes=None, claims=None)

    asyncio.run(run())
