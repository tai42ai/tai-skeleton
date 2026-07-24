"""The ACTING-PRINCIPAL resolution the operations authority rules key on.

``resolve_caller`` names the principal the dispatch actually runs as: the bound execution
identity inside a fire, the request-scoped id outside one. Both routes run the SAME
classification, so the ownership gate never keys on one principal while the dispatch runs
as another.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz.execution_identity import reset_execution_identity, set_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.operations import _authority as authority
from tai42_skeleton.operations.errors import NotFoundError, OperationFailed


def _gate_on(monkeypatch: pytest.MonkeyPatch, *, caller_id: str | None, policies: dict[str, AccessPolicy]) -> None:
    """Turn the gate ON with ``caller_id`` as the request-scope caller and ``policies``
    answering the enforcer's ``get_policy`` (an absent id resolves to the scope-less
    policy an unknown key has)."""
    monkeypatch.setattr(authority, "access_control_settings", lambda: AccessControlSettings(enable=True))
    monkeypatch.setattr(authority, "get_current_user_id", lambda: caller_id)

    class _Enforcer:
        def __init__(self, _settings) -> None:
            pass

        async def get_policy(self, user_id: str) -> AccessPolicy:
            return policies.get(user_id, AccessPolicy(scopes=[]))

    monkeypatch.setattr(authority, "PolicyEnforcer", _Enforcer)


@asynccontextmanager
async def _fire_as(execution_key: str) -> AsyncIterator[None]:
    """Run the body under ``execution_key`` as the bound execution identity, released in
    a ``finally`` so the binding cannot outlive the block."""
    token = set_execution_identity(CallerIdentity(user_id=execution_key, claims={}))
    try:
        yield
    finally:
        reset_execution_identity(token)


async def test_execution_identity_is_the_acting_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fire runs as its execution key even while an unrelated request-scope principal is
    # still bound — and that principal being an ADMIN must not leak its authority in.
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": AccessPolicy(scopes=["*"]), "k-exec": AccessPolicy(scopes=["hooks"])},
    )
    async with _fire_as("k-exec"):
        caller = await authority.resolve_caller()
    assert caller.caller_id == "k-exec"
    assert caller.is_admin is False
    assert caller.policy.scopes == ["hooks"]


async def test_execution_identity_admin_classification_reads_its_own_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ownerless condition-free ``"*"`` execution key IS the admin discriminator, read
    # from the key's own stored policy — the same rule the request path applies.
    _gate_on(monkeypatch, caller_id=None, policies={"k-root": AccessPolicy(scopes=["*"])})
    async with _fire_as("k-root"):
        caller = await authority.resolve_caller()
    assert caller.is_admin is True
    assert caller.owner_claim is None


async def test_execution_identity_owned_key_is_never_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    # An owned key reads admin from its raw scopes, but the owner-claim conjunct denies
    # it the admin path on the fire path as on the request path.
    _gate_on(
        monkeypatch,
        caller_id=None,
        policies={"k-exec": AccessPolicy(scopes=["*"], policy_data={OWNER_USER_ID_CLAIM: "alice"})},
    )
    async with _fire_as("k-exec"):
        caller = await authority.resolve_caller()
    assert caller.is_admin is False
    assert caller.owner_claim == "alice"


async def test_execution_identity_condition_bearing_key_is_never_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    # A role-holder carries ``["*"]`` plus a jq condition; a scopes-only test would hand
    # every editor/viewer the admin path on a fire.
    _gate_on(monkeypatch, caller_id=None, policies={"k-editor": AccessPolicy(scopes=["*"], condition=".sub != null")})
    async with _fire_as("k-editor"):
        assert (await authority.resolve_caller()).is_admin is False


async def test_request_path_resolves_the_request_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    # No execution identity bound: the acting principal is the request-scope caller.
    _gate_on(monkeypatch, caller_id="alice", policies={"alice": AccessPolicy(scopes=["hooks"])})
    caller = await authority.resolve_caller()
    assert caller.caller_id == "alice"
    assert caller.is_admin is False


async def test_gate_off_is_admin_with_an_execution_identity_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # With access control OFF there is no principal to attenuate against and no policy
    # store to read, on the fire path as on the request path.
    monkeypatch.setattr(authority, "access_control_settings", lambda: AccessControlSettings(enable=False))
    async with _fire_as("k-exec"):
        caller = await authority.resolve_caller()
    assert caller.caller_id is None
    assert caller.is_admin is True


async def test_no_principal_at_all_raises_a_typed_loud_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Gate ON with neither an execution identity nor a caller: a LOUD typed 500, never a
    # silent admin path. The detail stays in the server log — the typed error's text is
    # echoed to the caller.
    _gate_on(monkeypatch, caller_id=None, policies={})
    with caplog.at_level("ERROR"), pytest.raises(OperationFailed) as exc_info:
        await authority.resolve_caller()
    assert exc_info.value.status == 500
    assert str(exc_info.value.message) == "access_control: internal authority-resolution failure"
    assert "no acting principal is bound" in caplog.text


# -- the shared ownership predicate ------------------------------------------


async def test_require_owned_by_caller_unknown_key_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # The api-key management surfaces owe their callers the 404/403 split, so an absent
    # key is distinguishable there — unlike at the execution-key bind door.
    async def _no_body(_user_id: str) -> dict | None:
        return None

    monkeypatch.setattr(management, "get_policy_body", _no_body)
    caller = authority.Caller(caller_id="alice", policy=AccessPolicy(scopes=[]), is_admin=False, owner_claim=None)
    with pytest.raises(NotFoundError, match="user not found"):
        await authority.require_owned_by_caller(caller, "ghost")


def test_owner_of_reads_the_one_ownership_home() -> None:
    # Absent, empty and ownerless policy_data all answer None, so no call site respells
    # the fallback.
    assert authority.owner_of({OWNER_USER_ID_CLAIM: "alice"}) == "alice"
    assert authority.owner_of({}) is None
    assert authority.owner_of(None) is None
