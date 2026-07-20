"""Op-level characterization for the api-keys/scopes operations.

The route oracles (``tests/routers/test_api_keys*``) drive these ops end to end
through the adapter; this pins the operation-level edge branches directly — the
gate-off / gate-on caller resolution, the non-admin ownership rejections, and the
``ValueError -> BadRequestError`` mappings — so each declared error class is
exercised at the operation itself, independent of the route surface.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.access_control.models import AccessPolicy

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.operations import api_keys as ops
from tai42_skeleton.operations.errors import BadRequestError, ForbiddenError, NotFoundError


def _caller(*, caller_id="c", scopes=None, is_admin=False, owner_claim=None) -> ops._Caller:
    return ops._Caller(
        caller_id=caller_id,
        policy=AccessPolicy(scopes=scopes or []),
        is_admin=is_admin,
        owner_claim=owner_claim,
    )


# -- caller resolution -------------------------------------------------------


async def test_resolve_caller_gate_off_is_admin(monkeypatch):
    # With access control OFF there is no principal to attenuate against — the caller is
    # treated as admin with no bound id.
    monkeypatch.setattr(ops, "access_control_settings", lambda: AccessControlSettings(enable=False))
    caller = await ops._resolve_caller()
    assert caller.is_admin is True
    assert caller.caller_id is None
    assert "*" in caller.policy.scopes


async def test_resolve_caller_gate_on_no_caller_raises(monkeypatch):
    # With the gate ON but the caller contextvar UNSET (an invariant breach), refuse
    # loudly rather than silently escalating to admin.
    monkeypatch.setattr(ops, "access_control_settings", lambda: AccessControlSettings(enable=True))
    with pytest.raises(RuntimeError, match="caller user id is unset"):
        await ops._resolve_caller()


def test_check_scope_subset_wildcard_caller_grants_anything():
    # A ``"*"`` caller may grant any scope — the early return, no excess computed.
    ops._check_scope_subset(_caller(scopes=["*"]), ["anything", "at-all"])


# -- ownership rejections ----------------------------------------------------


async def test_require_owned_by_caller_unknown_key_is_not_found(monkeypatch):
    async def _no_body(_user_id):
        return None

    monkeypatch.setattr(management, "get_policy_body", _no_body)
    with pytest.raises(NotFoundError, match="user not found"):
        await ops._require_owned_by_caller(_caller(caller_id="alice"), "ghost")


async def test_edit_api_key_non_admin_unknown_key_is_not_found(monkeypatch):
    monkeypatch.setattr(ops, "_resolve_caller", lambda: _make(_caller(caller_id="alice")))

    async def _no_body(_user_id):
        return None

    monkeypatch.setattr(management, "get_policy_body", _no_body)
    with pytest.raises(NotFoundError, match="user not found"):
        await ops.edit_api_key("ghost", {"description": "d"})


async def test_edit_api_key_non_admin_not_owned_is_forbidden(monkeypatch):
    monkeypatch.setattr(ops, "_resolve_caller", lambda: _make(_caller(caller_id="alice")))

    async def _bob_body(_user_id):
        return {"policy_data": {OWNER_USER_ID_CLAIM: "bob"}}

    monkeypatch.setattr(management, "get_policy_body", _bob_body)
    with pytest.raises(ForbiddenError, match="only edit API keys you own"):
        await ops.edit_api_key("k1", {"description": "d"})


async def test_edit_api_key_non_admin_scope_superset_is_bad_request(monkeypatch):
    monkeypatch.setattr(ops, "_resolve_caller", lambda: _make(_caller(caller_id="alice", scopes=["read"])))

    async def _own_body(_user_id):
        return {"policy_data": {OWNER_USER_ID_CLAIM: "alice"}}

    monkeypatch.setattr(management, "get_policy_body", _own_body)
    with pytest.raises(BadRequestError, match="exceed your own"):
        await ops.edit_api_key("k1", {"scopes": ["read", "write"]})


# -- ValueError -> BadRequestError mappings ----------------------------------


async def test_add_scope_url_value_error_maps_to_bad_request(monkeypatch):
    monkeypatch.setattr(ops, "access_control_settings", lambda: AccessControlSettings(enable=True))

    async def _boom(_scope_id, _url, _pattern):
        raise ValueError("bad scope mapping")

    monkeypatch.setattr(management, "add_url_to_scope", _boom)
    with pytest.raises(BadRequestError, match="bad scope mapping"):
        await ops.add_scope_url("s", "/u", None)


async def test_delete_scope_value_error_maps_to_bad_request(monkeypatch):
    async def _boom(_scope_id):
        raise ValueError("cannot delete")

    monkeypatch.setattr(management, "remove_scope", _boom)
    with pytest.raises(BadRequestError, match="cannot delete"):
        await ops.delete_scope("s")


async def test_revoke_api_key_value_error_maps_to_bad_request(monkeypatch):
    monkeypatch.setattr(ops, "_resolve_caller", lambda: _make(_caller(is_admin=True)))

    async def _boom(_user_id):
        raise ValueError("revoke failed")

    monkeypatch.setattr(management, "revoke_api_key", _boom)
    with pytest.raises(BadRequestError, match="revoke failed"):
        await ops.revoke_api_key("k1")


async def test_rollback_policy_restore_value_error_maps_to_bad_request(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(ops, "_resolve_caller", lambda: _make(_caller(is_admin=True)))
    monkeypatch.setattr("tai42_skeleton.versioning.versioned_store_configured", lambda: True)

    class _Store:
        async def get_version(self, _user_id, _version):
            return SimpleNamespace(body={"scopes": []})

    monkeypatch.setattr(ops, "ac_policy_store", lambda: _Store())

    async def _boom(_user_id, _body):
        raise ValueError("restore rejected")

    monkeypatch.setattr(management, "restore_policy_body", _boom)
    with pytest.raises(BadRequestError, match="restore rejected"):
        await ops.rollback_policy("k1", 1)


# -- helper ------------------------------------------------------------------


async def _make(caller: ops._Caller) -> ops._Caller:
    return caller


@pytest.fixture(autouse=True)
def _no_bound_caller():
    # Ensure the request-user contextvar is clean around each op-level test.
    token = set_request_user_id(None)
    try:
        yield
    finally:
        reset_request_user_id(token)
