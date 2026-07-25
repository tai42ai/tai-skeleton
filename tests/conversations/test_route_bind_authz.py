"""Route create/edit binds the ``execution_key`` through the shared bind gate:
pass-role plus the token-free-evaluable condition scan.

The gate runs for real against a faked policy store (only acting-caller resolution is
stubbed); a refusal must write NO row.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations import conversations as ops
from tai42_skeleton.operations.errors import BadRequestError, ForbiddenError
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")


class _DictManager(BaseConversationsManager):
    """A non-in-memory routing-row store that records every write, so a refused bind can
    be shown to have stored nothing."""

    def __init__(self) -> None:
        super().__init__(ConversationsSettings())
        self.rows: dict[str, object] = {}

    async def put_route(self, route) -> bool:
        created = route.route_name not in self.rows
        self.rows[route.route_name] = route
        return created

    async def get_route(self, route_name):
        return self.rows.get(route_name)

    async def delete_route(self, route_name):
        return self.rows.pop(route_name, None) is not None

    async def list_routes(self):
        return dict(self.rows)


class _FakeAgents:
    def all_agents(self):
        return {"triage": object()}


class _FakeApp:
    agents = _FakeAgents()


class _FakeResourceManager:
    async def render_by_id_or_content(self, *, content, template_id, kwargs):
        # The auth gate's policy condition is inline jq — returned unchanged, as the real
        # renderer does for inline content.
        return content


class _FakeStorage:
    resource_manager = _FakeResourceManager()


class _FakeCondApp:
    """Bound onto ``tai42_app`` so the token-free scan can render a policy condition."""

    storage = _FakeStorage()


@pytest.fixture
def env(monkeypatch):
    """A faked policy store carrying the keys under test, the dict routing manager, and a
    fake agent registry — the real bind gate reads the store, only the acting caller is
    set per test."""
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    # alice: a non-admin caller (a real scope, no owner claim, not ``"*"``).
    pg.add_policy("alice", scopes=["conversations"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-alice"})
    # root: the admin discriminator — a condition-free ``"*"`` policy that is not owned.
    pg.add_policy("root", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-root"})
    # A key alice owns, token-free-evaluable — the bindable one.
    pg.add_policy(
        "k-owned",
        scopes=["conversations"],
        policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-owned"},
    )
    # carol owns ``k-other`` and carries real authority, so the only refusal on that key
    # is pass-role.
    pg.add_policy("carol", scopes=["conversations"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-carol"})
    # A key owned by someone else — alice may not delegate it.
    pg.add_policy(
        "k-other",
        scopes=["conversations"],
        policy_data={OWNER_USER_ID_CLAIM: "carol", KEY_FINGERPRINT_CLAIM: "fp-k-other"},
    )
    # A key alice owns whose jq condition a tokenless fire cannot evaluate.
    pg.add_policy(
        "k-cond",
        scopes=["conversations"],
        policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-cond"},
        condition='.identity.department == "eng"',
    )
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))
    role_grants_module.reset_role_grants_cache()

    manager = _DictManager()
    monkeypatch.setattr(ops, "get_conversations_manager", lambda: manager)
    from tai42_skeleton.app import instance

    monkeypatch.setattr(instance, "app", _FakeApp(), raising=False)

    from tai42_contract.app import tai42_app

    with tai42_app.bound(_FakeCondApp()):
        yield manager


def _as_caller(monkeypatch, user_id: str) -> None:
    from tai42_contract.access_control.context import set_request_user_id

    set_request_user_id(user_id)
    monkeypatch.setattr(ops, "get_execution_identity", lambda: None, raising=False)


async def _create(execution_key: str) -> dict:
    return await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key=execution_key,
        callback_url="https://cb.example/x",
    )


# -- pass-role: may this caller delegate this key? ----------------------------


async def test_binds_a_key_the_caller_owns(env, monkeypatch):
    _as_caller(monkeypatch, "alice")
    result = await _create("k-owned")
    assert result["created"] is True
    assert "support" in env.rows


async def test_rejects_a_key_the_caller_does_not_own(env, monkeypatch):
    _as_caller(monkeypatch, "alice")
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own"):
        await _create("k-other")
    # A refused bind leaves NO row behind.
    assert "support" not in env.rows


async def test_admin_may_bind_a_key_it_does_not_own(env, monkeypatch):
    _as_caller(monkeypatch, "root")
    result = await _create("k-other")
    assert result["created"] is True


# -- the token-free-evaluable condition scan ----------------------------------


async def test_rejects_a_token_dependent_condition_key(env, monkeypatch):
    _as_caller(monkeypatch, "alice")
    # alice OWNS k-cond, so pass-role clears and the refusal is the condition scan —
    # carrying its raw diagnostic, which she may read.
    with pytest.raises(BadRequestError, match="condition reads an identity claim beyond"):
        await _create("k-cond")
    assert "support" not in env.rows
