"""The API door's caller authorization, over the REAL auth stack.

``POST /api/conversations/{route_name}/messages`` decides WHO may send from the
PRESENTING CALLER's policy, before the handler runs and independent of the route's
``execution_key``. The turn engine is stubbed; an admitted call must carry the CALLER's
own principal into the turn.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
from tai42_identity_redis import redis_api_key_provider as provider_module
from tai42_kit.utils.data.string_util import hash_api_key

import tai42_skeleton.conversations as conversations_package
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.conversations.turn import ApiSubmitResult
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, _FakeApp, make_client_ctx, make_pg_ctx

# The enforcer's alru cache is created in the test's loop and first used in the
# TestClient's portal loop — a benign loop-change reset that is a test artifact.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_MESSAGES_PATTERN = r"/api/conversations/.+/messages"
_TEMPLATE = "conversations-messages"
_SCOPE = "conversations-protected"
_PATH = "/api/conversations/support/messages"
# The two credentials, and the principals the identity store resolves them to.
_SENDER_KEY = "sender-key"
_STRANGER_KEY = "stranger-key"


def _door():
    """The door handler itself. Its module registers the conversation routes on the
    server's HTTP surface at import, so the import runs with the real app bound — exactly
    as a boot does."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import app as skeleton_app

    with tai42_app.bound(skeleton_app):
        from tai42_skeleton.routers import conversations as router

    return router.send_conversation_message


class _Recorder:
    """Stands in for the turn engine so the door's gate is what is under test: every call
    that reaches it is recorded, and a denied caller must produce none."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def __call__(self, route_name, external_user_id, text, caller_principal, wait_seconds):
        self.calls.append((route_name, external_user_id, caller_principal))
        return ApiSubmitResult(message_id="m-1", thread_id=f"bridge:{route_name}:{external_user_id}", answer=None)


@pytest.fixture
def turns(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(conversations_package, "submit_api_message", recorder)
    return recorder


@pytest.fixture
def bound_app():
    """A fake ``tai42_app`` so the enforcer can render an admitted key's (empty) policy
    condition."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    with tai42_app.bound(app):
        yield app


@pytest.fixture
def client(monkeypatch, bound_app) -> TestClient:
    """The door behind the real auth stack: the identity store resolves each credential to
    its principal, and the policy store carries that principal's grants.

    One request per client — the enforcer's cache is built in the test's loop and used in
    the client's portal loop, so a second request on the same client is decided under a
    reset cache rather than under this fixture's store."""
    ac_settings = AccessControlSettings(path_patterns={_MESSAGES_PATTERN: _TEMPLATE})
    redis = FakeRedis(
        hashes={
            f"{ac_settings.key_prefix}{hash_api_key(_SENDER_KEY)}": {"user_id": "sender", "description": "d"},
            f"{ac_settings.key_prefix}{hash_api_key(_STRANGER_KEY)}": {"user_id": "stranger", "description": "d"},
        }
    )
    pg = FakeAccessControlPg()
    pg.add_route(_TEMPLATE, _SCOPE)
    # ``sender`` holds the door's resource scope, ``stranger`` a different one: what
    # separates them is the door's decision, not authentication.
    pg.add_policy("sender", scopes=[_SCOPE])
    pg.add_policy("stranger", scopes=["unrelated"])

    ctx = make_client_ctx(redis)
    monkeypatch.setattr(verifier_module, "client_ctx", ctx)
    monkeypatch.setattr(policy_module, "client_ctx", ctx)
    monkeypatch.setattr(provider_module, "client_ctx", ctx)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    role_grants_module.reset_role_grants_cache()

    routes = [Route("/api/conversations/{route_name}/messages", _door(), methods=["POST"])]
    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)


def _send(client: TestClient, key: str | None = None):
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post(_PATH, json={"external_user_id": "u-7", "text": "hello"}, headers=headers)


def test_an_uncredentialed_caller_never_reaches_the_turn(client, turns):
    response = _send(client)
    assert response.status_code == 401
    assert turns.calls == []


def test_a_caller_without_the_doors_scope_is_denied(client, turns):
    # The route's key has real authority; the refusal is the door's decision about the
    # CALLER, taken before any turn is scheduled and before the row is read.
    response = _send(client, _STRANGER_KEY)
    assert response.status_code == 403
    assert turns.calls == []


def test_an_authorized_caller_is_admitted_and_invokes_the_turn_as_itself(client, turns):
    response = _send(client, _SENDER_KEY)
    assert response.status_code == 202
    assert response.json()["data"]["message_id"] == "m-1"
    # The turn carries the authorized CALLER — the principal the caller-scoped read door
    # later matches on — not the route's execution key.
    assert turns.calls == [("support", "u-7", "sender")]
