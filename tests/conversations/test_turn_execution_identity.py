"""The turn runs AS the route's execution key: the shared tool seam enforces every tool
the agent calls mid-turn, and revocation follows the key's live grants.

Driven over the LIVE authz stack (real bind, run authorization and ``tools/binding.py``
seam) with only the routing-row manager, answer store and outbound channel faked. What
is pinned is the bridge's wiring: the identity is set for the turn, a denied tool never
runs and becomes a delivered error outcome, and the contextvar is released at turn end.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app
from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.app.instance import app
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.conversations import caps as caps_module
from tai42_skeleton.conversations import delivery as delivery_module
from tai42_skeleton.conversations import ledger as ledger_module
from tai42_skeleton.conversations import records as records_module
from tai42_skeleton.conversations import turn as turn_module
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations.registry import operation_registry
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

from .fake_record_redis import FakeRecordRedis, make_record_client_ctx

pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_PROBE_ROUTER = "tests.authz._fixtures.execution_probe"
_AGENT = "conv-agent"
_RUN_PATH = f"/api/agents/{_AGENT}/runs"
_FENCED_PATH = "/api/exec-probe/deploy/fenced"
_READ_PATH = "/api/exec-probe/read"
_PROBE_SCOPE = "probe"
_RUN_SCOPE = "agents"
_KEPT_SCOPE = "misc"


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "api_tools": {"enabled": True},
            "tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_b", "include": ["shout"]}],
            "routers_modules": [_PROBE_ROUTER, "tai42_skeleton.routers.agents"],
            "default_routers": "none",
        }
    )


class _ToolInput(BaseModel):
    user_message: str = ""


class ToolCallingAgent(Agent):
    """Resolves ONE tool by name mid-turn from the process-global facet and invokes it,
    answering with its result — the agent-facing seam an agent (or a subagent) crosses for
    a tool it resolves itself."""

    tool_name = "conv-agent"
    ToolInput = _ToolInput

    def __init__(self, tool: str, arguments: dict) -> None:
        self._tool = tool
        self._arguments = arguments

    async def run(self, *, user_message: str = "", thread_id: str | None = None, **kwargs):
        [tool] = await tai42_app.tools.get_client_tools([self._tool])
        return str(await tool.ainvoke(self._arguments))


class FakeManager:
    def __init__(self, route: ConversationRoute) -> None:
        self._route = route

    async def list_routes(self):
        return {self._route.route_name: self._route}

    async def get_route(self, name: str):
        return self._route if name == self._route.route_name else None


class FakeChannel:
    def __init__(self) -> None:
        self.sends: list = []

    async def notify(self, notification):
        self.sends.append(notification)
        return [f"out-{len(self.sends)}"]


class _FakeChannels:
    def __init__(self, channel: FakeChannel) -> None:
        self._channel = channel

    def get(self, name: str) -> FakeChannel:
        return self._channel


class _FakeDeliveryApp:
    def __init__(self, channel: FakeChannel) -> None:
        self.channels = _FakeChannels(channel)


def _route(execution_key: str) -> ConversationRoute:
    return ConversationRoute(
        route_name="line",
        door="channel",
        agent_name=_AGENT,
        execution_key=execution_key,
        channel="twilio",
        our_identity="+15550001111",
        execution_key_fingerprint=f"fp-{execution_key}",
    )


@pytest.fixture
def ac(monkeypatch) -> FakeAccessControlPg:
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    pg.add_route(_RUN_PATH, _RUN_SCOPE)
    pg.add_route(_FENCED_PATH, _PROBE_SCOPE)
    pg.add_route(_READ_PATH, _PROBE_SCOPE)
    # An admin key — clears the run door and the fenced fence.
    pg.add_policy("k-admin", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-admin"})
    # A non-admin key holding the run door's scope AND the probe scope: runs the agent,
    # clears the grantable read op's scope, but is fenced out of the fenced op.
    pg.add_policy(
        "k-run", scopes=[_RUN_SCOPE, _PROBE_SCOPE, _KEPT_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-run"}
    )
    # A non-admin key holding the run door's scope but NOT the probe scope: runs the agent
    # but is denied the grantable read op for want of scope.
    pg.add_policy("k-runonly", scopes=[_RUN_SCOPE, _KEPT_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-runonly"})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))
    reset_route_index()
    role_grants_module.reset_role_grants_cache()
    return pg


@pytest.fixture
def store(monkeypatch) -> ConversationRecordStore:
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")
    caps_module._CAPS_CACHE.clear()
    fake = FakeRecordRedis()
    monkeypatch.setattr(records_module, "client_ctx", make_record_client_ctx(fake))
    monkeypatch.setattr(ledger_module, "client_ctx", make_record_client_ctx(fake))
    return ConversationRecordStore(ConversationsSettings())


@pytest.fixture(autouse=True)
def _isolate_registries():
    routes_snapshot = dict(route_registry._routes)
    ops_snapshot = dict(operation_registry._operations)
    with tai42_app.bound(None):
        try:
            yield
        finally:
            route_registry._routes = routes_snapshot
            operation_registry._operations = ops_snapshot


def _wire_turn(monkeypatch, execution_key: str, tool: str, arguments: dict) -> tuple[FakeChannel, ConversationRoute]:
    route = _route(execution_key)
    manager = FakeManager(route)
    channel = FakeChannel()
    monkeypatch.setattr(turn_module, "get_conversations_manager", lambda: manager)
    monkeypatch.setattr(delivery_module, "get_conversations_manager", lambda: manager)
    monkeypatch.setattr(delivery_module, "tai42_app", _FakeDeliveryApp(channel))
    monkeypatch.setattr(turn_module, "_agent_registry", lambda: {_AGENT: ToolCallingAgent(tool, arguments)})
    return channel, route


async def _run_turn_answer(store: ConversationRecordStore) -> tuple[str, str, str | None]:
    """Accept one channel message, wait for its turn to write an outcome, and return the
    record's ``(answer_status, answer, error)``. The record exists from the moment the
    message is accepted, so the wait is for the outcome, not for the record."""
    import asyncio

    message_id = await turn_module.accept("twilio", "+15550001111", "+15550002222", "hi", f"PID-{time.time()}")
    for _ in range(200):
        record = await store.get_record(message_id)
        if record is not None and record.answer_status is not None:
            assert record.answer is not None
            return (record.answer_status, record.answer, record.error)
        await asyncio.sleep(0.01)
    raise AssertionError("the turn never wrote its outcome")


# -- the seam fires under the turn's bound identity ----------------------------


async def test_a_capability_tool_runs_within_a_turn(ac, store, monkeypatch):
    _wire_turn(monkeypatch, "k-run", "shout", {"text": "hi there"})
    async with app.app_context(_manifest()):
        answer_status, answer, _error = await _run_turn_answer(store)
    # A capability tool has no per-call scope model; it runs and its result is the answer,
    # so the turn plainly ran AS the key with the seam active.
    assert answer_status == "answered"
    assert answer == "hi there"


class _IdentityProbeAgent(Agent):
    """Reads the bound execution identity from inside the running turn and answers with
    the principal it is running as."""

    tool_name = _AGENT
    ToolInput = _ToolInput

    async def run(self, *, user_message: str = "", thread_id: str | None = None, **kwargs):
        identity = get_execution_identity()
        return "unbound" if identity is None else identity.user_id


async def test_the_identity_is_bound_for_the_turn_and_released_when_it_ends(ac, store, monkeypatch):
    # The binding lives in the TURN TASK's context (a copy of the accept-time one), so
    # both halves are read from inside that task: during the run, and at the record
    # write that follows the bound block.
    released: list[str | None] = []
    complete_turn = ConversationRecordStore.complete_turn

    async def _observe_then_complete(self, record):
        identity = get_execution_identity()
        released.append(None if identity is None else identity.user_id)
        return await complete_turn(self, record)

    _wire_turn(monkeypatch, "k-run", "shout", {})
    monkeypatch.setattr(turn_module, "_agent_registry", lambda: {_AGENT: _IdentityProbeAgent()})
    monkeypatch.setattr(ConversationRecordStore, "complete_turn", _observe_then_complete)

    async with app.app_context(_manifest()):
        answer_status, answer, _error = await _run_turn_answer(store)

    # The agent ran under the route's key...
    assert (answer_status, answer) == ("answered", "k-run")
    # ...and the moment the bound block exited, the turn's own context carried nothing —
    # the binding does not leak past the turn that opened it.
    assert released == [None]


def _probe_calls() -> list:
    # The probe router is re-imported on every boot, rebinding ``calls``, so the module
    # must be fetched from ``sys.modules`` AFTER the boot or the list reads empty.
    import sys

    return sys.modules[_PROBE_ROUTER].calls


async def test_an_operation_tool_the_key_lacks_is_denied_and_never_runs(ac, store, monkeypatch):
    _wire_turn(monkeypatch, "k-runonly", "exec_probe_read", {"mark": "m"})
    async with app.app_context(_manifest()):
        _probe_calls().clear()
        answer_status, answer, error = await _run_turn_answer(store)
        # The key runs the agent (holds the run scope) but lacks the read op's scope: the
        # seam denies the tool, so the turn is a delivered error outcome and the op body
        # never ran.
        assert answer_status == "error"
        assert "something went wrong" in answer.lower()
        assert error is not None
        assert "denied" in error
        assert _probe_calls() == []


async def test_a_grantable_operation_the_key_holds_runs_within_a_turn(ac, store, monkeypatch):
    _wire_turn(monkeypatch, "k-run", "exec_probe_read", {"mark": "m"})
    async with app.app_context(_manifest()):
        _probe_calls().clear()
        answer_status, answer, _error = await _run_turn_answer(store)
        # k-run holds the probe scope, so the same op the lacking key was denied runs here
        # — the deny above is the seam's, not the tool being unreachable.
        assert answer_status == "answered"
        assert answer == "read:m"
        assert _probe_calls() == [("read", "m")]


async def test_a_fenced_operation_is_denied_under_a_non_admin_key(ac, store, monkeypatch):
    _wire_turn(monkeypatch, "k-run", "exec_probe_fenced", {"target": "deploy", "mark": "m"})
    async with app.app_context(_manifest()):
        _probe_calls().clear()
        answer_status, _answer, error = await _run_turn_answer(store)
        # k-run HOLDS the probe scope (scope/jq pass allows); the deny is purely the per-tag
        # fence — a fenced op is admin-only even under a scope-holding key.
        assert answer_status == "error"
        assert error is not None
        assert "denied" in error
        assert _probe_calls() == []


async def test_a_fenced_operation_runs_under_an_admin_key(ac, store, monkeypatch):
    _wire_turn(monkeypatch, "k-admin", "exec_probe_fenced", {"target": "deploy", "mark": "m"})
    async with app.app_context(_manifest()):
        _probe_calls().clear()
        answer_status, answer, _error = await _run_turn_answer(store)
        # ALLOW parity, so the deny above is the fence and not an unreachable route.
        assert answer_status == "answered"
        assert answer == "fenced:deploy:m"
        assert _probe_calls() == [("fenced", "m")]


# -- revocation is automatic: no cascade, just the key's live grants -----------


async def test_a_deleted_execution_key_denies_the_turn(ac, store, monkeypatch):
    # A key with no policy: the identity build itself refuses, so deleting the key
    # revokes the route with no cascade.
    _wire_turn(monkeypatch, "ghost", "shout", {"text": "hi"})
    async with app.app_context(_manifest()):
        answer_status, _answer, error = await _run_turn_answer(store)
    assert answer_status == "error"
    assert error is not None
    assert "denied" in error


async def test_descoping_the_key_denies_the_next_turn(ac, store, monkeypatch):
    from tai42_skeleton.access_control.settings import access_control_settings

    # First turn: k-run holds the run-door scope, so the run is authorized and the
    # capability tool answers.
    _wire_turn(monkeypatch, "k-run", "shout", {"text": "ok"})
    async with app.app_context(_manifest()):
        answer_status, answer, _error = await _run_turn_answer(store)
        assert answer_status == "answered"
        assert answer == "ok"

        # De-scope the LIVE key; the version bump makes the next read a cache miss.
        policy = ac.policy("k-run")
        policy["scopes"] = [_KEPT_SCOPE]
        version_key = access_control_settings().policy_version_key
        store_module_redis = verifier_module.client_ctx
        # The verifier and policy enforcer share the fake redis; bump the version counter.
        async with store_module_redis(None) as r:  # type: ignore[misc]
            await r.set(version_key, "2")

        answer_status2, _answer2, error2 = await _run_turn_answer(store)
    # The next turn under the SAME key is denied at the run gate: every turn reads the
    # key's live grants.
    assert answer_status2 == "error"
    assert error2 is not None
    assert "denied" in error2
