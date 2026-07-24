"""The conversation-route CRUD operations: create (pass-role bind + agent-exists +
callback-secret mint), get/list (secret withheld), delete, and the slug guard."""

from __future__ import annotations

import pytest
from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations import conversations as ops
from tai42_skeleton.operations.errors import BadRequestError, NotFoundError


class _DictManager(BaseConversationsManager):
    """A dict-backed routing-row store standing in for the redis manager (it is NOT the
    in-memory 501 manager, so ``_require_backend`` admits it)."""

    def __init__(self) -> None:
        super().__init__(ConversationsSettings())
        self.rows: dict[str, ConversationRoute] = {}

    async def put_route(self, route: ConversationRoute) -> bool:
        created = route.route_name not in self.rows
        self.rows[route.route_name] = route
        return created

    async def get_route(self, route_name: str) -> ConversationRoute | None:
        return self.rows.get(route_name)

    async def delete_route(self, route_name: str) -> bool:
        return self.rows.pop(route_name, None) is not None

    async def list_routes(self) -> dict[str, ConversationRoute]:
        return dict(self.rows)


class _FakeAgents:
    def __init__(self, names: set[str]) -> None:
        self._names = names

    def all_agents(self) -> dict[str, object]:
        return {name: object() for name in self._names}


class _FakeApp:
    def __init__(self, agents: set[str]) -> None:
        self.agents = _FakeAgents(agents)


@pytest.fixture
def wired(monkeypatch):
    """Wire a dict-backed manager, a pass-role bind that returns a fingerprint, and an
    agent registry holding ``triage`` — the standard happy-path environment."""
    manager = _DictManager()
    monkeypatch.setattr(ops, "get_conversations_manager", lambda: manager)

    async def _bindable(caller, execution_key):
        return "fp-derived"

    async def _caller():
        return object()

    monkeypatch.setattr(ops, "assert_execution_key_bindable", _bindable)
    monkeypatch.setattr(ops, "resolve_caller", _caller)

    from tai42_skeleton.app import instance

    monkeypatch.setattr(instance, "app", _FakeApp({"triage"}), raising=False)
    return manager


async def test_create_api_route_mints_and_shows_the_secret_once(wired):
    result = await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc",
        callback_url="https://example.com/cb",
    )
    assert result["created"] is True
    assert result["callback_secret"]  # shown once here
    # The stored fingerprint is the one the bind derived, never a client value.
    assert wired.rows["support"].execution_key_fingerprint == "fp-derived"
    assert wired.rows["support"].callback_secret == result["callback_secret"]
    # The route view withholds the secret.
    assert "callback_secret" not in result["route"]


async def test_create_channel_route_carries_no_secret(wired):
    result = await ops.create_conversation_route(
        route_name="line",
        door="channel",
        agent_name="triage",
        execution_key="svc",
        channel="twilio",
        our_identity="+15550001111",
    )
    assert result["callback_secret"] is None
    assert wired.rows["line"].callback_secret is None


async def test_create_is_an_upsert(wired):
    await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc",
        callback_url="https://example.com/cb",
    )
    result = await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc2",
        callback_url="https://example.com/cb2",
    )
    assert result["created"] is False
    assert wired.rows["support"].execution_key == "svc2"


async def test_a_channel_identity_is_stored_canonicalized(wired):
    # Inbound routing matches by equality on the canonical form, so the row is stored
    # canonicalized — a verbatim row would match nothing and hide duplicates.
    await ops.create_conversation_route(
        route_name="line",
        door="channel",
        agent_name="triage",
        execution_key="svc",
        channel="twilio",
        our_identity="  +15550001111  ",
    )
    assert wired.rows["line"].our_identity == "+15550001111"


async def test_a_second_route_claiming_one_channel_identity_is_refused(wired):
    await ops.create_conversation_route(
        route_name="line-a",
        door="channel",
        agent_name="triage",
        execution_key="svc",
        channel="twilio",
        our_identity="+15550001111 ",
    )
    # The same number under a different spelling is one identity, so the second route is
    # refused here rather than leaving the routing table unresolvable.
    with pytest.raises(BadRequestError, match="already routed by 'line-a'"):
        await ops.create_conversation_route(
            route_name="line-b",
            door="channel",
            agent_name="triage",
            execution_key="svc",
            channel="twilio",
            our_identity="+15550001111",
        )
    assert "line-b" not in wired.rows


async def test_a_route_may_re_claim_its_own_channel_identity(wired):
    # The create door is an UPSERT: editing a row must not trip over the identity that row
    # itself already holds.
    for execution_key in ("svc", "svc2"):
        await ops.create_conversation_route(
            route_name="line",
            door="channel",
            agent_name="triage",
            execution_key=execution_key,
            channel="twilio",
            our_identity="+15550001111",
        )
    assert wired.rows["line"].execution_key == "svc2"


async def test_the_same_identity_on_another_channel_is_a_different_route(wired):
    # The pair is (channel, identity): the same handle on a different medium is a
    # different destination, and both rows resolve.
    for route_name, channel in (("line", "twilio"), ("tg", "telegram")):
        await ops.create_conversation_route(
            route_name=route_name,
            door="channel",
            agent_name="triage",
            execution_key="svc",
            channel=channel,
            our_identity="+15550001111",
        )
    assert set(wired.rows) == {"line", "tg"}


async def test_create_rejects_a_colon_channel_name(wired):
    # The channel name qualifies the dedupe and outbound-index keys ahead of the provider's
    # own id, so a ``:`` in it would move that boundary.
    with pytest.raises(BadRequestError, match="free of ':'"):
        await ops.create_conversation_route(
            route_name="line",
            door="channel",
            agent_name="triage",
            execution_key="svc",
            channel="twi:lio",
            our_identity="+15550001111",
        )
    assert wired.rows == {}


async def test_create_rejects_unknown_agent(wired):
    with pytest.raises(NotFoundError, match="agent not found"):
        await ops.create_conversation_route(
            route_name="support",
            door="api",
            agent_name="ghost",
            execution_key="svc",
            callback_url="https://example.com/cb",
        )


async def test_create_rejects_a_colon_route_name(wired):
    with pytest.raises(BadRequestError, match="slug"):
        await ops.create_conversation_route(
            route_name="bad:name",
            door="api",
            agent_name="triage",
            execution_key="svc",
            callback_url="https://example.com/cb",
        )


async def test_create_bind_refusal_leaves_no_row(wired, monkeypatch):
    async def _refuse(caller, execution_key):
        raise BadRequestError("not yours")

    monkeypatch.setattr(ops, "assert_execution_key_bindable", _refuse)
    with pytest.raises(BadRequestError, match="not yours"):
        await ops.create_conversation_route(
            route_name="support",
            door="api",
            agent_name="triage",
            execution_key="svc",
            callback_url="https://example.com/cb",
        )
    assert "support" not in wired.rows


async def test_get_withholds_the_secret_and_404s_unknown(wired):
    await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc",
        callback_url="https://example.com/cb",
    )
    view = await ops.get_conversation_route("support")
    assert "callback_secret" not in view
    assert view["route_name"] == "support"
    with pytest.raises(NotFoundError):
        await ops.get_conversation_route("missing")


async def test_get_rejects_a_colon_route_name(wired):
    with pytest.raises(BadRequestError, match="slug"):
        await ops.get_conversation_route("bad:name")


async def test_list_withholds_secrets(wired):
    await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc",
        callback_url="https://example.com/cb",
    )
    listed = await ops.list_conversation_routes()
    assert listed["total"] == 1
    assert all("callback_secret" not in item for item in listed["items"])


async def test_delete_removes_then_404s(wired):
    await ops.create_conversation_route(
        route_name="support",
        door="api",
        agent_name="triage",
        execution_key="svc",
        callback_url="https://example.com/cb",
    )
    assert (await ops.delete_conversation_route("support"))["removed"] is True
    with pytest.raises(NotFoundError):
        await ops.delete_conversation_route("support")


async def test_unclaimed_channel_identity_rejects_a_blank_identity(wired):
    # The identity guard canonicalizes before comparing; a value blank once trimmed keys no
    # route, so it is refused as a 400 rather than stored unresolvable.
    with pytest.raises(BadRequestError, match="invalid our_identity"):
        await ops._unclaimed_channel_identity(wired, route_name="line", channel="twilio", our_identity="   ")


async def test_operations_501_without_a_backend(monkeypatch):
    from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager
    from tai42_skeleton.operations.errors import NotSupportedError

    monkeypatch.setattr(ops, "get_conversations_manager", lambda: InMemoryConversationsManager(ConversationsSettings()))
    with pytest.raises(NotSupportedError):
        await ops.list_conversation_routes()
