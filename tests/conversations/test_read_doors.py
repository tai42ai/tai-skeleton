"""The caller-scoped operator read doors: one answer record by id, and the
admin-tier failed-delivery listing.

An api-door record is readable only by the caller that invoked the turn or an admin; a
channel-door record has a null ``caller_principal``, so it is admin-only. Each deny
asserts the SPECIFIC typed error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from tai42_skeleton.conversations import records as records_module
from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.models import ConversationRecord, DeliveryStatus
from tai42_skeleton.conversations.records import ConversationRecordStore
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations import conversations as ops
from tai42_skeleton.operations.errors import ForbiddenError, NotFoundError

from .fake_record_redis import FakeRecordRedis, make_record_client_ctx


class _DictManager(BaseConversationsManager):
    """A non-in-memory routing-row store so ``_require_backend`` admits the read door."""

    def __init__(self) -> None:
        super().__init__(ConversationsSettings())

    async def put_route(self, route):
        raise NotImplementedError

    async def get_route(self, route_name):
        return None

    async def delete_route(self, route_name):
        return False

    async def list_routes(self):
        return {}


@dataclass
class _Caller:
    caller_id: str
    is_admin: bool


def _record(
    message_id: str,
    *,
    door: str,
    caller_principal: str | None,
    status: DeliveryStatus,
    error: str | None = None,
) -> ConversationRecord:
    now = time.time()
    return ConversationRecord(
        error=error,
        message_id=message_id,
        route_name="support",
        door=door,  # type: ignore[arg-type]
        thread_id=f"bridge:support:{message_id}",
        client_address="user-7",
        channel="twilio" if door == "channel" else None,
        our_identity="+15550001111" if door == "channel" else None,
        callback_url="https://cb.example/x" if door == "api" else None,
        caller_principal=caller_principal,
        answer_status="answered",
        answer="the answer",
        delivery_status=status,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def store(monkeypatch) -> ConversationRecordStore:
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")
    fake = FakeRecordRedis()
    monkeypatch.setattr(records_module, "client_ctx", make_record_client_ctx(fake))
    monkeypatch.setattr(ops, "get_conversations_manager", lambda: _DictManager())
    return ConversationRecordStore(ConversationsSettings())


def _as_caller(monkeypatch, caller: _Caller) -> None:
    async def _resolve():
        return caller

    monkeypatch.setattr(ops, "resolve_caller", _resolve)


# -- get_conversation_message: api records are caller-or-admin ----------------


async def test_api_record_readable_by_its_invoking_caller(store, monkeypatch):
    await store.create_record(_record("m1", door="api", caller_principal="alice", status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("alice", is_admin=False))
    view = await ops.get_conversation_message("support", "m1")
    assert view["message_id"] == "m1"
    assert view["caller_principal"] == "alice"


async def test_api_record_hidden_from_another_caller(store, monkeypatch):
    await store.create_record(_record("m1", door="api", caller_principal="alice", status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("bob", is_admin=False))
    with pytest.raises(ForbiddenError, match="only read conversation records from turns you invoked"):
        await ops.get_conversation_message("support", "m1")


async def test_api_record_readable_by_admin(store, monkeypatch):
    await store.create_record(_record("m1", door="api", caller_principal="alice", status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("root", is_admin=True))
    view = await ops.get_conversation_message("support", "m1")
    assert view["message_id"] == "m1"


# -- the caller reads a projection, the admin reads the record ----------------

#: The internal detail of a turn that ran as the ROUTE's execution key — a principal the
#: invoking caller has no authority over — plus the delivery machine's own bookkeeping.
_INTERNAL_FIELDS = ("error", "attempts", "outbound_message_ids", "callback_url", "channel", "our_identity")


async def test_a_non_admin_caller_never_reads_the_internal_turn_detail(store, monkeypatch):
    # A denied turn stores the raw refusal, which names the route key's grants; the
    # caller reads the outcome and nothing about that principal.
    denial = "turn denied: access denied: POST /api/agents/foo/runs is not permitted for 'finance-svc'"
    await store.create_record(
        _record("m1", door="api", caller_principal="alice", status=DeliveryStatus.FAILED, error=denial)
    )
    _as_caller(monkeypatch, _Caller("alice", is_admin=False))
    view = await ops.get_conversation_message("support", "m1")

    assert view["message_id"] == "m1"
    assert view["answer"] == "the answer"
    assert view["delivery_status"] == "failed"
    for field in _INTERNAL_FIELDS:
        assert field not in view
    assert "finance-svc" not in repr(view)


async def test_an_admin_reads_the_whole_record_including_the_internal_detail(store, monkeypatch):
    denial = "turn denied: access denied for 'finance-svc'"
    await store.create_record(
        _record("m1", door="api", caller_principal="alice", status=DeliveryStatus.FAILED, error=denial)
    )
    _as_caller(monkeypatch, _Caller("root", is_admin=True))
    view = await ops.get_conversation_message("support", "m1")

    assert view["error"] == denial
    for field in _INTERNAL_FIELDS:
        assert field in view


# -- get_conversation_message: channel records are admin-only -----------------


async def test_channel_record_is_admin_only(store, monkeypatch):
    await store.create_record(_record("c1", door="channel", caller_principal=None, status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("alice", is_admin=False))
    with pytest.raises(ForbiddenError, match="only read conversation records from turns you invoked"):
        await ops.get_conversation_message("support", "c1")


async def test_channel_record_readable_by_admin(store, monkeypatch):
    await store.create_record(_record("c1", door="channel", caller_principal=None, status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("root", is_admin=True))
    view = await ops.get_conversation_message("support", "c1")
    assert view["message_id"] == "c1"


async def test_unknown_or_cross_route_record_is_404(store, monkeypatch):
    await store.create_record(_record("m1", door="api", caller_principal="alice", status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("alice", is_admin=False))
    with pytest.raises(NotFoundError, match="conversation record not found"):
        await ops.get_conversation_message("support", "missing")
    # A record that exists under a DIFFERENT route is a 404 (not a 403) — it is not this
    # route's record to reveal even the existence of.
    with pytest.raises(NotFoundError, match="conversation record not found"):
        await ops.get_conversation_message("other-route", "m1")


# -- list_failed_conversations: admin-tier ------------------------------------


async def test_failed_listing_is_admin_only(store, monkeypatch):
    await store.create_record(_record("f1", door="api", caller_principal="alice", status=DeliveryStatus.FAILED))
    _as_caller(monkeypatch, _Caller("alice", is_admin=False))
    with pytest.raises(ForbiddenError, match="restricted to administrators"):
        await ops.list_failed_conversations()


async def test_failed_listing_returns_only_failed_records_for_admin(store, monkeypatch):
    await store.create_record(_record("f1", door="api", caller_principal="alice", status=DeliveryStatus.FAILED))
    await store.create_record(_record("d1", door="api", caller_principal="alice", status=DeliveryStatus.DELIVERED))
    _as_caller(monkeypatch, _Caller("root", is_admin=True))
    listed = await ops.list_failed_conversations()
    assert listed["total"] == 1
    assert listed["items"][0]["message_id"] == "f1"
