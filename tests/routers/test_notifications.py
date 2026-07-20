"""Notifications router: the authed inbox door returns the sink's records,
newest-first, inside the ``{"data": ...}`` envelope."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

import pytest
from starlette.requests import Request
from tai_contract.access_control import OWNER_USER_ID_CLAIM
from tai_contract.access_control.context import reset_request_user_id, set_request_user_id

from tai_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai_skeleton.channels import notifications_sink
from tai_skeleton.operations import notifications as notifications_ops
from tai_skeleton.routers import notifications as router


@contextmanager
def _identity(*, user_id: str | None = None, owner: str | None = None) -> Iterator[None]:
    """Bind a caller identity: ``owner`` set makes it a RESTRICTED owned key (reads its
    own per-identity feed, keyed on its OWN id ``user_id`` — NOT its owner), ``owner=None``
    an unrestricted caller (shared feed). Tests pass an ``owner`` DIFFERENT from
    ``user_id`` so the key-own-vs-owner distinction is exercised."""
    claims: dict[str, str] = {} if owner is None else {OWNER_USER_ID_CLAIM: owner}
    uid_token = set_request_user_id(user_id) if user_id is not None else None
    claims_token = set_request_identity_claims(claims)
    try:
        yield
    finally:
        reset_request_identity_claims(claims_token)
        if uid_token is not None:
            reset_request_user_id(uid_token)


@pytest.fixture
def sink_redis(monkeypatch, fake_redis):
    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake_redis

    monkeypatch.setattr(notifications_sink, "client_ctx", _ctx)
    return fake_redis


def _get_request() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/api/notifications", "query_string": b"", "headers": []}
    return Request(scope)


async def test_list_notifications_empty(sink_redis) -> None:
    resp = await router.list_notifications(_get_request())
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body)) == {"data": {"notifications": []}}


async def test_list_notifications_returns_sink_records_newest_first(sink_redis) -> None:
    first = await notifications_sink.record_notification("first")
    second = await notifications_sink.record_notification("second", recipient="ops")

    resp = await router.list_notifications(_get_request())

    assert json.loads(bytes(resp.body)) == {"data": {"notifications": [second, first]}}


def _post_request(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/notifications", "query_string": b"", "headers": []}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def test_notify_user_route_sends_and_wraps_in_envelope(monkeypatch) -> None:
    sent: list = []

    async def _helper(message, *, channel=None, recipient=None, audience=None):
        sent.append((message, channel, recipient, audience))

    monkeypatch.setattr(notifications_ops, "_notify_user", _helper)
    resp = await router.notify_user(_post_request(b'{"message": "hi", "channel": "telegram"}'))
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body)) == {"data": "notification sent via 'telegram'"}
    assert sent == [("hi", "telegram", None, None)]


async def test_notify_user_route_threads_audience_to_helper(monkeypatch) -> None:
    sent: list = []

    async def _helper(message, *, channel=None, recipient=None, audience=None):
        sent.append((message, channel, recipient, audience))

    monkeypatch.setattr(notifications_ops, "_notify_user", _helper)
    resp = await router.notify_user(
        _post_request(b'{"message": "hi", "channel": "sms", "recipient": "+15550000000", "audience": "alice"}')
    )
    assert resp.status_code == 200
    # Both axes are threaded independently: recipient (address) AND audience (identity).
    assert sent == [("hi", "sms", "+15550000000", "alice")]


async def test_notify_user_route_maps_valueerror_to_400(monkeypatch) -> None:
    async def _helper(message, *, channel=None, recipient=None, audience=None):
        raise ValueError("channel must be a non-empty string")

    monkeypatch.setattr(notifications_ops, "_notify_user", _helper)
    resp = await router.notify_user(_post_request(b'{"message": "hi", "channel": ""}'))
    assert resp.status_code == 400
    assert "channel" in json.loads(bytes(resp.body))["error"]


# -- audience isolation on the read door -------------------------------------


def _ids(resp) -> list[str]:
    return [n["message"] for n in json.loads(bytes(resp.body))["data"]["notifications"]]


async def test_restricted_caller_reads_only_its_own_records(sink_redis) -> None:
    # A broadcast, a keyA-addressed record, a record addressed to keyA's OWNER "alice",
    # and a bob-addressed record.
    await notifications_sink.record_notification("broadcast")
    await notifications_sink.record_notification("for-keyA", audience="keyA")
    await notifications_sink.record_notification("for-owner", audience="alice")
    await notifications_sink.record_notification("for-bob", audience="bob")
    with _identity(user_id="keyA", owner="alice"):
        resp = await router.list_notifications(_get_request())
    # keyA sees ONLY records addressed to its OWN id — not the broadcast, not bob's, and
    # NOT its owner's (key-keyed, not owner-keyed isolation).
    assert _ids(resp) == ["for-keyA"]


async def test_unrestricted_caller_reads_full_shared_feed(sink_redis) -> None:
    await notifications_sink.record_notification("broadcast")
    await notifications_sink.record_notification("for-alice", audience="alice")
    with _identity(user_id="op1", owner=None):
        resp = await router.list_notifications(_get_request())
    # The operator sees everything on the shared feed (both, newest-first).
    assert _ids(resp) == ["for-alice", "broadcast"]


async def test_restricted_feed_stays_complete_when_shared_feed_flooded(sink_redis, monkeypatch) -> None:
    # The completeness pin: keyA's addressed record, then the shared feed flooded
    # past its cap by OTHER records. keyA's per-identity feed still contains its
    # record — proving a per-identity feed, not a post-filtered shared window (which
    # would have LTRIM'd its record out before any filter ran).
    from tai_skeleton.interactions.settings import InteractionsSettings

    cap = 5
    monkeypatch.setattr(
        notifications_sink, "interactions_settings", lambda: InteractionsSettings(notifications_feed_max=cap)
    )

    await notifications_sink.record_notification("for-keyA", audience="keyA")
    for i in range(cap + 3):
        await notifications_sink.record_notification(f"flood-{i}")

    # The shared feed evicted keyA's record (an unrestricted read no longer has it)…
    with _identity(user_id="op1", owner=None):
        shared = await router.list_notifications(_get_request())
    assert "for-keyA" not in _ids(shared)
    # …but keyA's own per-identity feed keeps it, complete.
    with _identity(user_id="keyA", owner="alice"):
        own = await router.list_notifications(_get_request())
    assert _ids(own) == ["for-keyA"]


async def test_record_without_audience_is_broadcast_only(sink_redis) -> None:
    # A record with no audience field has no per-identity feed entry: an unrestricted
    # caller reads it, a restricted caller does not.
    await notifications_sink.record_notification("broadcast")
    with _identity(user_id="op1", owner=None):
        assert _ids(await router.list_notifications(_get_request())) == ["broadcast"]
    with _identity(user_id="keyA", owner="alice"):
        assert _ids(await router.list_notifications(_get_request())) == []
