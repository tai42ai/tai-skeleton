"""Op-level oracles for the notifications operations.

``notify_user`` forwards every argument verbatim to the channels helper, returns the
bare confirmation string, and NEVER swallows a failure — mapping the helper's loud
errors to the operation's typed errors (ValueError→400, NotImplementedError→501,
ChannelDeliveryError→502). ``list_notifications`` reads the sink, and the destructive
projection carries ``destructiveHint``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.app import tai42_app
from tai42_contract.channels import ChannelDeliveryError, ChannelNotification
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai42_skeleton.app.instance import app
from tai42_skeleton.channels import notifications_sink
from tai42_skeleton.operations import (
    BadRequestError,
    ForbiddenError,
    NotSupportedError,
    OperationRegistry,
    UpstreamError,
    operation_metadata_of,
)
from tai42_skeleton.operations import notifications as notifications_ops
from tai42_skeleton.operations.projection import project_operations


@contextmanager
def _restricted(own_id: str, owner: str | None = None) -> Iterator[None]:
    """Bind a RESTRICTED owned-key caller isolated to its OWN id ``own_id``. The owner
    claim (a DISTINCT ``owner-of-{own_id}`` by default) is what MARKS the caller
    restricted, but the isolation identity is its own id — each key is its own island —
    so the write clamp scopes writes to ``own_id``, never the owner."""
    owner_claim = owner if owner is not None else f"owner-of-{own_id}"
    claims_token = set_request_identity_claims({OWNER_USER_ID_CLAIM: owner_claim})
    uid_token = set_request_user_id(own_id)
    try:
        yield
    finally:
        reset_request_user_id(uid_token)
        reset_request_identity_claims(claims_token)


@pytest.fixture
def sink_redis(monkeypatch, fake_redis):
    """Point the internal notifications sink's Redis at the shared fake so the real
    helper's ``channel=None`` writes land somewhere readable back in the test."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake_redis

    monkeypatch.setattr(notifications_sink, "client_ctx", _ctx)
    return fake_redis


class _RecordingHelper:
    def __init__(self, *, raise_exc: Exception | None = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self._raise = raise_exc

    async def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))
        if self._raise is not None:
            raise self._raise


# -- notify_user --------------


async def test_notify_user_forwards_arguments_and_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper()
    monkeypatch.setattr(notifications_ops, "_notify_user", helper)

    result = await notifications_ops.notify_user("Deploy finished", channel="telegram", recipient="@ops")

    assert result == "notification sent via 'telegram'"
    assert helper.calls == [(("Deploy finished",), {"channel": "telegram", "recipient": "@ops", "audience": None})]


async def test_notify_user_defaults_forwarded_and_maps_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    # The op forwards ``channel=None``/``recipient=None`` verbatim; the helper's
    # missing-channel ValueError is mapped to a loud BadRequestError (400), never
    # swallowed.
    helper = _RecordingHelper(raise_exc=ValueError("no internal notifications surface configured; name a channel"))
    monkeypatch.setattr(notifications_ops, "_notify_user", helper)

    with pytest.raises(BadRequestError, match="no internal notifications surface configured"):
        await notifications_ops.notify_user("hello")

    assert helper.calls == [(("hello",), {"channel": None, "recipient": None, "audience": None})]


async def test_notify_user_channel_omitted_records_to_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper()
    monkeypatch.setattr(notifications_ops, "_notify_user", helper)

    result = await notifications_ops.notify_user("hi")

    assert result == "notification recorded to the internal sink"


async def test_notify_user_propagates_delivery_failure_as_502(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(raise_exc=ChannelDeliveryError("provider unreachable"))
    monkeypatch.setattr(notifications_ops, "_notify_user", helper)

    with pytest.raises(UpstreamError, match="provider unreachable"):
        await notifications_ops.notify_user("hello", channel="telegram")


async def test_notify_user_channel_cannot_notify_is_501(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(raise_exc=NotImplementedError("this channel cannot notify"))
    monkeypatch.setattr(notifications_ops, "_notify_user", helper)

    with pytest.raises(NotSupportedError, match="cannot notify"):
        await notifications_ops.notify_user("hello", channel="webhook")


# -- write-side isolation clamp: a restricted caller touches ONLY its own feed ----
# Exercised end-to-end through the operation door over the REAL channels helper +
# sink (no monkeypatched helper), so the clamp is proven on the operation surface.


async def test_restricted_notify_rejects_other_identity_as_403(sink_redis) -> None:
    # A restricted bob addressing alice is a cross-identity injection: the helper's
    # loud CrossIdentityAudienceError surfaces as a 403 authorization denial (the
    # write-side mirror of the read door's ForbiddenError, NOT a 400 bad request) and
    # NOTHING lands in alice's feed (or the shared feed) — the exfil/inject path is
    # closed at the write door.
    with _restricted("bob"), pytest.raises(ForbiddenError, match="may address only its own identity"):
        await notifications_ops.notify_user("hi alice", audience="alice")

    assert await notifications_sink.read_notifications(audience="alice") == []
    assert await notifications_sink.read_notifications() == []


async def test_restricted_notify_rejects_own_owner_as_403(sink_redis) -> None:
    # Under key-keyed isolation each key is its own island: the caller's OWN OWNER is
    # a FOREIGN write target, no longer a privileged one. A restricted bob (own id
    # bob, owner-claim alice) addressing its owner alice is rejected exactly like any
    # other cross-identity inject — a loud 403 — and NOTHING lands in alice's (the
    # owner's) feed or the shared feed. This pins that the owner lost its former write
    # privilege; an owner-privileged write model would let this through.
    with _restricted("bob", owner="alice"), pytest.raises(ForbiddenError, match="may address only its own identity"):
        await notifications_ops.notify_user("hi owner", audience="alice")

    assert await notifications_sink.read_notifications(audience="alice") == []
    assert await notifications_sink.read_notifications() == []


class _RecordingChannel:
    """Records every notification handed to ``notify`` — proves whether the channel
    path was reached. ``notify_user`` never calls ``deliver`` (that is the ``ask_user``
    surface), so its protocol stub asserts if ever reached."""

    def __init__(self) -> None:
        self.notifications: list[ChannelNotification] = []

    async def deliver(self, delivery: object) -> None:
        raise AssertionError("notify_user must never call deliver")

    async def notify(self, notification: ChannelNotification) -> None:
        self.notifications.append(notification)


async def test_restricted_notify_channel_path_rejects_other_identity_as_403(sink_redis) -> None:
    # The clamp fires on the CHANNEL path too, BEFORE the channel is resolved: a
    # restricted bob addressing alice over a channel is rejected as a 403 and NOTHING
    # leaks — the recording channel is never touched AND alice's feed stays empty. This
    # pins the clamp OUTSIDE the ``if channel is None`` branch: a sink-only clamp would
    # skip the channel path, deliver to the channel, and write alice's feed.
    app._channel_registry.reset()
    channel = _RecordingChannel()
    tai42_app.channels.register("fake", channel)
    try:
        with _restricted("bob"), pytest.raises(ForbiddenError, match="may address only its own identity"):
            await notifications_ops.notify_user("hi alice", channel="fake", audience="alice")
    finally:
        app._channel_registry.reset()

    assert channel.notifications == []
    assert await notifications_sink.read_notifications(audience="alice") == []
    assert await notifications_sink.read_notifications() == []


async def test_restricted_notify_channel_path_rejects_own_owner_as_403(sink_redis) -> None:
    # The owner is a FOREIGN target on the CHANNEL path too: a restricted bob (owner
    # alice) addressing its owner alice over a channel is rejected as a 403 BEFORE the
    # channel is resolved — the recording channel is never touched AND alice's feed
    # stays empty. Under key-keyed isolation the owner holds no channel-write privilege.
    app._channel_registry.reset()
    channel = _RecordingChannel()
    tai42_app.channels.register("fake", channel)
    try:
        with (
            _restricted("bob", owner="alice"),
            pytest.raises(ForbiddenError, match="may address only its own identity"),
        ):
            await notifications_ops.notify_user("hi owner", channel="fake", audience="alice")
    finally:
        app._channel_registry.reset()

    assert channel.notifications == []
    assert await notifications_sink.read_notifications(audience="alice") == []
    assert await notifications_sink.read_notifications() == []


async def test_restricted_notify_scopes_unset_audience_to_self(sink_redis) -> None:
    # An unset audience is scoped to the restricted caller's OWN feed (its own id),
    # never its owner's.
    with _restricted("bob"):
        await notifications_ops.notify_user("status")

    own = await notifications_sink.read_notifications(audience="bob")
    assert len(own) == 1
    assert own[0]["audience"] == "bob"
    # Nothing leaked into another identity's feed.
    assert await notifications_sink.read_notifications(audience="alice") == []


async def test_unrestricted_notify_may_address_any_identity(sink_redis) -> None:
    # Regression guard: an unrestricted caller (no bound owner claim) is NOT clamped
    # — it may address any identity, exactly as before.
    await notifications_ops.notify_user("hi alice", audience="alice")

    own = await notifications_sink.read_notifications(audience="alice")
    assert len(own) == 1
    assert own[0]["audience"] == "alice"


# -- list_notifications -------------------------------------------------------


async def test_list_notifications_returns_sink_records(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"id": "2", "message": "b"}, {"id": "1", "message": "a"}]

    async def _read(audience: str | None = None) -> list:
        return records

    monkeypatch.setattr(notifications_ops, "read_notifications", _read)
    assert await notifications_ops.list_notifications() == {"notifications": records}


# -- projection ---------------------------------------------------------------


def test_notify_user_projects_with_destructive_hint() -> None:
    # notify_user is destructive (an external side-effect) — off the default surface
    # but includable; when projected it carries the destructiveHint annotation.
    reg = OperationRegistry()
    reg.register(operation_metadata_of(notifications_ops.notify_user))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    class _App:
        def __init__(self) -> None:
            self.tools = _Rec()

    app = _App()
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert "notify_user" in names
    assert app.tools.registered["notify_user"]["annotations"].destructiveHint is True
