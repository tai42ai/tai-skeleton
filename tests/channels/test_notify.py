"""The ``notify_user`` helper: with a named channel, one fire-and-forget send
through that channel (the notification reaches its ``notify`` verbatim); with no
channel, the message is recorded to the internal notifications sink. Every
failure propagates loudly (``ChannelDeliveryError``, and the protocol's
default-body ``NotImplementedError`` from a channel that cannot notify), and a
blank message, an unknown channel name, or a blank recipient is rejected before
any send.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from tai_contract.app import tai_app
from tai_contract.channels import ChannelDeliveryError, ChannelNotification

from tai_skeleton.app.instance import app
from tai_skeleton.channels import notifications_sink
from tai_skeleton.channels.notify import notify_user


class RecordingChannel:
    """Records every notification handed to ``notify``."""

    def __init__(self) -> None:
        self.notifications: list[ChannelNotification] = []

    async def notify(self, notification: ChannelNotification) -> None:
        self.notifications.append(notification)


class FailingChannel:
    async def notify(self, notification: ChannelNotification) -> None:
        raise ChannelDeliveryError("provider unreachable")


class CannotNotifyChannel:
    """A channel that cannot notify — its ``notify`` raises exactly what the
    contract protocol's default body raises."""

    async def notify(self, notification: ChannelNotification) -> None:
        raise NotImplementedError


@pytest.fixture
def register_channel():
    """Yield a registrar that installs a channel under a name; the registry is
    reset around every test."""
    app._channel_registry.reset()

    def _register(name, channel):
        tai_app.channels.register(name, channel)
        return channel

    yield _register
    app._channel_registry.reset()


# -- the send -------------------------------------------------------------------


async def test_notify_sends_through_the_named_channel(register_channel):
    channel = register_channel("fake", RecordingChannel())

    await notify_user("Deploy finished", channel="fake")

    assert channel.notifications == [ChannelNotification(message="Deploy finished", recipient=None)]


async def test_notify_forwards_recipient_verbatim(register_channel):
    channel = register_channel("fake", RecordingChannel())

    await notify_user("Ping", channel="fake", recipient="123456")

    assert channel.notifications == [ChannelNotification(message="Ping", recipient="123456")]


# -- loud failures: every error propagates, nothing is swallowed -----------------


async def test_delivery_failure_propagates(register_channel):
    register_channel("boom", FailingChannel())

    with pytest.raises(ChannelDeliveryError, match="provider unreachable"):
        await notify_user("hello", channel="boom")


async def test_cannot_notify_channel_surfaces_not_implemented(register_channel):
    # The contract protocol's default ``notify`` body raises NotImplementedError;
    # a channel that cannot notify surfaces exactly that — present and loud,
    # never a silent no-op.
    register_channel("mute", CannotNotifyChannel())

    with pytest.raises(NotImplementedError):
        await notify_user("hello", channel="mute")


# -- loud validation: rejected before any send -----------------------------------


@pytest.fixture
def sink_redis(monkeypatch, fake_redis):
    """Point the internal sink's Redis at the shared fake so ``channel=None``
    writes land somewhere readable back in the test."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake_redis

    monkeypatch.setattr(notifications_sink, "client_ctx", _ctx)
    return fake_redis


async def test_channel_none_records_to_sink(register_channel, sink_redis):
    channel = register_channel("fake", RecordingChannel())

    await notify_user("build passed")

    # It routes to the internal sink, not to any registered channel.
    assert channel.notifications == []
    records = await notifications_sink.read_notifications()
    assert len(records) == 1
    assert records[0]["message"] == "build passed"
    assert records[0]["recipient"] is None
    assert records[0]["id"]
    assert records[0]["created_at"]


async def test_channel_none_stores_recipient(sink_redis):
    await notify_user("ping", recipient="ops")

    records = await notifications_sink.read_notifications()
    assert records[0]["recipient"] == "ops"


@pytest.mark.parametrize("bad_recipient", ["", "   "])
async def test_channel_none_blank_recipient_rejected(sink_redis, bad_recipient):
    with pytest.raises(ValueError, match="recipient must be a non-empty address"):
        await notify_user("hello", recipient=bad_recipient)
    assert await notifications_sink.read_notifications() == []


# -- audience: the in-app identity axis, honored even on the channel path --------


async def test_channel_with_audience_records_in_app_and_sends(register_channel, sink_redis):
    # The two axes coexist: the channel delivers to ``recipient`` (an address) AND
    # the record lands in ``audience``'s in-app feed (an identity).
    channel = register_channel("fake", RecordingChannel())

    await notify_user("shipped", channel="fake", recipient="123", audience="alice")

    assert channel.notifications == [ChannelNotification(message="shipped", recipient="123")]
    own = await notifications_sink.read_notifications(audience="alice")
    assert len(own) == 1
    assert own[0]["message"] == "shipped"
    assert own[0]["audience"] == "alice"
    assert own[0]["recipient"] == "123"


async def test_channel_without_audience_stores_nothing(register_channel, sink_redis):
    # A plain channel send with no audience records nothing — today's behavior.
    register_channel("fake", RecordingChannel())

    await notify_user("plain", channel="fake")

    assert await notifications_sink.read_notifications() == []


async def test_channel_none_with_audience_writes_the_per_identity_feed(sink_redis):
    await notify_user("hi", audience="alice")

    own = await notifications_sink.read_notifications(audience="alice")
    assert len(own) == 1
    assert own[0]["audience"] == "alice"
    # It is also on the shared feed (operators still see it).
    assert len(await notifications_sink.read_notifications()) == 1


@pytest.mark.parametrize("bad_audience", ["", "   "])
async def test_blank_audience_rejected(sink_redis, bad_audience):
    with pytest.raises(ValueError, match="audience must be a non-empty identity"):
        await notify_user("hello", audience=bad_audience)
    assert await notifications_sink.read_notifications() == []


async def test_unknown_channel_rejected(register_channel):
    with pytest.raises(ValueError, match="unknown channel: 'nope'"):
        await notify_user("hello", channel="nope")


async def test_empty_channel_name_rejected(register_channel):
    with pytest.raises(ValueError, match="channel must be a non-empty string"):
        await notify_user("hello", channel="")


@pytest.mark.parametrize("bad_message", ["", "   ", "\n\t"])
async def test_blank_message_rejected(register_channel, bad_message):
    channel = register_channel("fake", RecordingChannel())

    with pytest.raises(ValueError, match="message must be a non-blank string"):
        await notify_user(bad_message, channel="fake")
    assert channel.notifications == []


async def test_non_string_message_rejected(register_channel):
    channel = register_channel("fake", RecordingChannel())

    with pytest.raises(ValueError, match="message must be a non-blank string"):
        await notify_user(42, channel="fake")  # type: ignore[arg-type]
    assert channel.notifications == []
