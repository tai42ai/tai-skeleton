"""The author-facing ``notify_user`` surface — one fire-and-forget message to a
human, no reply expected.

Unlike ``ask_user`` there is no interaction, no ticket, no callback and no
blocking wait: the named channel sends the message and the call returns as soon
as the medium accepts it ("sent" means accepted by the medium, not seen by a
human). One send attempt, no retry; every failure raises loudly
(``ChannelDeliveryError`` from the channel, ``NotImplementedError`` from a
channel that cannot notify).

``channel`` is optional: with a named channel the message is delivered on that
medium; with ``channel=None`` it is recorded to the internal notifications sink
(the Studio inbox reads it back), so nothing is ever silently dropped.

``audience`` is the IDENTITY (a user_id) whose in-app inbox shows the
message — distinct from ``recipient`` (a channel delivery address). It is honored
even when a channel is set: an ``audience``-addressed call records the in-app entry
(shared + per-identity feed) REGARDLESS of whether a channel also delivers it, so
``notify_user(channel="sms", recipient=…, audience=A)`` both pushes to SMS AND lands
in A's in-app feed (matching ``ask_user``, which always persists). An UNRESTRICTED
caller's channel send with no audience stores nothing; a RESTRICTED caller's audience
is clamped to its OWN identity, so its channel send ALWAYS records to its own feed too
(in addition to the channel push) — a channel send storing nothing is the unrestricted-caller
case only.
"""

from __future__ import annotations

from tai_contract.app import tai_app
from tai_contract.channels import Channel, ChannelNotification

from tai_skeleton.access_control.user import clamp_write_audience
from tai_skeleton.channels.notifications_sink import record_notification


def _resolve_channel(channel: str) -> Channel:
    """Resolve a named channel loudly — an unknown name raises ``ValueError``
    (mirroring the ``ask_user`` helper's channel guard), never a soft ignore."""
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a non-empty string")
    try:
        return tai_app.channels.get(channel)
    except KeyError as exc:
        raise ValueError(f"unknown channel: {channel!r}") from exc


async def notify_user(
    message: str, *, channel: str | None = None, recipient: str | None = None, audience: str | None = None
) -> None:
    """Notify a human of ``message``, fire-and-forget.

    With a named ``channel`` the message is sent on that medium; a plain return
    means the medium ACCEPTED it — not that a human saw it. One send attempt, no
    retry, no reply.

    With ``channel=None`` the message is recorded to the internal notifications
    sink (Redis), which the Studio inbox reads back; the interactions Redis must
    be reachable or the write raises loudly (never a silent no-op).

    ``recipient`` is an OPTIONAL per-call address (chat id, phone number, ...).
    On a named channel it is carried to the channel, which validates it against
    its operator allowlist; omitted, the channel sends to its operator-configured
    default recipient. With ``channel=None`` it is stored verbatim on the sink
    record. A set value must be a non-blank string.

    ``audience`` is the IDENTITY (a user_id) whose in-app inbox shows this,
    distinct from ``recipient``. When set, the in-app record is written (shared +
    per-identity feed) REGARDLESS of whether a channel also delivers the message —
    so an addressed notification lands in the identity's feed even on the channel
    path. A blank value is rejected.

    Raises ``ValueError`` for a blank message, an unknown channel name, or a
    blank ``recipient``/``audience``; ``CrossIdentityAudienceError`` when a
    restricted caller addresses another identity (a cross-identity authorization
    denial the operation door maps to a 403); ``ChannelDeliveryError`` when a
    channel send fails; ``NotImplementedError`` when a channel cannot notify.
    Every failure propagates loudly — nothing is swallowed.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-blank string")
    if audience is not None and (not isinstance(audience, str) or not audience.strip()):
        raise ValueError("audience must be a non-empty identity")
    # Write-side isolation clamp — before any record is written. A restricted caller
    # may address only its own feed: an unset audience is scoped to its own identity,
    # and any other identity is rejected loudly (cross-identity injection). An unrestricted
    # caller is unchanged. The cross-identity rejection is an authorization denial the
    # operation door maps to a 403 (the write-side mirror of the read door).
    audience = clamp_write_audience(audience)
    if channel is None:
        # Internal sink path: the sink stores the address verbatim, so a set
        # value is guarded here (a channel, by contrast, owns its own recipient
        # validation — the delivery path below is left untouched).
        if recipient is not None and (not isinstance(recipient, str) or not recipient.strip()):
            raise ValueError("recipient must be a non-empty address")
        await record_notification(message, recipient=recipient, audience=audience)
        return
    channel_obj = _resolve_channel(channel)
    # An addressed notification lands in the identity's in-app feed even on the
    # channel path, matching ``ask_user`` (which always persists). After the clamp a
    # restricted caller always has an audience here (scoped to its own identity), so
    # its channel send records to its own feed too; only an unrestricted caller's send
    # with no audience stores nothing.
    if audience is not None:
        await record_notification(message, recipient=recipient, audience=audience)
    await channel_obj.notify(ChannelNotification(message=message, recipient=recipient))
