"""Notifications operations — read the internal inbox and send a notification.

* ``list_notifications`` returns the deployment's internal notifications feed —
  channel-less sends plus any audience-addressed notification (recorded even when
  a channel also delivers it) — newest-first.
* ``notify_user`` sends a human ONE fire-and-forget message — on a named channel,
  or (channel omitted) recorded to the internal notifications sink the Studio inbox
  reads. No reply, no blocking wait. It delivers over the channels feature's
  ``notify_user`` helper, mapping the helper's loud
  failures to the operation's typed errors: a blank message / unknown channel /
  blank recipient/audience is a :class:`BadRequestError` (400), a restricted caller
  addressing another identity is a cross-identity authorization denial mapped to a
  :class:`ForbiddenError` (403) — the same 403 the read-side answer door raises for the
  symmetric read denial — a channel that cannot notify
  is a :class:`NotSupportedError` (501), and a channel delivery failure is an
  :class:`UpstreamError` (502) — a failure is never swallowed.

``notify_user`` causes an external side-effect (a message leaves the deployment),
so it is ``destructive=True``. It is a messaging door, not a privilege-shaping one,
so it is NOT authority-changing and stays a plain (includable) projected tool.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from tai_contract.channels import ChannelDeliveryError

from tai_skeleton.access_control.user import CrossIdentityAudienceError, request_identity
from tai_skeleton.channels.notifications_sink import read_notifications
from tai_skeleton.channels.notify import notify_user as _notify_user
from tai_skeleton.operations import BadRequestError, ForbiddenError, NotSupportedError, UpstreamError, operation


class NotifyUser(BaseModel):
    """A notification to send: the ``message`` text, an optional named ``channel``
    that carries it (omit to record to the internal sink), an optional per-call
    ``recipient`` delivery address, and an optional ``audience`` identity whose in-app
    inbox shows it (honored even with a channel set; distinct from ``recipient``)."""

    message: str
    channel: str | None = None
    recipient: str | None = None
    audience: str | None = Field(
        default=None,
        description=(
            "The identity (user_id) whose in-app inbox shows this (honored even with a channel set); "
            "leave unset for an operator/broadcast notification. Distinct from recipient, which is a "
            "channel delivery address."
        ),
    )


@operation(summary="List internal notifications", tags=["notifications"])
async def list_notifications() -> dict:
    """List internal notifications, newest-first.

    A RESTRICTED caller reads its OWN per-identity feed (complete within its own
    bound — never truncated by other identities' volume, never a broadcast); an
    UNRESTRICTED caller reads the shared feed unchanged (today's operator view)."""
    _user_id, restricted = request_identity()
    return {"notifications": await read_notifications(audience=restricted)}


@operation(
    summary="Send a human a one-way notification",
    tags=["notifications"],
    destructive=True,
    errors=[BadRequestError, ForbiddenError, NotSupportedError, UpstreamError],
    request_model=NotifyUser,
)
async def notify_user(
    message: str, channel: str | None = None, recipient: str | None = None, audience: str | None = None
) -> str:
    """Send a human a one-way notification, fire-and-forget.

    No reply is expected and nothing blocks. With a named channel the message is
    sent and the call returns as soon as the medium ACCEPTED it (not that a human
    saw it). With ``channel`` omitted the message is recorded to the internal
    notifications sink the Studio inbox reads. One send attempt, no retry; every
    failure raises loudly, never a silent no-op:

    * a blank message, an unknown channel name, or a blank recipient/audience → 400;
    * a restricted caller addressing another identity (cross-identity denial) → 403;
    * a channel that cannot notify → 501;
    * a channel delivery failure → 502.

    ``audience`` addresses the in-app record to an identity's feed; it is honored
    even when a channel also delivers the message (channel push AND in-app record).

    Returns a short confirmation string — ``"notification sent via '<channel>'"``
    for a channel send, ``"notification recorded to the internal sink"`` otherwise.
    """
    try:
        await _notify_user(message, channel=channel, recipient=recipient, audience=audience)
    except CrossIdentityAudienceError as exc:
        # A restricted caller addressing another identity is a cross-identity boundary
        # violation — an AUTHORIZATION denial (403), the write-side mirror of the read
        # door's ForbiddenError, NOT a bad request. Genuine input-validation errors stay
        # the ValueError→400 below.
        raise ForbiddenError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    except NotImplementedError as exc:
        raise NotSupportedError(str(exc)) from exc
    except ChannelDeliveryError as exc:
        raise UpstreamError(str(exc)) from exc
    if channel is None:
        return "notification recorded to the internal sink"
    return f"notification sent via '{channel}'"
