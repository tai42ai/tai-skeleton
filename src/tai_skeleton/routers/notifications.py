"""HTTP surface for the internal notifications sink and the send door.

- ``GET /api/notifications`` (AUTHED) — return the deployment's internal
  notifications, newest-first. Each record is
  ``{"id", "message", "recipient", "audience", "created_at"}``. A restricted caller
  reads its own per-identity feed (only its ``audience``-addressed records); an
  unrestricted caller reads the shared feed.
- ``POST /api/notifications`` (AUTHED) — send a human a one-way, fire-and-forget
  notification: on a named ``channel``, or (channel omitted) recorded to the
  internal sink. A blank message / unknown channel / blank recipient / blank
  audience is a loud 400, a channel that cannot notify a 501, and a delivery
  failure a 502.

Both doors are thin adapters over operations in
``tai_skeleton.operations.notifications``. Success bodies are ``{"data": ...}``;
failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.operations import operation_metadata_of, register_operation_route
from tai_skeleton.operations.notifications import list_notifications as _list_notifications_op
from tai_skeleton.operations.notifications import notify_user as _notify_user_op

list_notifications = register_operation_route(
    tai_app,
    operation_metadata_of(_list_notifications_op),
    path="/api/notifications",
    method="GET",
)

notify_user = register_operation_route(
    tai_app,
    operation_metadata_of(_notify_user_op),
    path="/api/notifications",
    method="POST",
)
