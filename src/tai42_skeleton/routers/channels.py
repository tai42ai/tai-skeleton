"""HTTP surface for the channels feature — the authed catalog door the Studio
admin surface consumes.

- ``GET /api/channels`` (AUTHED) — list the registered channel names, i.e. the
  delivery media ``ask_user(channel=...)`` can currently resolve. Registration
  itself is import-only (a manifest ``channel_modules`` entry); this door is
  read-only. Success bodies are ``{"data": ...}``.

The route is a thin adapter over the :func:`list_channels` operation; the
operation logic lives in ``tai42_skeleton.operations.channels``.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.channels import list_channels as _list_channels_op

list_channels = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_channels_op),
    path="/api/channels",
    method="GET",
)
