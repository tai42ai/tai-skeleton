"""Channels operations — the authed catalog read.

``list_channels`` returns the registered channel names (the delivery media
``ask_user(channel=...)`` can resolve). Registration is import-only (a manifest
``channel_modules`` entry); this operation is read-only.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.operations import operation


@operation(summary="List registered channels", tags=["channels"])
async def list_channels() -> dict:
    return {"channels": tai_app.channels.names()}
