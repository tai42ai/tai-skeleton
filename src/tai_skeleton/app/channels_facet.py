"""The app's channels facet — the ``app.channels`` namespace, in its own module
like ``app.clients`` (``app/clients.py``).

Forwards to the app's :class:`~tai_skeleton.channels.registry.ChannelRegistry`.
A channel plugin registers a named deliverer here via an import-only
``channel_modules`` manifest entry; the ``ask_user`` helper resolves it by name
at ask time, and the channels catalog route lists the registered names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tai_contract.channels import Channel

if TYPE_CHECKING:
    from tai_skeleton.app.server import TaiMCP


class ChannelsFacet:
    """``app.channels`` — channel registration + lookup (``AppChannels``)."""

    __slots__ = ("_app",)

    def __init__(self, app: TaiMCP) -> None:
        self._app = app

    def register(self, name: str, channel: Channel) -> None:
        return self._app._channel_registry.register(name, channel)

    def get(self, name: str) -> Channel:
        return self._app._channel_registry.get(name)

    def names(self) -> list[str]:
        return self._app._channel_registry.names()
