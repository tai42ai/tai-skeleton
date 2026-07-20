"""The process-wide channel registry — the body behind ``app.channels``.

A channel plugin registers under a name via an import-only ``channel_modules``
manifest entry (importing the module runs its ``tai42_app.channels.register(...)``
call). The ``ask_user`` helper resolves a named channel at ask time.

The registry is reset on every ``start()`` (like the webhook-verifier registry)
so a reload re-imports the channel modules and re-registers cleanly; a duplicate
name within one load raises loudly (a silent overwrite could swap the deliverer
out from under a live question).
"""

from __future__ import annotations

from tai42_contract.channels import Channel


class ChannelRegistry:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, name: str, channel: Channel) -> None:
        if name in self._channels:
            raise ValueError(f"channel {name!r} is already registered")
        self._channels[name] = channel

    def get(self, name: str) -> Channel:
        try:
            return self._channels[name]
        except KeyError:
            raise KeyError(f"unknown channel {name!r} (registered: {sorted(self._channels)})") from None

    def names(self) -> list[str]:
        return sorted(self._channels)

    def reset(self) -> None:
        self._channels.clear()
