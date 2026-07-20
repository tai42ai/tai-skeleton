"""Channel-delivery surface: the channel registry behind ``app.channels``.

A channel is a registered deliverer that pushes an interaction question to a
human on a specific medium and bridges the reply back through the public
interactions callback door; the ``ask_user`` helper resolves a named channel
from the registry here at ask time. The package also hosts the ``notify_user``
helper (``notify.py``): one fire-and-forget message to a human on a named
channel, no reply expected.
"""

from __future__ import annotations

from tai_skeleton.channels.registry import ChannelRegistry

__all__ = ["ChannelRegistry"]
