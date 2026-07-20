"""Access control: ASGI auth gate (identity + policy + route guard).

The public surface is :class:`AuthAdapter` (builds the middleware stack and
verifies tokens) and the cached :func:`access_control_settings`. The identity,
policy and model contracts live in ``tai_contract.access_control``; the classes
here implement them.
"""

from tai_skeleton.access_control.adapter import AuthAdapter
from tai_skeleton.access_control.settings import (
    AccessControlSettings,
    access_control_settings,
)

__all__ = [
    "AccessControlSettings",
    "AuthAdapter",
    "access_control_settings",
]
