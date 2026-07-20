"""Studio-plugin registry operation — list the installed plugins' manifest
contents from the startup-built registry.

The plugin asset-serving and SPA-hosting routes are content servers (they answer
file bytes / injected HTML with per-response security headers, not the
``{"data": ...}`` envelope), so they stay handlers in the router. Only the
registry listing is an enveloped-JSON read, so it lives here as an operation. A
registry that has not been built is a loud 500 (:class:`OperationFailed`) rather
than a fabricated empty list.
"""

from __future__ import annotations

from tai_skeleton.operations import OperationFailed, operation
from tai_skeleton.plugins.registry import StudioPluginError, current_registry


@operation(summary="List the registered studio plugins", tags=["plugins"], errors=[OperationFailed])
async def list_studio_plugins() -> list:
    try:
        registry = current_registry()
    except StudioPluginError as exc:
        raise OperationFailed(str(exc)) from exc
    return registry.manifest_contents()
