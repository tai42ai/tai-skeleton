"""A manifest ``backend_module`` fixture: registers a launch-recording backend.

Importing this module registers :class:`LaunchRecordingBackend` through the
``tai_app`` handle at import time, exactly as a real backend plugin module does —
so it only works when the app is already bound when the import runs. Each
``launch`` call appends its args to the class-level ``launched`` list, letting a
test assert a non-worker (beat/flower-style) invocation reached the registered
backend.
"""

from __future__ import annotations

from typing import Any, ClassVar

from tai_contract.app import tai_app
from tai_contract.backend import Backend


@tai_app.backends.register_backend
class LaunchRecordingBackend(Backend):
    # Class-level so the test reads the record without holding the instance the
    # registration decorator constructed.
    launched: ClassVar[list[list[str]]] = []

    async def launch(self, args) -> None:
        type(self).launched.append(list(args))

    async def reload_mcp(self, title, targets=None) -> Any:
        return None

    async def deregister_mcp(self, title, targets=None) -> Any:
        return None

    async def reload_tool(self, kind, name, targets=None) -> Any:
        return None

    async def remove_tool(self, kind, name, targets=None) -> Any:
        return None

    async def reload_config(self, targets=None) -> Any:
        return None

    async def reload_failed_mcps(self, targets=None) -> Any:
        return None

    async def list_failed_mcps(self, targets=None) -> Any:
        return {}

    async def list_workers(self) -> list[str]:
        return []
