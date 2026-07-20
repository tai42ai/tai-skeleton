"""Catalog cache refresh wiring on the MCP app.

A community provider add propagates fleet-wide through the backend reload
dispatch, so the catalog refresh must be registered for BOTH process startup
and the in-place re-init (``update()``) — startup-only registration would
leave running workers blind to new catalog rows until a restart.
"""

from __future__ import annotations

import pytest

# Trigger app setup before anything adapter-related — same preamble as
# test_adapter_dispatch / test_constant_alignment.
import tai_skeleton.app.instance as app_mod
from tai_skeleton.app.lifecycle import TaiMCPLifecycleMixin
from tai_skeleton.manifest import Manifest


def test_refresh_catalog_registered_for_startup_and_reload() -> None:
    handler = app_mod.refresh_catalog_if_connectors_in_use

    app = app_mod.app
    assert handler in app._startup_handlers.values()
    reload_key = f"{handler.__module__}.{handler.__qualname__}"
    assert app._reload_handlers.get(reload_key) is handler


class _Mixin(TaiMCPLifecycleMixin):
    """Concrete-enough subclass to exercise update()'s handler re-run without
    an event server, network, or the full app: start() is stubbed to skip the
    re-init and there are no live tools to drop."""

    def __init__(self):
        super().__init__()
        self.started_with = None

    def _mcp_tools(self, config, tools):  # abstract in the mixin
        pass

    def start(self, manifest):
        self.started_with = manifest


def test_update_reruns_reload_handlers() -> None:
    mixin = _Mixin()
    ran = []

    @mixin._on_reload
    async def _reload_marker() -> None:
        ran.append("reload")

    manifest = Manifest()
    mixin._update(manifest)

    assert mixin.started_with is manifest
    assert ran == ["reload"]


def test_update_raises_when_reload_handler_fails() -> None:
    # raise_on_error on the update path: a failed refresh must fail the op
    # loudly, never leave the worker silently behind the catalog.
    mixin = _Mixin()

    @mixin._on_reload
    async def _boom() -> None:
        raise RuntimeError("refresh blew up")

    with pytest.raises(RuntimeError, match="lifecycle handlers failed"):
        mixin._update(Manifest())
