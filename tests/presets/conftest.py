"""Preset-engine test wiring.

The router/extension test dirs bind the process app singleton at collection so the
``@tai_app`` decorators (routes, tools, extensions) that fire at module import land
on a live app; the preset-engine tests import ``_fixtures`` (which decorates tools
+ extensions) the same way, so bind here too.

``pg`` reuses the versioned-store suite's stateful fake Postgres, monkeypatched
over the pooled ``client_ctx`` so the REAL ``PostgresVersionedStore`` +
``PresetStoreView`` run against an in-memory pair of tables — the engine exercises
the true store path with no live database.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from tai_contract.app import tai_app
from tai_kit.clients.impl.postgres import PostgresClient

import tai_skeleton.versioning.store as store_module
from tai_skeleton.app import instance
from tests.versioning.conftest import FakeVersioningPg

tai_app.bind(instance.build_app())


@pytest.fixture
def pg(monkeypatch) -> FakeVersioningPg:
    fake = FakeVersioningPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(store_module, "client_ctx", fake_client_ctx)
    return fake


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    """Tear down every runtime-registered preset after each test.

    The process app is a singleton whose FastMCP server AND ``PresetManager``
    (spec map + quarantine set) outlive a single ``app_context``, so a preset a
    test binds would otherwise leak into the next one (a stale registration,
    a false name collision). Clean it up so each engine test starts from a bare
    tool registry — the store is already isolated per test by the ``pg`` fake."""
    yield
    app = instance.build_app()
    manager = app.preset_manager

    # The singleton FastMCP server + ``PresetManager`` outlive one ``app_context``,
    # so a base tool (weather/echo) bound by this test's manifest or a leaked preset
    # would collide with the next test's bind under ``on_duplicate="error"``. Clear
    # every remaining preset and tool.
    async def _clear() -> None:
        for name in list(manager.registered_names()):
            await manager.remove(name)
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    for name in list(manager.quarantined_names()):
        manager.drop_quarantine(name)
