"""Startup/reload rehydration through the real lifecycle hook.

Where ``test_manager`` drives ``PresetManager.rehydrate()`` directly, these cases
exercise the wired ``@on_startup`` handler
(:func:`instance.rehydrate_versioned_presets_if_store_in_use`) end-to-end: a
persisted versioned preset is re-registered as the app BOOTS, and a stale preset
is QUARANTINED without aborting the boot (which runs under ``raise_on_error=True``).

The store is configured by setting ``VERSIONING_STORE_PG_PASSWORD`` (the gate that
lets the handler open Postgres), and the ``pg`` fixture points that open at the
in-memory fake — so the boot path runs the true store + engine offline.
"""

from __future__ import annotations

import asyncio

import pytest
from tai_contract.presets import PresetBody

from tai_skeleton.app.instance import app
from tai_skeleton.manifest import Manifest
from tests.versioning.conftest import FakeVersioningPg

_MANIFEST = {
    "extensions_modules": ["tests.presets._ext_fixtures"],
    "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
}


def _manifest() -> Manifest:
    return Manifest.model_validate(_MANIFEST)


@pytest.fixture
def store_configured(monkeypatch) -> None:
    """Set the store's DSN env so the boot hook's gate opens Postgres."""
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")


async def _seed(name: str, base_tool: str) -> None:
    """Persist a versioned preset directly through the generic store, so it is
    present in the store BEFORE the app boots (the create route's view guard is
    bypassed to also model a name that only became a foreign tool after persist)."""
    body = PresetBody(base_tool=base_tool, description="d", fixed_kwargs={"units": "v"}, extensions=[], tags=[])
    await app.versioning.store.create("preset", name, body.model_dump())


def test_startup_hook_reregisters_versioned_preset(pg: FakeVersioningPg, store_configured):
    async def run():
        await _seed("ver", "weather")
        # Boot: the on_startup handler rehydrates the persisted preset.
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            assert mgr.is_registered("ver")
            assert not mgr.is_quarantined("ver")
            assert await app.tools.run_tool("ver", {"city": "x"}) == {"city": "x", "units": "v"}

    asyncio.run(run())


def test_startup_hook_quarantines_foreign_name_without_bricking_boot(pg: FakeVersioningPg, store_configured):
    async def run():
        # A persisted preset whose NAME is occupied by the live base tool "echo".
        body = PresetBody(base_tool="weather", description="d", fixed_kwargs={}, extensions=[], tags=[])
        await app.versioning.store.create("preset", "echo", body.model_dump())

        # Boot succeeds (no raise) even though this preset can't register.
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            assert mgr.is_quarantined("echo")
            assert not mgr.is_registered("echo")
            # The foreign tool that owns the name is untouched and still runnable.
            assert await app.tools.run_tool("echo", {"text": "hi"}) == "hi"

    asyncio.run(run())


def test_startup_hook_quarantines_missing_base_tool_without_bricking_boot(pg: FakeVersioningPg, store_configured):
    async def run():
        await _seed("orphan", "gone_tool")
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            assert mgr.is_quarantined("orphan")
            assert not mgr.is_registered("orphan")
            assert "orphan" not in await app.tools.get_tools()

    asyncio.run(run())


def test_startup_hook_skipped_when_store_unconfigured(pg: FakeVersioningPg):
    async def run():
        # No VERSIONING_STORE_* env: the boot hook must not touch the store, even
        # though a row exists — so the preset is NOT rehydrated (and no pg read).
        await _seed("ver", "weather")
        pg.executed.clear()
        async with app.app_context(_manifest()):
            assert not app.preset_manager.is_registered("ver")
        assert pg.executed == []  # the gate skipped the Postgres open entirely

    asyncio.run(run())


def test_startup_hook_preset_owned_base_quarantined_either_order(pg: FakeVersioningPg, store_configured):
    async def run():
        # "chained" precedes "legit" by name (list_presets orders by name), and its
        # base_tool is a preset — quarantined regardless of load order.
        await _seed("legit", "weather")
        chained = PresetBody(base_tool="legit", description="d", fixed_kwargs={}, extensions=[], tags=[])
        await app.versioning.store.create("preset", "chained", chained.model_dump())

        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            assert mgr.is_registered("legit")  # the legitimate preset rebuilt
            assert mgr.is_quarantined("chained")  # the preset-on-preset rejected
            assert not mgr.is_registered("chained")

    asyncio.run(run())
