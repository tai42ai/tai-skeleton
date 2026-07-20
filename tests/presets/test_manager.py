"""The preset register/reload engine (:class:`PresetManager`).

Every case drives the REAL bind path — ``app.app_context`` with a live
``ToolRegistry``, the true ``PostgresVersionedStore`` + ``PresetStoreView`` over
the stateful fake Postgres (the ``pg`` fixture) — so the engine is exercised
end-to-end: a preset becomes a runnable MCP tool, its baked kwargs are served and
a baked key rejected, extension combos branch off the bare name, versioning
carries the whole body forward, reload/rollback re-serve the right kwargs, a
failed re-register never drops the live tool, and a stale preset is quarantined
rather than bricking boot.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastmcp.tools.base import Tool
from tai_contract.agent.base import PresetSpec
from tai_contract.presets import PresetBody
from tai_contract.presets.errors import PresetNameConflictError, PresetNotFoundError

from tai_skeleton.app.instance import app
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.manifest import Manifest
from tests.versioning.conftest import FakeVersioningPg

_MANIFEST = {
    "extensions_modules": ["tests.presets._ext_fixtures"],
    "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
}


def _manifest() -> Manifest:
    return Manifest.model_validate(_MANIFEST)


def _live_tool_names() -> list[str]:
    """Every tool name held by the live FastMCP provider, WITH duplicates — so a
    reload that leaked a second copy of a branch shows up as a repeated name (a
    plain ``get_tools()`` dict would collapse it)."""
    components = app.fastmcp.local_provider._components
    return [c.name for c in components.values() if isinstance(c, Tool)]


async def _create_versioned(name: str, base_tool: str, fixed_kwargs, extensions, tags, description="d") -> None:
    """Persist a versioned preset AND register it — the create route's two steps."""
    await app.presets.store.create_preset(
        PresetSpec(name=name, description=description, base_tool=base_tool, fixed_kwargs=fixed_kwargs),
        extensions=extensions,
        tags=tags,
    )
    body = await app.presets.store.get_active_body(name)
    await app.preset_manager.register(
        name, body.base_tool, body.fixed_kwargs, body.extensions, body.tags, body.description
    )


# -- runnable + baked kwargs -------------------------------------------------


def test_register_binds_runnable_tool_and_rejects_baked_key(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await mgr.register("paris", "weather", {"units": "imperial"}, [], [], "Paris weather")

            assert "paris" in await app.tools.get_tools()
            # The baked value is served as a fixed constant...
            assert await app.tools.run_tool("paris", {"city": "paris"}) == {"city": "paris", "units": "imperial"}
            # ...and a caller that passes the baked key is REJECTED, never overriding it.
            with pytest.raises(TypeError):
                await app.tools.run_tool("paris", {"city": "paris", "units": "metric"})
            # ``register`` binds the live tool only — it never writes the store.
            assert pg.documents == []

    asyncio.run(run())


def test_versioned_preset_runnable_and_typed_schema(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            await _create_versioned("wv", "weather", {"units": "imperial"}, [], [])
            assert await app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "imperial"}

            # The baked key is HIDDEN from the exposed schema; the remaining arg
            # keeps its real typed schema (name + type), not one opaque blob.
            tool = await app.tools.get_tool("wv")
            props = tool.parameters.get("properties", {})
            assert "units" not in props
            assert props["city"]["type"] == "string"

    asyncio.run(run())


# -- spec map is authoritative + in lockstep ---------------------------------


def test_spec_map_serves_active_baked_kwargs(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await mgr.register("eph", "weather", {"units": "eph"}, [], ["t"], "d")
            await _create_versioned("ver", "weather", {"units": "v1"}, [], ["cat"])

            # The spec map is the source of truth for baked kwargs.
            assert mgr.baked_kwargs("eph") == {"units": "eph"}
            assert mgr.baked_kwargs("ver") == {"units": "v1"}
            assert mgr.get_spec("ver").tags == ["cat"]
            assert set(mgr.registered_names()) == {"eph", "ver"}

            # The map stays in lockstep with the active version after an edit.
            await app.presets.store.save_version("ver", fixed_kwargs={"units": "v2"})
            await mgr.reload("ver")
            assert mgr.baked_kwargs("ver") == {"units": "v2"}

            with pytest.raises(PresetNotFoundError):
                mgr.get_spec("nope")

    asyncio.run(run())


# -- reload / rollback serve the right kwargs --------------------------------


def test_reload_and_rollback_serve_right_kwargs(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            store = app.presets.store
            mgr = app.preset_manager
            await _create_versioned("wv", "weather", {"units": "v1"}, [], [])

            await store.save_version("wv", fixed_kwargs={"units": "v2"})
            await mgr.reload("wv")
            assert await app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v2"}

            await store.rollback("wv", 1)
            await mgr.reload("wv")
            assert await app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v1"}

    asyncio.run(run())


def test_save_version_numbering_is_max_plus_one_post_rollback(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            store = app.presets.store
            mgr = app.preset_manager
            await _create_versioned("wv", "weather", {"units": "v1"}, [], [])
            for units in ("v2", "v3", "v4", "v5"):
                await store.save_version("wv", fixed_kwargs={"units": units})
            await store.rollback("wv", 2)  # active trails MAX
            new = await store.save_version("wv", fixed_kwargs={"units": "v6"})
            assert new.version == 6  # MAX+1, not active(2)+1
            await mgr.reload("wv")
            assert await app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v6"}

    asyncio.run(run())


# -- a wrapper branch of a preset keeps the preset's description -------------


def test_wrapper_branch_of_preset_keeps_preset_description(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            # ``exta`` is a ``functools.wraps`` wrapper, so its branch inherits the
            # base callable's docstring. The adoption guard must compare against the
            # BRANCH BASE callable's docstring (not the ``Tool`` object's class
            # docstring), so the wrapper is recognized as authoring no new
            # description and the preset's own description survives onto the branch.
            await _create_versioned("shouty", "echo", {}, [["exta"]], [], description="Preset desc")

            base = await app.tools.get_tool("shouty")
            branch = await app.tools.get_tool("shouty_exta")
            assert base.description == "Preset desc"
            assert branch.description == "Preset desc"

    asyncio.run(run())


# -- extensions survive versioning (two independent combos) ------------------


def test_extensions_two_combos_survive_versioning(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            store = app.presets.store
            mgr = app.preset_manager
            await _create_versioned("shouty", "echo", {}, [["exta"], ["extb"]], [])

            # Bare name runnable + BOTH independent branches + NO stacked branch.
            tools = set(await app.tools.get_tools())
            assert {"shouty", "shouty_exta", "shouty_extb"} <= tools
            assert "shouty_exta_extb" not in tools
            assert await app.tools.run_tool("shouty", {"text": "hi"}) == "hi"
            assert await app.tools.run_tool("shouty_exta", {"text": "hi"}) == "hi|a"
            assert await app.tools.run_tool("shouty_extb", {"text": "hi"}) == "hi|b"

            # Save a new version WITHOUT passing extensions, then reload: the new
            # active body carried base_tool + BOTH combos forward.
            await store.save_version("shouty", fixed_kwargs={})
            await mgr.reload("shouty")
            tools = set(await app.tools.get_tools())
            assert {"shouty", "shouty_exta", "shouty_extb"} <= tools
            assert "shouty_exta_extb" not in tools
            assert await app.tools.run_tool("shouty_exta", {"text": "yo"}) == "yo|a"

            # Each branch is bound EXACTLY once (the reload's teardown ran before
            # re-register — no leaked pre-reload duplicate), and the base's
            # _extend_tools holds exactly one entry per branch.
            names = _live_tool_names()
            assert names.count("shouty_exta") == 1
            assert names.count("shouty_extb") == 1
            branches = {b for b, base in app._tool_registry._extend_tools.items() if base == "shouty" and b != "shouty"}
            assert branches == {"shouty_exta", "shouty_extb"}

    asyncio.run(run())


# -- remove tears down base + branches ---------------------------------------


def test_remove_tears_down_base_and_branches(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await _create_versioned("shouty", "echo", {}, [["exta"], ["extb"]], [])
            assert {"shouty", "shouty_exta", "shouty_extb"} <= set(await app.tools.get_tools())

            await mgr.remove("shouty")

            tools = set(await app.tools.get_tools())
            assert not ({"shouty", "shouty_exta", "shouty_extb"} & tools)
            assert not mgr.is_registered("shouty")
            for name in ("shouty", "shouty_exta", "shouty_extb"):
                with pytest.raises(RuntimeError):
                    await app.tools.run_tool(name, {"text": "hi"})

    asyncio.run(run())


# -- categorization tags -> native FastMCP tags (re-projected on reload) -----


def test_body_tags_project_to_native_tags_and_survive_reload(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await _create_versioned("wv", "weather", {"units": "v1"}, [], ["geo", "eu"])
            assert (await app.tools.get_tool("wv")).tags == {"geo", "eu"}

            # A simulated reload wipes native tags; rehydrate re-projects them
            # from the persisted body.
            await mgr.remove("wv")
            await mgr.rehydrate()
            assert (await app.tools.get_tool("wv")).tags == {"geo", "eu"}

    asyncio.run(run())


# -- name-collision guard raises before any store write ----------------------


def test_name_conflict_raises_before_store_write(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            # "weather" is a live non-preset base tool.
            assert await app.preset_manager.name_conflicts("weather") is True
            with pytest.raises(PresetNameConflictError):
                await app.presets.store.create_preset(
                    PresetSpec(name="weather", base_tool="echo", fixed_kwargs={}),
                    extensions=[],
                    tags=[],
                )
            assert pg.documents == []  # nothing persisted

    asyncio.run(run())


def test_register_rejects_invalid_name(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            # A preset name is a live tool name + a route segment, so the manager
            # rejects a name outside the tool-name-safe alphabet/length before any
            # bind (the create route's 400 guard shares this rule).
            for bad in ("a/b", "x" * 65, "bad name"):
                with pytest.raises(ValueError, match="invalid preset name"):
                    await app.preset_manager.register(bad, "echo", {}, [], [], "d")

    asyncio.run(run())


def test_register_failure_leaves_no_partial_registration(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # An unknown extension makes the branch bind raise inside register.
            with pytest.raises(TaiValidationError):
                await mgr.register("bad", "echo", {}, [["ghost_ext"]], [], "d")
            tools = set(await app.tools.get_tools())
            assert "bad" not in tools
            assert not mgr.is_registered("bad")
            # The structured-registry seed was rolled back too.
            assert list(app._tool_registry.tool_extensions_iterator("bad")) == []

    asyncio.run(run())


# -- edit-path re-register is atomic (never drops the live tool) -------------


def test_edit_path_reload_failure_restores_old_registration(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            store = app.presets.store
            mgr = app.preset_manager
            await _create_versioned("wv", "echo", {}, [["exta"]], [])
            assert await app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"

            # A new version whose extensions reference an unknown ext will fail the
            # reload's re-register; the committed store bump is NOT unwound.
            await store.save_version("wv", extensions=[["ghost_ext"]])
            with pytest.raises(TaiValidationError):
                await mgr.reload("wv")

            # The PRIOR registration survived: base AND its branch still runnable,
            # and the spec map still holds the old body (the restore ran).
            assert await app.tools.run_tool("wv", {"text": "hi"}) == "hi"
            assert await app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            assert mgr.get_spec("wv").extensions == [["exta"]]

    asyncio.run(run())


# -- rehydrate durability (store-backed presets survive) ---------------------


def test_rehydrate_reregisters_only_store_backed_presets(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # A registration with no store row (bound directly, never persisted)
            # alongside a persisted one.
            await mgr.register("unbacked", "weather", {"units": "e"}, [], [], "d")
            await _create_versioned("ver", "weather", {"units": "v"}, [], [])

            # Simulate reload_config wiping every runtime preset, then rehydrate.
            for name in ("unbacked", "ver"):
                await mgr.remove(name)
            await mgr.rehydrate()

            tools = set(await app.tools.get_tools())
            assert "ver" in tools
            assert "unbacked" not in tools
            assert set(mgr.registered_names()) == {"ver"}  # only the store-backed one rebuilt
            assert await app.tools.run_tool("ver", {"city": "x"}) == {"city": "x", "units": "v"}

    asyncio.run(run())


def test_rehydrate_skips_record_whose_active_body_is_absent(pg: FakeVersioningPg, monkeypatch, caplog):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # Two persisted presets. ``list_presets`` and ``list_active_bodies`` are
            # two separate store reads, so a delete landing between them leaves a
            # record whose active body is already gone. Model that read-skew: both
            # records survive in ``list_presets`` while ``absent`` is dropped from
            # the active-body map.
            await _create_versioned("present", "weather", {"units": "v"}, [], [])
            await _create_versioned("absent", "weather", {"units": "g"}, [], [])

            # Simulate reload_config wiping every runtime preset before rehydrate.
            for name in ("present", "absent"):
                await mgr.remove(name)

            real_bodies = app.presets.list_active_bodies

            async def _bodies_without_absent():
                bodies = await real_bodies()
                bodies.pop("absent", None)
                return bodies

            monkeypatch.setattr(app.presets, "list_active_bodies", _bodies_without_absent)

            # A record with no active body must be SKIPPED, never a bare
            # ``bodies[rec.name]`` KeyError that aborts the whole boot/reload.
            with caplog.at_level(logging.WARNING, logger="tai_skeleton.presets.manager"):
                await mgr.rehydrate()

            # The skipped record is neither registered nor quarantined — just dropped.
            assert not mgr.is_registered("absent")
            assert not mgr.is_quarantined("absent")
            assert "absent" not in await app.tools.get_tools()
            # The present-body preset rebuilt normally and stays runnable.
            assert mgr.is_registered("present")
            assert await app.tools.run_tool("present", {"city": "x"}) == {"city": "x", "units": "v"}
            # The skip was logged loudly for the missing name.
            assert "absent" in caplog.text

    asyncio.run(run())


# -- quarantine: the three stale-preset causes -------------------------------


def test_rehydrate_quarantines_foreign_name(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # Seed a persisted preset whose NAME is a live base tool. The store's
            # create-time collision guard (rightly) blocks this via the preset
            # view, so seed through the generic store to model a name that only
            # BECAME a base tool after the preset was persisted.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await app.versioning.store.create("preset", "weather", body.model_dump())
            await mgr.rehydrate()  # app still boots — no raise

            assert mgr.is_quarantined("weather")
            # Not registered as a preset; the foreign base tool still owns the name.
            assert not mgr.is_registered("weather")
            assert await app.tools.run_tool("weather", {"city": "x"}) == {"city": "x", "units": "metric"}

            # The DELETE-conflicted branch drops the quarantine entry immediately.
            mgr.drop_quarantine("weather")
            assert not mgr.is_quarantined("weather")

    asyncio.run(run())


def test_rehydrate_quarantines_missing_base_tool(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await app.presets.store.create_preset(
                PresetSpec(name="orphan", base_tool="gone_tool", fixed_kwargs={}), extensions=[], tags=[]
            )
            await mgr.rehydrate()
            assert mgr.is_quarantined("orphan")
            assert not mgr.is_registered("orphan")
            assert "orphan" not in await app.tools.get_tools()

    asyncio.run(run())


def test_rehydrate_quarantines_preset_owned_base(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # A valid versioned preset, plus another whose base_tool is that preset
            # — a preset may not be another preset's base, in EITHER load order.
            await _create_versioned("base_preset", "weather", {"units": "v"}, [], [])
            await app.presets.store.create_preset(
                PresetSpec(name="chained", base_tool="base_preset", fixed_kwargs={}), extensions=[], tags=[]
            )
            await mgr.remove("base_preset")  # clear runtime state before the rehydrate
            await mgr.rehydrate()

            assert mgr.is_registered("base_preset")  # the legitimate one rebuilt
            assert mgr.is_quarantined("chained")  # the preset-on-preset rejected
            assert not mgr.is_registered("chained")

    asyncio.run(run())


def test_rehydrate_idempotent_self_registration_no_conflict(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await _create_versioned("wv", "weather", {"units": "v"}, [["exta"]], [])
            # A second rehydrate (as a redundant reload would trigger) rebuilds the
            # same preset cleanly — no conflict, still runnable, exactly one branch.
            await mgr.rehydrate()
            assert mgr.is_registered("wv")
            assert not mgr.is_quarantined("wv")
            assert await app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v"}
            assert _live_tool_names().count("wv_exta") == 1

    asyncio.run(run())


# -- reconcile after a scoped MCP change -------------------------------------


# -- quarantine reason storage -----------------------------------------------


def test_quarantine_reason_readable_and_cleared(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            # A stored preset whose name is a live base tool quarantines on rehydrate.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await app.versioning.store.create("preset", "weather", body.model_dump())
            await mgr.rehydrate()

            assert mgr.is_quarantined("weather")
            reason = mgr.quarantine_reason("weather")
            assert reason is not None
            assert "occupied by an existing tool" in reason
            # A non-quarantined / unknown name carries no reason.
            assert mgr.quarantine_reason("nope") is None
            # Drop clears BOTH membership and the reason.
            mgr.drop_quarantine("weather")
            assert not mgr.is_quarantined("weather")
            assert mgr.quarantine_reason("weather") is None

    asyncio.run(run())


def test_quarantine_reason_bulk_reset_is_coherent(pg: FakeVersioningPg):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await app.presets.store.create_preset(
                PresetSpec(name="orphan", base_tool="gone_tool", fixed_kwargs={}), extensions=[], tags=[]
            )
            await mgr.rehydrate()
            assert "gone_tool" in (mgr.quarantine_reason("orphan") or "")

            # A second rehydrate wipes the map wholesale then rebuilds it — still
            # quarantined, same reason, no stale entry accreted.
            await mgr.rehydrate()
            assert mgr.is_quarantined("orphan")
            assert "gone_tool" in (mgr.quarantine_reason("orphan") or "")
            assert set(mgr.quarantined_names()) == {"orphan"}

    asyncio.run(run())


def test_reconcile_quarantines_on_reregister_failure(pg: FakeVersioningPg, monkeypatch):
    async def run():
        async with app.app_context(_manifest()):
            mgr = app.preset_manager
            await _create_versioned("wv", "weather", {"units": "v"}, [], [])

            # The base tool is still live, but re-registration fails (the environment
            # changed after the reload) — the preset is quarantined, never left
            # half-bound to a stale closure.
            async def _boom(*args, **kwargs):
                raise TaiValidationError("reconcile re-register failure")

            monkeypatch.setattr(mgr, "_register", _boom)
            await mgr.reconcile_bases({"weather"})

            assert mgr.is_quarantined("wv")
            assert not mgr.is_registered("wv")

    asyncio.run(run())
