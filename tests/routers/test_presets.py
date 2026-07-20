"""The presets router — list/create/get/versions/rollback/delete over the REAL
engine + store.

Every case drives the route handlers directly (the router-test pattern) inside a
live ``app.app_context`` with the true ``PostgresVersionedStore`` +
``PresetStoreView`` over the stateful fake Postgres (the ``pg`` fixture) and the
real ``PresetManager`` — so create/edit atomicity, the wire-diff emit matrix, and
the base-plus-branch teardown are exercised end-to-end, not mocked. ``emit`` swaps
the concrete server's ``emit_list_changed`` for a recorder so each mutation's
notification count is asserted exactly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import pytest
from starlette.requests import Request
from tai_contract.presets import PresetBody
from tai_contract.presets.errors import PresetNotFoundError
from tai_kit.clients.impl.postgres import PostgresClient

import tai_skeleton.versioning.store as store_module
from tai_skeleton.app import instance
from tai_skeleton.app.bus import FleetOrigin, FleetResult, OpOutcome, OriginKind, OriginResult
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.manifest import Manifest
from tai_skeleton.operations import presets as preset_ops
from tai_skeleton.routers import presets as router
from tai_skeleton.routers import tools as tools_router
from tests.versioning.conftest import FakeVersioningPg

_MANIFEST = {
    "extensions_modules": ["tests.presets._ext_fixtures"],
    "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
}


def _manifest() -> Manifest:
    return Manifest.model_validate(_MANIFEST)


# -- request / response helpers ----------------------------------------------


def _request(method: str, path: str, *, body: Any = None, query: str = "", **path_params: str) -> Request:
    payload = b"" if body is None else json.dumps(body).encode()
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", b"application/json")],
        "query_string": query.encode(),
        "path_params": path_params,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _data(resp) -> Any:
    return json.loads(bytes(resp.body))["data"]


def _err(resp) -> str:
    return json.loads(bytes(resp.body))["error"]


def _non_role_documents(pg: FakeVersioningPg) -> list[dict[str, Any]]:
    """The versioned documents excluding the admin/editor/viewer role templates the
    access-control startup seeds into the store at every boot, so an assertion over a
    preset operation's own store writes ignores that boot seed."""
    return [d for d in pg.documents if d["kind"] != "role"]


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def pg(monkeypatch) -> FakeVersioningPg:
    fake = FakeVersioningPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(store_module, "client_ctx", fake_client_ctx)
    # Signal a store-configured deployment (the gate list/delete/reconcile consult):
    # faking the store transport must also set its ``VERSIONING_STORE_*`` namespace.
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")
    return fake


@pytest.fixture
def emit(monkeypatch) -> list[str]:
    """Record every ``emit_list_changed`` the routes fire."""
    calls: list[str] = []

    async def spy(kind: str) -> None:
        calls.append(kind)

    monkeypatch.setattr(instance.app, "emit_list_changed", spy)
    return calls


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    """Tear down every runtime-registered / quarantined preset after each test —
    the process app (FastMCP server + ``PresetManager``) is a singleton that
    outlives one ``app_context``, so a preset a test binds would otherwise leak."""
    yield
    mgr = instance.app.preset_manager

    # Clear every remaining registered preset and base tool: the singleton server
    # + ``PresetManager`` outlive one ``app_context``, so a manifest-bound base
    # (weather/echo) or a leaked preset would collide with the next test's bind
    # under ``on_duplicate="error"``.
    async def _clear() -> None:
        for name in list(mgr.registered_names()):
            await mgr.remove(name)
        provider = instance.app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    for name in list(mgr.quarantined_names()):
        mgr.drop_quarantine(name)


# -- create body helpers -----------------------------------------------------


def _create_body(name: str, base_tool: str = "weather", **over: Any) -> dict[str, Any]:
    body = {"name": name, "base_tool": base_tool}
    body.update(over)
    return body


async def _create_versioned(name: str, base_tool: str = "weather", **over: Any) -> None:
    resp = await router.create_preset(_request("POST", "/api/presets", body=_create_body(name, base_tool, **over)))
    assert resp.status_code == 200, _err(resp)


# -- list --------------------------------------------------------------------


def test_list_returns_store_backed_rows(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"}, tags=["cat"])

            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            assert [r["name"] for r in rows] == ["ver"]
            assert rows[0]["active_version"] == 1
            assert rows[0]["conflicted"] is False
            assert rows[0]["tags"] == ["cat"]
            # No row carries an ``ephemeral`` key.
            assert "ephemeral" not in rows[0]

    asyncio.run(run())


def test_list_ignores_removed_tier_param(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            # ``tier`` is not a query parameter the route reads — an unknown query
            # key is ignored (never a 400), and the response is the same store-backed
            # listing.
            resp = await router.list_presets(_request("GET", "/api/presets", query="tier=all"))
            assert resp.status_code == 200
            assert [r["name"] for r in _data(resp)] == ["ver"]

    asyncio.run(run())


def test_list_surfaces_conflicted_badge(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A stored preset whose NAME is a live base tool: seed through the
            # GENERIC store (the view's create-guard would block it), then rehydrate
            # so it lands in the quarantine set.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()

            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            row = next(r for r in rows if r["name"] == "weather")
            assert row["conflicted"] is True

    asyncio.run(run())


# -- create: happy paths + emit ----------------------------------------------


def test_create_persists_registers_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("wv", fixed_kwargs={"units": "imperial"}, description="d", tags=["c"]),
                )
            )
            data = _data(resp)
            assert data["active_version"] == 1
            # The record carries no ``ephemeral`` / ``persisted`` keys.
            assert "ephemeral" not in data
            assert "persisted" not in data
            # Live + runnable, the baked value served as a fixed constant.
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "imperial"}
            # Persisted, and exactly one notification.
            assert [d["name"] for d in _non_role_documents(pg)] == ["wv"]
            assert emit == ["tool"]

    asyncio.run(run())


def test_create_ignores_removed_versioned_field(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``versioned`` is not a body field the route reads — sending it (any
            # value) is ignored, and the create always persists (creation is
            # versioned-only).
            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("wv", versioned=False, fixed_kwargs={"units": "e"}),
                )
            )
            assert resp.status_code == 200, _err(resp)
            assert [d["name"] for d in _non_role_documents(pg)] == ["wv"]  # persisted despite versioned=false
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "e"}

    asyncio.run(run())


def test_create_typed_schema_hides_baked_key(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "imperial"})
            tool = await instance.app.tools.get_tool("wv")
            props = tool.parameters.get("properties", {})
            assert "units" not in props  # baked key hidden
            assert props["city"]["type"] == "string"  # remaining arg keeps its typed schema

    asyncio.run(run())


# -- create: atomicity + guards ----------------------------------------------


def test_create_collision_guard_before_write_409_no_row_no_emit(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("weather", base_tool="echo"))
            )
            assert resp.status_code == 409
            assert _non_role_documents(pg) == []  # guard ran BEFORE the store write
            assert emit == []

    asyncio.run(run())


def test_create_bad_extension_rejected_before_write_no_emit(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # An unknown extension fails the pre-write registry validation, so the
            # create is a 400 that never writes a store row (validate-then-commit).
            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("bad", base_tool="echo", extensions=[["ghost_ext"]]),
                )
            )
            assert resp.status_code == 400
            assert "ghost_ext" in _err(resp)
            assert _non_role_documents(pg) == []  # nothing committed
            assert "bad" not in await instance.app.tools.get_tools()
            assert emit == []

    asyncio.run(run())


def test_create_bad_fixed_kwargs_rejected_before_write(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A ``fixed_kwargs`` key that is not an argument of the base tool fails
            # the dry-run bake — a 400 with nothing committed.
            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("bad", base_tool="echo", fixed_kwargs={"nope": 1}),
                )
            )
            assert resp.status_code == 400
            assert _non_role_documents(pg) == []
            assert emit == []

    asyncio.run(run())


def test_create_residual_register_fail_preserves_soft_deleted_ghost(pg, emit, monkeypatch):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A soft-deleted ghost of the same name already exists.
            await _create_versioned("g", base_tool="echo", fixed_kwargs={})
            await instance.app.presets.store.soft_delete("g")
            await instance.app.preset_manager.remove("g")
            ghost_docs = [dict(d) for d in _non_role_documents(pg)]  # the is_active=False row
            ghost_versions = [dict(v) for v in pg.versions]  # its audit history
            assert ghost_docs
            assert all(not d["is_active"] for d in ghost_docs)

            # Force a RESIDUAL (post-validation) register failure — the environment
            # changing after the pre-write checks passed — so the store write
            # commits and must be rolled back via the scoped HARD delete.
            async def _boom(*args, **kwargs):
                raise TaiValidationError("residual register failure")

            monkeypatch.setattr(instance.app.preset_manager, "register", _boom)

            with pytest.raises(TaiValidationError):
                await router.create_preset(_request("POST", "/api/presets", body=_create_body("g", base_tool="echo")))

            # The scoped hard delete wiped ONLY the failed create's active row + its
            # versions; the audit ghost AND its version history remain intact.
            assert [dict(d) for d in _non_role_documents(pg)] == ghost_docs
            assert [dict(v) for v in pg.versions] == ghost_versions

    asyncio.run(run())


def test_create_over_existing_preset_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("dup", fixed_kwargs={"units": "e"})
            resp = await router.create_preset(_request("POST", "/api/presets", body=_create_body("dup")))
            assert resp.status_code == 409
            # The live tool is untouched and no second store row was written.
            assert await instance.app.tools.run_tool("dup", {"city": "x"}) == {"city": "x", "units": "e"}
            assert [d["name"] for d in _non_role_documents(pg)] == ["dup"]

    asyncio.run(run())


def test_create_quarantined_name_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await instance.app.presets.store.create_preset(
                _spec("orphan", base_tool="gone_tool"), extensions=[], tags=[]
            )
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("orphan")
            resp = await router.create_preset(_request("POST", "/api/presets", body=_create_body("orphan")))
            assert resp.status_code == 409
            assert "delete the quarantined record first" in _err(resp)

    asyncio.run(run())


def test_create_preset_typed_base_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("basep", fixed_kwargs={"units": "v"})
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("chained", base_tool="basep"))
            )
            assert resp.status_code == 400

    asyncio.run(run())


def test_create_unknown_base_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("x", base_tool="nope"))
            )
            assert resp.status_code == 400

    asyncio.run(run())


def test_create_explicit_empty_extensions_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("x", base_tool="echo", extensions=[]))
            )
            assert resp.status_code == 400
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("x", base_tool="echo", extensions=[[]]))
            )
            assert resp.status_code == 400

    asyncio.run(run())


# -- get / versions ----------------------------------------------------------


def test_create_over_stale_requested_unbound_name_seeds_extensions(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # Model a manifest tool that is REQUESTED but never bound (its MCP server
            # was down at bind time): present in the registry's requested set + its
            # _tools entry, absent from the live tool set — so it slips past the
            # bound-only name-collision guard. Creating a preset on that name must
            # clear the stale requested entry first (the register unregister-first
            # step), so the preset's extension combos still seed and the branch
            # binds — never a silent drop.
            instance.app._tool_registry.register_tool("phantom")
            assert "phantom" not in await instance.app.tools.get_tools()

            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("phantom", base_tool="echo", extensions=[["exta"]]),
                )
            )
            assert resp.status_code == 200, _err(resp)
            tools = set(await instance.app.tools.get_tools())
            assert {"phantom", "phantom_exta"} <= tools
            assert await instance.app.tools.run_tool("phantom_exta", {"text": "hi"}) == "hi|a"

    asyncio.run(run())


def test_created_preset_tags_appear_in_tools_tags_map(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="weather", tags=["geo", "eu"])
            # The additive GET /api/tools/tags surfaces the native tags a preset
            # body projects onto its bound tool, sorted for a stable order.
            resp = await tools_router.tool_tags(_request("GET", "/api/tools/tags"))
            entries = json.loads(bytes(resp.body))["data"]
            wv = next(e for e in entries if e["name"] == "wv")
            assert wv["tags"] == ["eu", "geo"]

    asyncio.run(run())


def test_get_preset_and_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"}, tags=["c"])
            data = _data(await router.get_preset(_request("GET", "/api/presets/wv", name="wv")))
            assert data["fixed_kwargs"] == {"units": "v"}
            assert data["tags"] == ["c"]
            assert data["extensions"] == []
            # An absent name is a 404.
            assert (await router.get_preset(_request("GET", "/api/presets/nope", name="nope"))).status_code == 404

    asyncio.run(run())


def test_versions_list_and_get(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v2"})
            rows = _data(await router.list_versions(_request("GET", "/api/presets/wv/versions", name="wv")))
            assert [r["version"] for r in rows] == [1, 2]
            assert [r["is_current"] for r in rows] == [False, True]
            one = _data(await router.get_version(_request("GET", "/api/presets/wv/versions/1", name="wv", version="1")))
            assert one["body"]["fixed_kwargs"] == {"units": "v1"}
            missing = await router.get_version(_request("GET", "/api/presets/wv/versions/9", name="wv", version="9"))
            assert missing.status_code == 404

    asyncio.run(run())


# -- save-version + rollback -------------------------------------------------


def test_save_version_reloads_and_serves_new_kwargs(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": {"units": "v2"}})
            )
            assert _data(resp)["version"] == 2
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v2"}

    asyncio.run(run())


def test_save_version_empty_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            resp = await router.save_version(_request("POST", "/api/presets/wv/versions", name="wv", body={}))
            assert resp.status_code == 400

    asyncio.run(run())


def test_save_version_absent_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.save_version(
                _request("POST", "/api/presets/nope/versions", name="nope", body={"tags": ["x"]})
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_rollback_serves_old_kwargs(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v2"})
            await instance.app.preset_manager.reload("wv")
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
            )
            assert _data(resp)["active_version"] == 1
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v1"}

    asyncio.run(run())


def test_rollback_missing_version_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 9})
            )
            assert resp.status_code == 404

    asyncio.run(run())


# -- edit-path re-register atomicity -----------------------------------------


def test_save_version_bad_extension_rejected_before_write(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]])
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            emit.clear()

            # A new version referencing an unknown ext is rejected BEFORE any store
            # write: a 400, the store version is unchanged, the branch survives, and
            # NO emit (validate-then-commit, no bricking quarantine).
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"extensions": [["ghost_ext"]]})
            )
            assert resp.status_code == 400
            assert "ghost_ext" in _err(resp)
            versions = _data(await router.list_versions(_request("GET", "/api/presets/wv/versions", name="wv")))
            assert [v["version"] for v in versions] == [1]  # nothing committed
            assert await instance.app.tools.run_tool("wv", {"text": "hi"}) == "hi"
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            assert emit == []

    asyncio.run(run())


def test_save_version_bad_fixed_kwargs_rejected_before_write(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={})
            emit.clear()
            # A ``fixed_kwargs`` key that is not an argument of the base tool fails
            # the dry-run bake — a 400, store version unchanged, no emit.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": {"nope": 1}})
            )
            assert resp.status_code == 400
            versions = _data(await router.list_versions(_request("GET", "/api/presets/wv/versions", name="wv")))
            assert [v["version"] for v in versions] == [1]
            assert emit == []

    asyncio.run(run())


def test_save_version_residual_reload_failure_repoints_active_no_emit(pg, emit, monkeypatch):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={})
            emit.clear()

            # Force a RESIDUAL reload failure (the environment changed after the
            # pre-write validation passed): the appended row stays as inert history,
            # but the active version is re-pointed to the prior one so the store and
            # live never diverge, and no emit fires.
            async def _boom(_name):
                raise TaiValidationError("residual reload failure")

            monkeypatch.setattr(instance.app.preset_manager, "reload", _boom)

            with pytest.raises(TaiValidationError):
                await router.save_version(_request("POST", "/api/presets/wv/versions", name="wv", body={"tags": ["x"]}))
            # active_version re-pointed back to 1 (the appended v2 stays as history).
            record = await instance.app.presets.store.get_preset("wv")
            assert record.active_version == 1
            assert emit == []

    asyncio.run(run())


# -- delete ------------------------------------------------------------------


def test_delete_unregisters_base_and_branches_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"], ["extb"]])
            assert {"wv", "wv_exta", "wv_extb"} <= set(await instance.app.tools.get_tools())
            emit.clear()

            resp = await router.delete_preset(_request("DELETE", "/api/presets/wv", name="wv"))
            assert resp.status_code == 200
            tools = set(await instance.app.tools.get_tools())
            assert not ({"wv", "wv_exta", "wv_extb"} & tools)
            for name in ("wv", "wv_exta", "wv_extb"):
                with pytest.raises(RuntimeError):
                    await instance.app.tools.run_tool(name, {"text": "hi"})
            assert emit == ["tool"]

    asyncio.run(run())


def test_delete_absent_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.delete_preset(_request("DELETE", "/api/presets/nope", name="nope"))
            assert resp.status_code == 404

    asyncio.run(run())


def test_delete_conflicted_store_side_only_no_emit(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("weather")
            emit.clear()

            resp = await router.delete_preset(_request("DELETE", "/api/presets/weather", name="weather"))
            assert resp.status_code == 200
            # The stored doc is gone, the quarantine entry dropped, and the FOREIGN
            # base tool that owns the name stays runnable — no teardown, no emit.
            assert _non_role_documents(pg) == []
            assert not instance.app.preset_manager.is_quarantined("weather")
            assert await instance.app.tools.run_tool("weather", {"city": "x"}) == {"city": "x", "units": "metric"}
            assert emit == []

    asyncio.run(run())


def test_conflicted_write_locked_then_clean_delete_recreate(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await instance.app.presets.store.create_preset(
                _spec("orphan", base_tool="gone_tool"), extensions=[], tags=[]
            )
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("orphan")
            emit.clear()

            # Write-locked while conflicted.
            assert (
                await router.save_version(
                    _request("POST", "/api/presets/orphan/versions", name="orphan", body={"tags": ["x"]})
                )
            ).status_code == 409
            assert (
                await router.rollback_preset(
                    _request("POST", "/api/presets/orphan/rollback", name="orphan", body={"version": 1})
                )
            ).status_code == 409
            # The 409 short-circuits before any reload — no emit either.
            assert emit == []

            # Clean delete-then-recreate: the name is free after the conflicted
            # delete (missing-base cause), so a fresh create yields a normal preset.
            assert (
                await router.delete_preset(_request("DELETE", "/api/presets/orphan", name="orphan"))
            ).status_code == 200
            resp = await router.create_preset(
                _request(
                    "POST",
                    "/api/presets",
                    body=_create_body("orphan", base_tool="weather", fixed_kwargs={"units": "v"}),
                )
            )
            assert _data(resp)["conflicted"] is False
            assert await instance.app.tools.run_tool("orphan", {"city": "x"}) == {"city": "x", "units": "v"}

    asyncio.run(run())


# -- the emit matrix ---------------------------------------------------------


def test_emit_baked_key_change_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={})  # both city + units exposed
            emit.clear()
            # Baking a NEW key hides it from the exposed inputSchema — a wire change.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": {"units": "v"}})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]

    asyncio.run(run())


def test_emit_baked_value_only_change_emits_nothing(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            emit.clear()
            # SAME baked-key set, only the hidden VALUE changes — the serialized wire
            # tool is byte-identical, so a correct wire-diff guard emits nothing.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": {"units": "v2"}})
            )
            assert resp.status_code == 200
            assert emit == []
            # The persisted value DID change (proving the save committed).
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v2"}

    asyncio.run(run())


def test_emit_tags_only_change_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"}, tags=["a"])
            emit.clear()
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"tags": ["b"]})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]

    asyncio.run(run())


def test_emit_extensions_only_change_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]])
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            emit.clear()
            # The BASE tool's wire dump is unchanged; only the branch set changes —
            # the guard's extensions comparison must still fire the emit.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"extensions": [["extb"]]})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]
            # The save's reload BINDS the new branch tool and TEARS DOWN the old one —
            # an extensions-only edit swaps the branch set.
            tools = set(await instance.app.tools.get_tools())
            assert "wv_extb" in tools
            assert "wv_exta" not in tools
            assert await instance.app.tools.run_tool("wv_extb", {"text": "hi"}) == "hi|b"
            with pytest.raises(RuntimeError):
                await instance.app.tools.run_tool("wv_exta", {"text": "hi"})

    asyncio.run(run())


def test_save_version_omitting_extensions_carries_them_forward(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]])
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            # A save-version that OMITS `extensions` (here a tags-only edit) must carry the
            # active version's extensions FORWARD (absent → None sentinel), NOT clear them.
            # The branch tool survives.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"tags": ["beta"]})
            )
            assert resp.status_code == 200
            assert "wv_exta" in set(await instance.app.tools.get_tools())
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            # The new active version persisted the carried-forward combos, not a cleared [].
            v2 = _data(await router.get_version(_request("GET", "/api/presets/wv/versions/2", name="wv", version="2")))
            assert v2["body"]["extensions"] == [["exta"]]

    asyncio.run(run())


def test_save_version_clearing_extensions_tears_branch_and_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]])
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"
            emit.clear()
            # An EXPLICIT `extensions: []` CLEARS (distinct from the absent carry-forward
            # sentinel): the reader's `[]`→clear branch tears the branch tool down and the
            # new version stores no combos — the explicit-clear write path.
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"extensions": []})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]
            assert "wv_exta" not in set(await instance.app.tools.get_tools())
            with pytest.raises(RuntimeError):
                await instance.app.tools.run_tool("wv_exta", {"text": "hi"})
            v2 = _data(await router.get_version(_request("GET", "/api/presets/wv/versions/2", name="wv", version="2")))
            assert v2["body"]["extensions"] == []

    asyncio.run(run())


def test_emit_noop_save_emits_nothing(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"}, tags=["a"])
            emit.clear()
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"tags": ["a"]})
            )  # identical
            assert resp.status_code == 200
            assert emit == []

    asyncio.run(run())


def test_emit_rollback_baked_key_change_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={})  # v1: nothing baked
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v"})  # v2: units baked
            await instance.app.preset_manager.reload("wv")
            emit.clear()
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]  # baked-key set changes back → wire change

    asyncio.run(run())


def test_emit_rollback_extensions_only_change_emits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]])  # v1
            # v2 clears the extensions (base tool's wire dump held constant), tearing wv_exta down.
            await router.save_version(_request("POST", "/api/presets/wv/versions", name="wv", body={"extensions": []}))
            assert "wv_exta" not in set(await instance.app.tools.get_tools())
            emit.clear()
            # Rolling back to v1 changes ONLY the extensions (base wire unchanged), so the
            # rollback door's extensions-comparison term must fire the emit — the rollback
            # partner of the save-version extensions-change guard (guards are duplicated inline).
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
            )
            assert resp.status_code == 200
            assert emit == ["tool"]
            assert "wv_exta" in set(await instance.app.tools.get_tools())
            assert await instance.app.tools.run_tool("wv_exta", {"text": "hi"}) == "hi|a"

    asyncio.run(run())


def test_emit_rehydrate_fires_no_per_preset_emit(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            await instance.app.preset_manager.remove("wv")
            emit.clear()
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_registered("wv")
            assert emit == []  # the lifecycle path emits once elsewhere, not per-preset

    asyncio.run(run())


# -- tags round-trip through the route ---------------------------------------


def test_tags_round_trip_through_save_version(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"}, tags=["release"])
            await router.save_version(_request("POST", "/api/presets/wv/versions", name="wv", body={"tags": ["beta"]}))
            assert (await instance.app.tools.get_tool("wv")).tags == {"beta"}
            v1 = _data(await router.get_version(_request("GET", "/api/presets/wv/versions/1", name="wv", version="1")))
            assert v1["body"]["tags"] == ["release"]

    asyncio.run(run())


def _spec(name: str, *, base_tool: str, fixed_kwargs: dict[str, Any] | None = None):
    from tai_contract.agent.base import PresetSpec

    return PresetSpec(name=name, description="d", base_tool=base_tool, fixed_kwargs=fixed_kwargs or {})


# -- body-validation + error branches ----------------------------------------


def _raw_request(method: str, path: str, *, raw: bytes, **path_params: str) -> Request:
    """A request whose body is delivered verbatim, so a genuinely malformed JSON
    payload reaches ``request.json()`` (the ``_request`` helper always emits valid
    JSON)."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": path_params,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(scope, receive)


def test_create_invalid_json_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(_raw_request("POST", "/api/presets", raw=b"not json"))
            assert resp.status_code == 400
            assert _err(resp) == "invalid JSON body"
            assert _non_role_documents(pg) == []

    asyncio.run(run())


def test_create_non_object_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(_request("POST", "/api/presets", body=[1, 2, 3]))
            assert resp.status_code == 400
            assert _err(resp) == "body must be a JSON object"

    asyncio.run(run())


def test_create_missing_name_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(_request("POST", "/api/presets", body={"base_tool": "echo"}))
            assert resp.status_code == 400
            assert "name" in _err(resp)

    asyncio.run(run())


def test_create_bad_field_types_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            base = {"name": "x", "base_tool": "echo"}
            resp = await router.create_preset(_request("POST", "/api/presets", body={**base, "description": 1}))
            assert resp.status_code == 400
            assert "description" in _err(resp)
            resp = await router.create_preset(_request("POST", "/api/presets", body={**base, "fixed_kwargs": 1}))
            assert resp.status_code == 400
            assert "fixed_kwargs" in _err(resp)
            resp = await router.create_preset(_request("POST", "/api/presets", body={**base, "tags": [1]}))
            assert resp.status_code == 400
            assert "tags" in _err(resp)
            resp = await router.create_preset(_request("POST", "/api/presets", body={**base, "extensions": "nope"}))
            assert resp.status_code == 400
            assert "extensions" in _err(resp)
            # every rejected body was refused BEFORE any store write
            assert _non_role_documents(pg) == []

    asyncio.run(run())


def test_create_over_unrehydrated_store_row_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A store row that was seeded but NEVER registered/rehydrated slips past
            # the manager's quarantine/collision/duplicate pre-checks, so the store's
            # own duplicate guard is the one that fires — a 409, nothing new persisted.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "seeded", body.model_dump())
            before = [dict(d) for d in _non_role_documents(pg)]
            resp = await router.create_preset(
                _request("POST", "/api/presets", body=_create_body("seeded", base_tool="echo"))
            )
            assert resp.status_code == 409
            assert "already exists" in _err(resp)
            assert [dict(d) for d in _non_role_documents(pg)] == before  # no second active row
            assert emit == []

    asyncio.run(run())


def test_list_versions_absent_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.list_versions(_request("GET", "/api/presets/nope/versions", name="nope"))
            assert resp.status_code == 404

    asyncio.run(run())


def test_get_version_non_integer_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            resp = await router.get_version(_request("GET", "/api/presets/wv/versions/abc", name="wv", version="abc"))
            assert resp.status_code == 400
            assert "integer" in _err(resp)

    asyncio.run(run())


def test_save_version_bad_field_types_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": 1})
            )
            assert resp.status_code == 400
            assert "fixed_kwargs" in _err(resp)
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"tags": [1]})
            )
            assert resp.status_code == 400
            assert "tags" in _err(resp)
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"extensions": "nope"})
            )
            assert resp.status_code == 400
            assert "extensions" in _err(resp)

    asyncio.run(run())


def test_rollback_bad_version_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": "x"})
            )
            assert resp.status_code == 400
            assert "integer" in _err(resp)

    asyncio.run(run())


def test_rollback_absent_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/nope/rollback", name="nope", body={"version": 1})
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_rollback_residual_reload_failure_repoints_active_no_emit(pg, emit, monkeypatch):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={})
            await instance.app.presets.store.save_version("wv", tags=["v2"])
            await instance.app.preset_manager.reload("wv")  # active = v2, live
            emit.clear()

            # Force a RESIDUAL reload failure (the environment changed after the
            # target validation passed): the re-point to v1 is committed, then reload
            # raises, so the active version is re-pointed back to the prior one (v2)
            # so store + live never diverge, and no emit fires. Mirrors the
            # save_version residual case on the rollback path.
            async def _boom(_name):
                raise TaiValidationError("residual reload failure")

            monkeypatch.setattr(instance.app.preset_manager, "reload", _boom)

            with pytest.raises(TaiValidationError):
                await router.rollback_preset(
                    _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
                )
            # active_version re-pointed back to 2 (the prior active), never left at 1.
            record = await instance.app.presets.store.get_preset("wv")
            assert record.active_version == 2
            assert emit == []

    asyncio.run(run())


def test_rollback_to_unbindable_target_400_nothing_committed(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", base_tool="echo", fixed_kwargs={})
            await instance.app.presets.store.save_version("wv", tags=["v2"])
            await instance.app.preset_manager.reload("wv")  # active = v2, live
            emit.clear()

            # The base tool the stored versions were authored over is gone from the
            # CURRENT live registry (an MCP deregister / a manifest edit removed it),
            # so the v1 target body no longer binds. Rollback reads the target and
            # validates THAT before any write, so this is a 400 that commits nothing
            # rather than a bricking re-point.
            instance.app.tools.remove_tool("echo")

            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
            )
            assert resp.status_code == 400
            assert "cannot bind" in _err(resp)
            # Active version unchanged, nothing committed, no emit.
            record = await instance.app.presets.store.get_preset("wv")
            assert record.active_version == 2
            assert emit == []

    asyncio.run(run())


def test_list_reads_active_bodies_in_one_batched_query(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("a", fixed_kwargs={"units": "a"})
            await _create_versioned("b", fixed_kwargs={"units": "b"})
            await _create_versioned("c", fixed_kwargs={"units": "c"})
            pg.executed.clear()

            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            assert {r["name"] for r in rows} == {"a", "b", "c"}

            # ONE batched JOIN read for all rows (list_active_bodies), never a
            # per-row get_active_body round-trip — a regression to per-row reads
            # fails this.
            batched = [s for s in pg.executed if s.startswith("SELECT d.name, v.body FROM versioned_documents d")]
            per_row = [s for s in pg.executed if s.startswith("SELECT v.body FROM versioned_documents d")]
            assert len(batched) == 1
            assert per_row == []

    asyncio.run(run())


# -- name validation (tool-name-safe) ----------------------------------------


def test_create_invalid_name_400_nothing_written(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            for bad in ("a/b", "x" * 65, "bad name", "po+preset"):
                resp = await router.create_preset(
                    _request("POST", "/api/presets", body=_create_body(bad, base_tool="echo"))
                )
                assert resp.status_code == 400, bad
            assert _non_role_documents(pg) == []
            assert emit == []

    asyncio.run(run())


# -- concurrency: per-name serialization -------------------------------------


def test_two_concurrent_creates_one_wins_one_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # Both creates pass the pre-checks (their awaits interleave), then race
            # the store write + the manager's per-name lock: the first persists and
            # registers, the second raises the typed conflict mapped to 409 — never a
            # silent clobber.
            reqs = [_request("POST", "/api/presets", body=_create_body("dup", base_tool="echo")) for _ in range(2)]
            results = await asyncio.gather(*(router.create_preset(r) for r in reqs))
            assert sorted(r.status_code for r in results) == [200, 409]
            assert instance.app.preset_manager.is_registered("dup")
            assert emit == ["tool"]  # exactly one create fired an emit

    asyncio.run(run())


def test_reload_completes_without_deadlock(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v2"})
            # reload holds the per-name lock across teardown + re-register and calls
            # the INTERNAL unlocked register, so the held non-reentrant lock is never
            # re-acquired; a bounded wait proves the lock structure is deadlock-free.
            await asyncio.wait_for(instance.app.preset_manager.reload("wv"), timeout=5)
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v2"}

    asyncio.run(run())


# -- delete frees the name ---------------------------------------------------


def test_delete_removes_and_frees_name(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("eph", fixed_kwargs={"units": "e"})
            emit.clear()
            resp = await router.delete_preset(_request("DELETE", "/api/presets/eph", name="eph"))
            assert resp.status_code == 200
            data = _data(resp)
            assert data["deleted"] is True
            assert "persisted" not in data
            assert not instance.app.preset_manager.is_registered("eph")
            assert "eph" not in await instance.app.tools.get_tools()
            assert emit == ["tool"]

            # The freed name is immediately recreatable (no lingering registration).
            await _create_versioned("eph", fixed_kwargs={"units": "e2"})
            assert await instance.app.tools.run_tool("eph", {"city": "x"}) == {"city": "x", "units": "e2"}

    asyncio.run(run())


def test_delete_neither_stored_nor_registered_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.delete_preset(_request("DELETE", "/api/presets/ghost", name="ghost"))
            assert resp.status_code == 404

    asyncio.run(run())


@pytest.fixture
def store_less(monkeypatch):
    """A store-less deployment: no ``VERSIONING_STORE_*`` env, and any versioned-store
    open is a hard error — the preset routes must never touch Postgres (a store-less
    deploy can hold no preset)."""
    for key in list(os.environ):
        if key.startswith("VERSIONING_STORE_"):
            monkeypatch.delenv(key, raising=False)

    @asynccontextmanager
    async def forbid_client_ctx(client_cls, settings=None, **kwargs):
        raise AssertionError("versioned store opened in a store-less deployment")
        yield  # pragma: no cover - unreachable, satisfies the context-manager protocol

    monkeypatch.setattr(store_module, "client_ctx", forbid_client_ctx)


def test_list_store_less_empty_without_opening_store(store_less, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A store-less deploy has no presets — the list is empty and never opens
            # the versioned store.
            assert _data(await router.list_presets(_request("GET", "/api/presets"))) == []

    asyncio.run(run())


def test_create_store_less_refuses_with_503_without_opening_store(store_less, emit):
    """A create on a store-less deploy is refused with 503 (the same
    versioned_store_configured predicate the list/delete/reconcile paths gate on)
    BEFORE any Postgres open — never a 500 from a failed store connection."""

    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.create_preset(_request("POST", "/api/presets", body=_create_body("vers")))
            assert resp.status_code == 503
            assert "versioned-document store" in _err(resp)
            # Nothing was registered — the refusal is total, not a partial create.
            assert not instance.app.preset_manager.is_registered("vers")

    asyncio.run(run())


def test_delete_store_less_404_without_opening_store(store_less, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # No preset can exist store-less, so any delete is a 404 that never opens
            # the versioned store.
            resp = await router.delete_preset(_request("DELETE", "/api/presets/ghost", name="ghost"))
            assert resp.status_code == 404

    asyncio.run(run())


# -- rename ------------------------------------------------------------------


def test_rename_moves_row_rebinds_and_emits_once(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned(
                "old", base_tool="echo", fixed_kwargs={}, extensions=[["exta"]], tags=["cat"], description="d"
            )
            await instance.app.presets.store.save_version("old", tags=["cat2"])  # v2 → history to preserve
            await instance.app.preset_manager.reload("old")
            assert await instance.app.tools.run_tool("old_exta", {"text": "hi"}) == "hi|a"
            emit.clear()

            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
            )
            assert resp.status_code == 200, _err(resp)
            assert _data(resp) == {"name": "new", "renamed_from": "old", "active_version": 2}

            # Store row moved; the full version history is intact under the new name.
            assert [d["name"] for d in _non_role_documents(pg)] == ["new"]
            versions = _data(await router.list_versions(_request("GET", "/api/presets/new/versions", name="new")))
            assert [v["version"] for v in versions] == [1, 2]

            # Old base + branch gone; new base + branch bound with the identical baked
            # spec (kwargs, extensions/branches, tags, description, output_schema).
            tools = set(await instance.app.tools.get_tools())
            assert not ({"old", "old_exta"} & tools)
            assert {"new", "new_exta"} <= tools
            assert await instance.app.tools.run_tool("new_exta", {"text": "hi"}) == "hi|a"
            spec = instance.app.preset_manager.get_spec("new")
            assert (spec.base_tool, spec.description, spec.tags) == ("echo", "d", ["cat2"])
            assert spec.extensions == [["exta"]]
            assert spec.output_schema is None
            # A rename changes the listing by definition — exactly one emit.
            assert emit == ["tool"]

    asyncio.run(run())


def test_rename_invalid_and_noop_names_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            emit.clear()
            for bad in ("", "a/b", "x" * 65):
                resp = await router.rename_preset(
                    _request("POST", "/api/presets/old/rename", name="old", body={"new_name": bad})
                )
                assert resp.status_code == 400, bad
            # Missing new_name.
            assert (
                await router.rename_preset(_request("POST", "/api/presets/old/rename", name="old", body={}))
            ).status_code == 400
            # A no-op rename is a loud 400, never a silent 200.
            noop = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "old"})
            )
            assert noop.status_code == 400
            assert "differ" in _err(noop)
            # Nothing moved, nothing emitted.
            assert [d["name"] for d in _non_role_documents(pg)] == ["old"]
            assert emit == []

    asyncio.run(run())


def test_rename_absent_old_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.rename_preset(
                _request("POST", "/api/presets/nope/rename", name="nope", body={"new_name": "x"})
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_rename_store_less_404_without_opening_store(store_less, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # No preset can exist store-less, so a rename is a 404 that never opens the
            # versioned store.
            resp = await router.rename_preset(
                _request("POST", "/api/presets/ghost/rename", name="ghost", body={"new_name": "x"})
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_rename_onto_foreign_live_tool_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", base_tool="echo", fixed_kwargs={})
            emit.clear()
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "weather"})
            )
            assert resp.status_code == 409
            assert "collides with an existing tool" in _err(resp)
            # Untouched: old still bound, store row still "old", no emit.
            assert await instance.app.tools.run_tool("old", {"text": "hi"}) == "hi"
            assert [d["name"] for d in _non_role_documents(pg)] == ["old"]
            assert emit == []

    asyncio.run(run())


def test_rename_onto_existing_preset_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "a"})
            await _create_versioned("taken", fixed_kwargs={"units": "b"})
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "taken"})
            )
            assert resp.status_code == 409
            assert "already exists" in _err(resp)

    asyncio.run(run())


def test_rename_quarantined_old_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await instance.app.presets.store.create_preset(
                _spec("orphan", base_tool="gone_tool"), extensions=[], tags=[]
            )
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("orphan")
            resp = await router.rename_preset(
                _request("POST", "/api/presets/orphan/rename", name="orphan", body={"new_name": "clean"})
            )
            assert resp.status_code == 409
            assert "delete-only" in _err(resp)

    asyncio.run(run())


def test_rename_onto_quarantined_new_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            await instance.app.presets.store.create_preset(
                _spec("orphan", base_tool="gone_tool"), extensions=[], tags=[]
            )
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("orphan")
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "orphan"})
            )
            assert resp.status_code == 409
            assert "delete the quarantined record first" in _err(resp)

    asyncio.run(run())


def test_rename_onto_agent_tool_name_400(pg, emit, monkeypatch):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            # An agent's ``tool_name`` may not be a bound tool, so name_conflicts misses
            # it — the dedicated agent-name-space guard catches it with a 400. The
            # guard lives on the operation, so the seam is patched there.
            monkeypatch.setattr(preset_ops, "_agent_tool_names", lambda: {"agentic"})
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "agentic"})
            )
            assert resp.status_code == 400
            assert "agent tool name" in _err(resp)

    asyncio.run(run())


def test_rename_blocked_by_referencing_presets_409_lists_all_sorted(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("target", base_tool="weather", fixed_kwargs={"units": "v"})
            # Two presets whose ACTIVE body composes "target" — one via top-level
            # tool_names, one via a nested subagents spec. Seeded through the GENERIC
            # store (the referee scan reads active bodies, not the live registry);
            # neutral names pin the SORTED, fully-listed referee message.
            direct = PresetBody(
                base_tool="echo", description="d", fixed_kwargs={"tool_names": ["target"]}, extensions=[], tags=[]
            )
            nested = PresetBody(
                base_tool="echo",
                description="d",
                fixed_kwargs={"subagents": [{"tool_names": ["target"]}]},
                extensions=[],
                tags=[],
            )
            await instance.app.versioning.store.create("preset", "z_ref", direct.model_dump())
            await instance.app.versioning.store.create("preset", "a_ref", nested.model_dump())
            emit.clear()

            resp = await router.rename_preset(
                _request("POST", "/api/presets/target/rename", name="target", body={"new_name": "renamed"})
            )
            assert resp.status_code == 409
            assert "['a_ref', 'z_ref']" in _err(resp)  # every referee, sorted
            # Nothing moved, nothing emitted.
            assert await instance.app.tools.run_tool("target", {"city": "x"}) == {"city": "x", "units": "v"}
            assert emit == []

    asyncio.run(run())


def test_rename_rejected_while_reload_locked(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            async with reload_gate.lock:
                resp = await router.rename_preset(
                    _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
                )
            assert resp.status_code == 503

    asyncio.run(run())


def test_rename_register_failure_compensates_repoints_store_no_emit(pg, emit, monkeypatch, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", base_tool="echo", fixed_kwargs={})
            emit.clear()
            backend.ops.clear()

            # Force a RESIDUAL re-register failure (the environment changed after the
            # pre-checks): the store move to "new" must be re-pointed BACK to "old" by
            # the compensating rename so store + live never diverge.
            async def _boom(_name):
                raise TaiValidationError("residual re-register failure")

            monkeypatch.setattr(instance.app.preset_manager, "reload", _boom)

            with pytest.raises(TaiValidationError):
                await router.rename_preset(
                    _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
                )
            # Store row re-pointed back to "old"; the OLD binding was never touched.
            assert [d["name"] for d in _non_role_documents(pg)] == ["old"]
            assert (await instance.app.presets.store.get_preset("old")).active_version == 1
            assert await instance.app.tools.run_tool("old", {"text": "hi"}) == "hi"
            assert not instance.app.preset_manager.is_registered("new")
            # The failure surfaced before either the emit or the fan-out.
            assert emit == []
            assert backend.ops == []

    asyncio.run(run())


# -- bus fan-out -------------------------------------------------------------


class _RecordingBus:
    """A fake worker bus recording the fleet ops the preset routes publish; a
    ``fail`` flag makes every publish raise the confirmed-broadcast error, and a
    ``non_converged`` flag makes every publish return a reachable-but-non-converged
    report (a silent sibling) so the publisher's loud non-convergence log is
    exercised. Its ``subscribe`` parks (mirroring the no-op local bus) so
    ``app_context`` startup does not hang."""

    def __init__(self) -> None:
        self.reloaded: list[tuple[str, str]] = []
        self.removed: list[tuple[str, str]] = []
        # A unified chronological log across BOTH ops so a cross-op ordering (rename's
        # reload-before-remove) is assertable; the per-list logs above stay for the
        # single-op cases.
        self.ops: list[tuple[str, str, str]] = []
        self.fail = False
        self.non_converged = False
        self._origin = FleetOrigin(origin="serve-test", kind=OriginKind.serve, pid=1)

    @property
    def origin(self) -> FleetOrigin:
        return self._origin

    async def subscribe(self, origin: Any, callback: Any, on_ready: Any = None) -> None:
        await asyncio.Event().wait()

    async def census(self) -> list[FleetOrigin]:
        return [self._origin]

    async def validate_targets(self, targets: Any) -> None:
        return None

    async def publish(self, op: dict[str, Any], targets: Any, local: Any) -> FleetResult:
        if self.fail:
            raise RuntimeError("worker gh-2 did not confirm within timeout")
        name, kind, tool = op["op"], op["kind"], op["name"]
        if name == "reload_tool":
            self.reloaded.append((kind, tool))
            self.ops.append(("reload", kind, tool))
        elif name == "remove_tool":
            self.removed.append((kind, tool))
            self.ops.append(("remove", kind, tool))
        if self.non_converged:
            # Reachable, but a sibling never confirmed applied — reachable=True and
            # ok=False, so the publisher's non-convergence check must fire loudly.
            return FleetResult(
                op=name,
                results=[
                    OriginResult(origin=self._origin.origin, outcome=OpOutcome.applied),
                    OriginResult(origin="gh-2", outcome=OpOutcome.missing),
                ],
            )
        # Mirror the real bus: a converged report (self entry applied), so the
        # publisher's non-convergence check is a no-op.
        return FleetResult(op=name, results=[OriginResult(origin=self._origin.origin, outcome=OpOutcome.applied)])


@pytest.fixture
def backend(monkeypatch) -> _RecordingBus:
    """Install a recording fake worker bus on the app so the routes' fan-out is
    observable; monkeypatch restores the real bus builder after the test."""
    fake = _RecordingBus()
    monkeypatch.setattr(instance.app, "_build_bus", lambda origin_kind: fake)
    return fake


def test_create_fans_out_reload_tool(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            assert backend.reloaded == [("preset", "wv")]
            assert backend.removed == []

    asyncio.run(run())


def test_save_version_fans_out_reload_tool(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            backend.reloaded.clear()
            resp = await router.save_version(
                _request("POST", "/api/presets/wv/versions", name="wv", body={"fixed_kwargs": {"units": "v2"}})
            )
            assert resp.status_code == 200
            assert backend.reloaded == [("preset", "wv")]

    asyncio.run(run())


def test_rollback_fans_out_reload_tool(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v2"})
            await instance.app.preset_manager.reload("wv")
            backend.reloaded.clear()
            resp = await router.rollback_preset(
                _request("POST", "/api/presets/wv/rollback", name="wv", body={"version": 1})
            )
            assert resp.status_code == 200
            assert backend.reloaded == [("preset", "wv")]

    asyncio.run(run())


def test_delete_fans_out_remove_tool(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            backend.reloaded.clear()
            resp = await router.delete_preset(_request("DELETE", "/api/presets/wv", name="wv"))
            assert resp.status_code == 200
            assert backend.removed == [("preset", "wv")]

    asyncio.run(run())


def test_delete_conflicted_fans_out_remove_tool(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("weather")
            resp = await router.delete_preset(_request("DELETE", "/api/presets/weather", name="weather"))
            assert resp.status_code == 200
            assert backend.removed == [("preset", "weather")]

    asyncio.run(run())


def test_create_no_fanout_without_backend(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # No backend registered — the local apply IS the whole fleet, so nothing
            # is published and the create still succeeds (the None guard is correct).
            assert instance.app.backends.backend is None
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            assert instance.app.preset_manager.is_registered("wv")

    asyncio.run(run())


def test_create_fanout_failure_surfaces_after_store_write(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            backend.fail = True
            with pytest.raises(RuntimeError, match="did not confirm"):
                await router.create_preset(
                    _request("POST", "/api/presets", body=_create_body("wv", fixed_kwargs={"units": "v"}))
                )
            # The store write + local apply already landed before the fan-out raised —
            # re-running the mutation is the recovery, never a swallowed failure.
            assert [d["name"] for d in _non_role_documents(pg)] == ["wv"]
            assert instance.app.preset_manager.is_registered("wv")

    asyncio.run(run())


def test_rename_fans_out_reload_new_before_remove_old(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            backend.reloaded.clear()
            backend.ops.clear()
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
            )
            assert resp.status_code == 200, _err(resp)
            # Reload of the NEW name strictly BEFORE remove of the OLD name — both
            # briefly alive beats neither alive.
            assert backend.ops == [("reload", "preset", "new"), ("remove", "preset", "old")]

    asyncio.run(run())


def test_rename_no_fanout_without_backend(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            assert instance.app.backends.backend is None
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            resp = await router.rename_preset(
                _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
            )
            assert resp.status_code == 200, _err(resp)
            assert instance.app.preset_manager.is_registered("new")
            assert not instance.app.preset_manager.is_registered("old")

    asyncio.run(run())


def test_rename_reload_fanout_failure_does_not_publish_remove(pg, emit, backend):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("old", fixed_kwargs={"units": "v"})
            backend.fail = True
            with pytest.raises(RuntimeError, match="did not confirm"):
                await router.rename_preset(
                    _request("POST", "/api/presets/old/rename", name="old", body={"new_name": "new"})
                )
            # The reload broadcast raised, so the remove fan-out is never published.
            assert backend.removed == []
            # The store move + local rebind already landed before the fan-out raised.
            assert [d["name"] for d in _non_role_documents(pg)] == ["new"]
            assert instance.app.preset_manager.is_registered("new")

    asyncio.run(run())


def test_create_non_converged_fanout_logs_loud_error(pg, emit, backend, caplog):
    async def run():
        async with instance.app.app_context(_manifest()):
            backend.non_converged = True
            # The create succeeds and embeds the report, but the reachable-but-silent
            # sibling is a loud ERROR, never a swallowed non-convergence.
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            assert instance.app.preset_manager.is_registered("wv")

    with caplog.at_level(logging.ERROR, logger="tai_skeleton.operations._broadcast"):
        asyncio.run(run())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("did not fully converge — unconfirmed origins" in r.getMessage() for r in errors)
    assert any("gh-2" in r.getMessage() for r in errors)


def test_delete_non_converged_fanout_logs_loud_error(pg, emit, backend, caplog):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            backend.non_converged = True
            # The remove fan-out returns a reachable-but-non-converged report — the
            # loud non-convergence log must fire for the remove op too.
            resp = await router.delete_preset(_request("DELETE", "/api/presets/wv", name="wv"))
            assert resp.status_code == 200

    with caplog.at_level(logging.ERROR, logger="tai_skeleton.operations._broadcast"):
        asyncio.run(run())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("did not fully converge — unconfirmed origins" in r.getMessage() for r in errors)


# -- preset tool-reloader (backend-bus dispatch target) ----------------------


def test_preset_tool_reloader_reload_rebinds_from_active_body(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v1"})
            await instance.app.presets.store.save_version("wv", fixed_kwargs={"units": "v2"})
            # The reloader re-reads the ACTIVE store body and rebinds (the fan-out
            # op carries only kind + name).
            await instance.app.admin.run_tool_reload("preset", "reload", "wv")
            assert await instance.app.tools.run_tool("wv", {"city": "x"}) == {"city": "x", "units": "v2"}

    asyncio.run(run())


def test_preset_tool_reloader_remove_tears_down(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("wv", fixed_kwargs={"units": "v"})
            await instance.app.admin.run_tool_reload("preset", "remove", "wv")
            assert not instance.app.preset_manager.is_registered("wv")
            assert "wv" not in await instance.app.tools.get_tools()

    asyncio.run(run())


def test_preset_tool_reloader_remove_drops_quarantine(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("weather")
            await instance.app.admin.run_tool_reload("preset", "remove", "weather")
            assert not instance.app.preset_manager.is_quarantined("weather")

    asyncio.run(run())


def test_preset_tool_reloader_remove_conflicted_spares_foreign_tool(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``weather`` is a REAL manifest-bound foreign tool. A store row colliding
            # with its name rehydrates into quarantine (never bound as a preset,
            # because the name is occupied). A conflicted-delete fan-out for that name
            # must ONLY drop the quarantine entry — tearing the name down would
            # destroy the foreign tool that owns it.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("weather")

            # Drive the reloader's remove path exactly as the fan-out delivers it.
            await instance._apply_preset_tool_reload("remove", "weather")

            # The foreign tool STILL runs and the quarantine entry is dropped.
            assert await instance.app.tools.run_tool("weather", {"city": "x"}) == {"city": "x", "units": "metric"}
            assert not instance.app.preset_manager.is_quarantined("weather")

    asyncio.run(run())


def test_preset_tool_reloader_reload_missing_row_raises(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A bogus fan-out must fail loudly — reload of a name with no store row
            # raises rather than quarantining.
            with pytest.raises(PresetNotFoundError):
                await instance.app.admin.run_tool_reload("preset", "reload", "ghost")

    asyncio.run(run())


def test_preset_tool_reloader_unknown_action_raises(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            with pytest.raises(ValueError, match="Unknown tool-reload action"):
                await instance.app.admin.run_tool_reload("preset", "bogus", "wv")

    asyncio.run(run())


# -- conflicted_reason -------------------------------------------------------


def test_conflicted_reason_present_in_list_and_detail(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A stored preset whose name is a live base tool quarantines on rehydrate.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()

            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            row = next(r for r in rows if r["name"] == "weather")
            assert row["conflicted"] is True
            assert "occupied by an existing tool" in row["conflicted_reason"]

    asyncio.run(run())


def test_conflicted_reason_null_when_healthy(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            assert rows[0]["conflicted"] is False
            assert rows[0]["conflicted_reason"] is None
            detail = _data(await router.get_preset(_request("GET", "/api/presets/ver", name="ver")))
            assert detail["conflicted_reason"] is None

    asyncio.run(run())


# -- referees door -----------------------------------------------------------


def test_referees_empty(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            data = _data(await router.preset_referees(_request("GET", "/api/presets/ver/referees", name="ver")))
            assert data == {"name": "ver", "referees": []}

    asyncio.run(run())


def test_referees_lists_referencing_presets(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("leaf", base_tool="weather", fixed_kwargs={"units": "v"})
            # A preset whose ACTIVE body composes ``leaf`` as a tool references it.
            # Seeded through the GENERIC store (the referee scan reads active bodies,
            # not the live registry).
            composer = PresetBody(
                base_tool="echo", description="d", fixed_kwargs={"tool_names": ["leaf"]}, extensions=[], tags=[]
            )
            await instance.app.versioning.store.create("preset", "composer", composer.model_dump())
            data = _data(await router.preset_referees(_request("GET", "/api/presets/leaf/referees", name="leaf")))
            assert data == {"name": "leaf", "referees": ["composer"]}

    asyncio.run(run())


def test_referees_404_unknown(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.preset_referees(_request("GET", "/api/presets/nope/referees", name="nope"))
            assert resp.status_code == 404

    asyncio.run(run())


# -- validate door: create mode ----------------------------------------------


def _validate(body: dict[str, Any]):
    return router.validate_preset(_request("POST", "/api/presets/validate", body=body))


def test_validate_create_valid(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            data = _data(await _validate({"name": "newp", "base_tool": "weather", "fixed_kwargs": {"units": "x"}}))
            assert data == {"valid": True, "error": None}
            # A dry-run writes nothing.
            assert _non_role_documents(pg) == []
            assert emit == []

    asyncio.run(run())


def test_validate_create_unknown_base_tool(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            data = _data(await _validate({"name": "newp", "base_tool": "ghost"}))
            assert data["valid"] is False
            assert "not a registered tool" in data["error"]

    asyncio.run(run())


def test_validate_create_bad_kwarg(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            data = _data(await _validate({"name": "newp", "base_tool": "weather", "fixed_kwargs": {"bogus": 1}}))
            assert data["valid"] is False
            assert "cannot bind" in data["error"]

    asyncio.run(run())


def test_validate_create_invalid_combo(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            data = _data(await _validate({"name": "newp", "base_tool": "weather", "extensions": [["ghost_ext"]]}))
            assert data["valid"] is False
            assert "unknown extension" in data["error"]

    asyncio.run(run())


def test_validate_create_schema_mismatch(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            data = _data(await _validate({"name": "newp", "base_tool": "weather", "output_schema": {"type": "string"}}))
            assert data["valid"] is False
            assert "object schema" in data["error"]

    asyncio.run(run())


def test_validate_create_name_collision(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``weather`` is a live base tool — a create under that name collides.
            data = _data(await _validate({"name": "weather", "base_tool": "echo"}))
            assert data["valid"] is False
            assert "collides with an existing tool" in data["error"]

    asyncio.run(run())


def test_validate_create_missing_base_tool_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _validate({"name": "newp"})
            assert resp.status_code == 400
            assert "base_tool" in _err(resp)

    asyncio.run(run())


# -- validate door: version mode ---------------------------------------------


def test_validate_version_valid(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            data = _data(await _validate({"name": "ver", "fixed_kwargs": {"units": "y"}}))
            assert data == {"valid": True, "error": None}

    asyncio.run(run())


def test_validate_version_invalid_combo(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", base_tool="echo")
            data = _data(await _validate({"name": "ver", "extensions": [["ghost_ext"]]}))
            assert data["valid"] is False
            assert "unknown extension" in data["error"]

    asyncio.run(run())


def test_validate_version_absent_fields_carry_forward(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            # No fields at all — everything carries forward from the active body, so
            # the draft is valid.
            data = _data(await _validate({"name": "ver"}))
            assert data == {"valid": True, "error": None}

    asyncio.run(run())


def test_validate_version_provided_differing_base_tool(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", base_tool="weather", fixed_kwargs={"units": "v"})
            data = _data(await _validate({"name": "ver", "base_tool": "echo"}))
            assert data["valid"] is False
            assert data["error"] == (
                "base_tool differs from the preset's active base tool; a version cannot change the base tool"
            )

    asyncio.run(run())


def test_validate_version_provided_differing_description(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"}, description="original")
            data = _data(await _validate({"name": "ver", "description": "changed"}))
            assert data["valid"] is False
            assert data["error"] == "description is not a version field; it carries forward from the preset"

    asyncio.run(run())


def test_validate_matching_base_tool_and_description_ok(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", base_tool="weather", fixed_kwargs={"units": "v"}, description="keep")
            # Providing the SAME base_tool / description as active is not a change.
            data = _data(await _validate({"name": "ver", "base_tool": "weather", "description": "keep"}))
            assert data == {"valid": True, "error": None}

    asyncio.run(run())


def test_validate_version_quarantined_is_invalid(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A stored preset whose NAME is a live tool rehydrates into quarantine —
            # a transient cause (the name is occupied) that leaves the base tool
            # ``echo`` fully bindable, so the bind chain alone would say valid. The
            # version-mode door must mirror save_version's first gate and reject it.
            body = PresetBody(base_tool="echo", description="d", fixed_kwargs={}, extensions=[], tags=[])
            await instance.app.versioning.store.create("preset", "weather", body.model_dump())
            await instance.app.preset_manager.rehydrate()
            assert instance.app.preset_manager.is_quarantined("weather")

            data = _data(await _validate({"name": "weather", "fixed_kwargs": {}}))
            assert data["valid"] is False
            assert data["error"] == "preset 'weather' is conflicted and is delete-only"

    asyncio.run(run())


def test_validate_store_less_503(monkeypatch, emit):
    # No VERSIONING_STORE_* env: mode resolution needs the store, so refuse cleanly.
    monkeypatch.delenv("VERSIONING_STORE_PG_PASSWORD", raising=False)

    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _validate({"name": "newp", "base_tool": "weather"})
            assert resp.status_code == 503

    asyncio.run(run())


# -- version-tags door -------------------------------------------------------


def test_set_version_tags_reflected_in_list_no_rebind(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            assert emit == ["tool"]  # only the create emitted

            resp = await router.set_preset_version_tags(
                _request(
                    "PUT",
                    "/api/presets/ver/versions/1/tags",
                    body={"tags": ["stable", "reviewed"]},
                    name="ver",
                    version="1",
                )
            )
            assert resp.status_code == 200
            assert _data(resp) == {"name": "ver", "version": 1, "tags": ["stable", "reviewed"]}

            # ``list_versions`` reflects the new tags; the body is unchanged and the
            # tool never rebound (no new emit, same baked value).
            versions = _data(await router.list_versions(_request("GET", "/api/presets/ver/versions", name="ver")))
            assert versions[0]["tags"] == ["stable", "reviewed"]
            assert emit == ["tool"]
            assert await instance.app.tools.run_tool("ver", {"city": "x"}) == {"city": "x", "units": "v"}

    asyncio.run(run())


def test_set_version_tags_clear(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"}, tags=["a"])
            resp = await router.set_preset_version_tags(
                _request("PUT", "/api/presets/ver/versions/1/tags", body={"tags": []}, name="ver", version="1")
            )
            assert _data(resp) == {"name": "ver", "version": 1, "tags": []}
            versions = _data(await router.list_versions(_request("GET", "/api/presets/ver/versions", name="ver")))
            assert versions[0]["tags"] == []

    asyncio.run(run())


def test_set_version_tags_404_unknown_preset(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await router.set_preset_version_tags(
                _request("PUT", "/api/presets/nope/versions/1/tags", body={"tags": []}, name="nope", version="1")
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_set_version_tags_404_unknown_version(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            resp = await router.set_preset_version_tags(
                _request("PUT", "/api/presets/ver/versions/99/tags", body={"tags": []}, name="ver", version="99")
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_set_version_tags_bad_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _create_versioned("ver", fixed_kwargs={"units": "v"})
            resp = await router.set_preset_version_tags(
                _request("PUT", "/api/presets/ver/versions/1/tags", body={"tags": "nope"}, name="ver", version="1")
            )
            assert resp.status_code == 400

    asyncio.run(run())
