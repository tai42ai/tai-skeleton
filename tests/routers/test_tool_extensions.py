"""The tool-extensions router — GET/POST ``/api/tools/{name}/extensions`` over
the REAL manifest edit + reload path.

Each case drives the route handlers directly (the router-test pattern) inside a
live ``app.app_context``, with an in-memory config manager that round-trips the
manifest dict (``read_manifest``/``write_manifest``) so ``reload_config`` actually
re-reads the written manifest and rebinds the branch tools — the same flow
``POST /api/mcp-config`` uses. Base tools ``shout``/``ping`` come from
``tests.app._fixtures.tools_b``; the extension catalog (WRAPPER ``marka``/``markb``/
``argswrap`` + BACKEND ``backendx``/``backendy``) from ``tests.app._fixtures.ext_kinds``.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

import pytest
from starlette.requests import Request

from tai_skeleton.app import instance
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.manifest import Manifest
from tai_skeleton.routers import tool_extensions as router
from tests._fakes.bus import FakeBus

_EXT_MODULE = "tests.app._fixtures.ext_kinds"
_TOOLS_B = "tests.app._fixtures.tools_b"
_SPARE = "tests.routers._ext_fixtures"
_DUP = "tests.routers._ext_dup"


# -- request / response helpers ----------------------------------------------


def _request(method: str, path: str, *, body: Any = None, **path_params: str) -> Request:
    payload = b"" if body is None else json.dumps(body).encode()
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": path_params,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _data(resp) -> Any:
    return json.loads(bytes(resp.body))["data"]


def _err(resp) -> str:
    return json.loads(bytes(resp.body))["error"]


async def _get(name: str) -> Any:
    return await router.get_tool_extensions(_request("GET", f"/api/tools/{name}/extensions", name=name))


async def _post(name: str, combos: list[list[str]]):
    return await router.set_tool_extensions(
        _request("POST", f"/api/tools/{name}/extensions", name=name, body={"combos": combos})
    )


async def _tools() -> set[str]:
    return set(await instance.app.tools.get_tools())


# -- manifest builders -------------------------------------------------------


def _single(extensions: dict[str, Any] | None = None, include: list[str] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "title": "fx",
        "module": _TOOLS_B,
        "include": ["shout", "ping"] if include is None else include,
    }
    if extensions is not None:
        cfg["extensions"] = extensions
    return {"extensions_modules": [_EXT_MODULE], "tools": [cfg]}


def _two(a_ext: dict[str, Any], b_ext: dict[str, Any]) -> dict[str, Any]:
    return {
        "extensions_modules": [_EXT_MODULE],
        "tools": [
            {"title": "fx", "module": _TOOLS_B, "include": ["shout", "ping"], "extensions": a_ext},
            {"title": "fx2", "module": _SPARE, "include": ["spare"], "extensions": b_ext},
        ],
    }


def _dup() -> dict[str, Any]:
    return {
        "extensions_modules": [_EXT_MODULE],
        "tools": [
            {"title": "fx", "module": _TOOLS_B, "include": ["shout"]},
            {"title": "dup", "module": _DUP, "include": ["shout"]},
        ],
    }


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def cfg(monkeypatch) -> dict[str, Any]:
    """An in-memory config manager: ``read_manifest`` returns the stored dict and
    ``write_manifest`` replaces it, so the route's edit + ``reload_config`` round
    trips exactly as against a file."""
    state: dict[str, Any] = {"manifest": {}}

    def read_manifest() -> dict[str, Any]:
        return deepcopy(state["manifest"])

    def write_manifest(manifest: dict[str, Any]) -> None:
        state["manifest"] = deepcopy(manifest)

    manager = instance.app.config.config_manager

    def mutate_manifest(mutator: Any) -> dict[str, Any]:
        # The transactional seam ConfigService drives: read the current document
        # (dynamically, so a test that re-patches ``read_manifest`` mid-flight is
        # honored), run the mutator, and persist only if it returns without raising.
        document = manager.read_manifest()
        mutator(document)
        manager.write_manifest(document)
        return document

    def replace_manifest(document: dict[str, Any]) -> dict[str, Any]:
        manager.write_manifest(document)
        return document

    monkeypatch.setattr(manager, "read_manifest", read_manifest)
    monkeypatch.setattr(manager, "write_manifest", write_manifest)
    monkeypatch.setattr(manager, "read_manifest_preserved", read_manifest)
    monkeypatch.setattr(manager, "mutate_manifest", mutate_manifest)
    monkeypatch.setattr(manager, "replace_manifest", replace_manifest)
    monkeypatch.setattr(manager, "read_env", dict)
    return state


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    """Tear down every runtime-registered / quarantined preset after each test —
    the ``PresetManager`` singleton outlives one ``app_context``."""
    yield
    mgr = instance.app.preset_manager

    async def _teardown() -> None:
        for name in list(mgr.registered_names()):
            await mgr.remove(name)

    asyncio.run(_teardown())
    for name in list(mgr.quarantined_names()):
        mgr.drop_quarantine(name)


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test. The server
    outlives one ``app_context``, so a branch tool bound in a prior test would
    otherwise linger and defeat this module's ABSENCE assertions (branch cleared,
    clear-all drops the branch)."""

    async def _clear() -> None:
        provider = instance.app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


@asynccontextmanager
async def _running(cfg: dict[str, Any], manifest_dict: dict[str, Any]):
    cfg["manifest"] = deepcopy(manifest_dict)
    async with instance.app.app_context(Manifest.model_validate(manifest_dict)):
        yield


# -- GET ---------------------------------------------------------------------


def test_get_reports_map_combos_and_available(cfg):
    async def run():
        async with _running(cfg, _single({"shout": [["marka"]]})):
            data = _data(await _get("shout"))
            assert data["combos"] == [["marka"]]
            catalog = {entry["name"]: entry["kind"] for entry in data["available"]}
            assert catalog["marka"] == "wrapper"
            assert catalog["backendx"] == "backend"

    asyncio.run(run())


def test_get_unknown_tool_404(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await _get("nope")
            assert resp.status_code == 404

    asyncio.run(run())


def test_get_multi_combo_is_lossless(cfg):
    async def run():
        async with _running(cfg, _single({"shout": [["marka"], ["marka", "markb"]]})):
            assert _data(await _get("shout"))["combos"] == [["marka"], ["marka", "markb"]]

    asyncio.run(run())


def test_get_unions_combos_across_configs(cfg):
    async def run():
        async with _running(cfg, _two({"shout": [["marka"]]}, {"shout": [["markb"]]})):
            # The manifest merges the two configs' combos; GET returns the union.
            assert _data(await _get("shout"))["combos"] == [["marka"], ["markb"]]

    asyncio.run(run())


def test_get_empty_when_no_map_entry(cfg):
    async def run():
        async with _running(cfg, _single()):
            assert _data(await _get("shout"))["combos"] == []

    asyncio.run(run())


# -- POST: apply + reload ----------------------------------------------------


def test_post_single_combo_binds_branch(cfg):
    async def run():
        async with _running(cfg, _single()):
            assert "shout_marka" not in await _tools()
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 200, _err(resp)
            assert "shout_marka" in await _tools()
            assert await instance.app.tools.run_tool("shout_marka", {"text": "hi"}) == "hi|a"
            # The persisted config carries the new map; include is untouched.
            written = cfg["manifest"]["tools"][0]
            assert written["extensions"] == {"shout": [["marka"]]}
            assert written["include"] == ["shout", "ping"]

    asyncio.run(run())


def test_post_multi_combo_binds_all_and_round_trips(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await _post("shout", [["marka"], ["markb"]])
            assert resp.status_code == 200, _err(resp)
            assert {"shout_marka", "shout_markb"} <= await _tools()
            assert _data(await _get("shout"))["combos"] == [["marka"], ["markb"]]

    asyncio.run(run())


def test_post_stacked_combo_binds_layered_branch(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await _post("shout", [["marka", "markb"]])
            assert resp.status_code == 200, _err(resp)
            tools = await _tools()
            assert {"shout_marka", "shout_marka_markb"} <= tools
            assert await instance.app.tools.run_tool("shout_marka_markb", {"text": "hi"}) == "hi|a|b"

    asyncio.run(run())


def test_post_edit_add_remove_reorder_survives_reload(cfg):
    async def run():
        async with _running(cfg, _single({"shout": [["marka"]]})):
            # Add a combo.
            assert (await _post("shout", [["marka"], ["markb"]])).status_code == 200
            assert _data(await _get("shout"))["combos"] == [["marka"], ["markb"]]
            # Remove a combo — the dropped branch tears down.
            assert (await _post("shout", [["markb"]])).status_code == 200
            tools = await _tools()
            assert "shout_markb" in tools
            assert "shout_marka" not in tools
            # Reorder.
            assert (await _post("shout", [["markb"], ["marka"]])).status_code == 200
            assert _data(await _get("shout"))["combos"] == [["markb"], ["marka"]]

    asyncio.run(run())


def test_post_empty_combos_clears_and_drops_key(cfg):
    async def run():
        async with _running(cfg, _single({"shout": [["marka"]]})):
            assert "shout_marka" in await _tools()
            resp = await _post("shout", [])
            assert resp.status_code == 200, _err(resp)
            assert "shout_marka" not in await _tools()
            assert _data(await _get("shout"))["combos"] == []
            # The clear drops the map key entirely (never a present-but-empty value).
            assert "extensions" not in cfg["manifest"]["tools"][0]

    asyncio.run(run())


# -- POST: guards (nothing written) ------------------------------------------


def test_post_double_backend_400_nothing_written(cfg):
    async def run():
        async with _running(cfg, _single()):
            before = deepcopy(cfg["manifest"])
            resp = await _post("shout", [["backendx", "backendy"]])
            assert resp.status_code == 400
            assert cfg["manifest"] == before

    asyncio.run(run())


def test_post_empty_inner_combo_400_both_shapes(cfg):
    async def run():
        async with _running(cfg, _single()):
            before = deepcopy(cfg["manifest"])
            assert (await _post("shout", [[]])).status_code == 400
            assert (await _post("shout", [["marka"], []])).status_code == 400
            assert cfg["manifest"] == before

    asyncio.run(run())


def test_post_unknown_extension_400(cfg):
    async def run():
        async with _running(cfg, _single()):
            before = deepcopy(cfg["manifest"])
            resp = await _post("shout", [["ghost"]])
            assert resp.status_code == 400
            assert "unknown extension" in _err(resp)
            assert cfg["manifest"] == before

    asyncio.run(run())


def test_post_consolidation_guard_409_nothing_written(cfg):
    async def run():
        async with _running(cfg, _two({"shout": [["marka"]]}, {"shout": [["markb"]]})):
            before = deepcopy(cfg["manifest"])
            resp = await _post("shout", [["argswrap"]])
            assert resp.status_code == 409
            assert "consolidate" in _err(resp)
            assert cfg["manifest"] == before

    asyncio.run(run())


def test_two_configs_binding_one_tool_name_collide_at_boot(cfg):
    # Two tools configs each providing a tool named ``shout`` is a genuine name
    # collision. Under the server's ``on_duplicate="error"`` the second bind fails
    # loudly at boot rather than silently last-write-win — caught earlier and more
    # visibly than the extension-apply route's later owning-config guard.
    async def run():
        with pytest.raises(ValueError, match="already exists"):
            async with _running(cfg, _dup()):
                pass

    asyncio.run(run())


def test_post_dynamic_unmaterialized_not_provided(cfg):
    async def run():
        async with _running(cfg, _single()):
            # A tool REQUESTED in the registry but never bound (absent from
            # resolved_includes) has no owning config to guess.
            instance.app._tool_registry.register_tool("phantom")
            resp = await _post("phantom", [["marka"]])
            assert resp.status_code == 400
            assert "not currently provided by any config" in _err(resp)

    asyncio.run(run())


def test_post_on_preset_tool_not_provided(cfg):
    async def run():
        async with _running(cfg, _single()):
            # A preset tool is registered by the engine, provided by NO manifest
            # config — its extensions are authored via the presets API, so the
            # manifest route correctly refuses it.
            await instance.app.preset_manager.register("pre", "shout", {}, [], [], "d")
            assert "pre" in await _tools()
            resp = await _post("pre", [["marka"]])
            assert resp.status_code == 400
            assert "not currently provided by any config" in _err(resp)

    asyncio.run(run())


def test_post_on_branch_tool_not_provided(cfg):
    async def run():
        # A branch/composed tool (the tool a combo PRODUCES, e.g. ``shout_marka``) is
        # never itself an extension TARGET — only its base ``shout`` is. Applying to
        # it must report "not currently provided" and persist nothing, never resolve a
        # bogus owner and write a branch-keyed mapping that then fails to rebind.
        async with _running(cfg, _single({"shout": [["marka"]]})):
            assert "shout_marka" in await _tools()  # a real, listable branch tool
            before = deepcopy(cfg["manifest"])
            resp = await _post("shout_marka", [["markb"]])
            assert resp.status_code == 400
            assert "not currently provided by any config" in _err(resp)
            assert cfg["manifest"] == before

    asyncio.run(run())


# -- dynamic (empty-include) config ------------------------------------------


def test_apply_on_dynamic_config_keeps_siblings_and_stays_dynamic(cfg):
    async def run():
        async with _running(cfg, _single(include=[])):
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 200, _err(resp)
            tools = await _tools()
            # The mapped tool got its branch; the sibling survived the reload.
            assert {"shout", "shout_marka", "ping"} <= tools
            assert await instance.app.tools.run_tool("ping", {}) == "pong"
            # include stays empty — the allowlist never flipped, so the config is
            # still dynamic (a later-added tool would still appear).
            assert cfg["manifest"]["tools"][0]["include"] == []

    asyncio.run(run())


# -- POST: body-validation branches ------------------------------------------


def _raw_post(name: str, raw: bytes) -> Request:
    """A POST whose body is delivered verbatim, so a malformed JSON payload reaches
    ``request.json()`` (the ``_request`` helper always emits valid JSON)."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/api/tools/{name}/extensions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": {"name": name},
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(scope, receive)


def test_post_invalid_json_body_400(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await router.set_tool_extensions(_raw_post("shout", b"not json"))
            assert resp.status_code == 400
            assert _err(resp) == "invalid JSON body"

    asyncio.run(run())


def test_post_non_object_body_400(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body=[["marka"]])
            )
            assert resp.status_code == 400
            assert _err(resp) == "body must be a JSON object"

    asyncio.run(run())


def test_post_missing_combos_400(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body={})
            )
            assert resp.status_code == 400
            assert "combos" in _err(resp)

    asyncio.run(run())


def test_post_combos_not_a_list_400(cfg):
    async def run():
        async with _running(cfg, _single()):
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body={"combos": "nope"})
            )
            assert resp.status_code == 400
            assert "combos" in _err(resp)

    asyncio.run(run())


# -- POST: fleet fan-out over the worker bus ---------------------------------


def _use_backend(monkeypatch, backend: object) -> None:
    monkeypatch.setattr(instance.app._backend_holder, "_backend", backend)


def test_post_apply_fans_out_over_the_worker_bus(cfg, monkeypatch):
    async def run():
        async with _running(cfg, _single()):
            # A multi-worker fleet: the reload broadcasts, and the response embeds the
            # per-origin report as its ``fanout`` summary.
            bus = FakeBus(origin="serve-x", remotes=["serve-w1"])
            monkeypatch.setattr(instance.app, "_bus", bus)
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 200, _err(resp)
            data = _data(resp)
            assert data["status"] == "ok"  # the local reload result rides the top level
            assert data["fanout"]["mode"] == "fleet"
            assert data["fanout"]["op"] == "reload_config"
            origins = {r["origin"] for r in data["fanout"]["results"]}
            assert origins == {"serve-x", "serve-w1"}
            # The reload was broadcast to the whole fleet (targets None).
            assert bus.publish_calls[0][1] is None
            # The local apply still bound the branch tool.
            assert "shout_marka" in await _tools()

    asyncio.run(run())


def test_post_apply_local_only_without_backend(cfg, monkeypatch):
    async def run():
        _use_backend(monkeypatch, None)
        async with _running(cfg, _single()):
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 200, _err(resp)
            data = _data(resp)
            # A lone worker: the local reload result at the top level plus the
            # collapsed local-only fan-out note.
            assert data["status"] == "ok"
            assert data["fanout"]["mode"] == "local-only"
            assert "shout_marka" in await _tools()

    asyncio.run(run())


def test_post_providing_config_vanished_from_manifest_400(cfg, monkeypatch):
    async def run():
        async with _running(cfg, _single()):
            # The live manifest resolves an owner, but the persisted manifest read for
            # the write no longer carries that config (e.g. a concurrent edit) — the
            # apply refuses with a 400 rather than writing a config that is not there.
            monkeypatch.setattr(instance.app.config.config_manager, "read_manifest", lambda: {"tools": []})
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 400
            assert "not found in the manifest" in _err(resp)

    asyncio.run(run())


def test_post_invalid_manifest_after_edit_400(cfg, monkeypatch):
    async def run():
        async with _running(cfg, _single()):
            # If the edited manifest fails validation it rejects loudly (400) BEFORE
            # persisting, so a malformed entry never corrupts the stored manifest.
            def boom(_: object) -> object:
                raise ValueError("bad manifest")

            monkeypatch.setattr(router.Manifest, "model_validate", boom)
            before = deepcopy(cfg["manifest"])
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 400
            assert "invalid extensions" in _err(resp)
            assert cfg["manifest"] == before

    asyncio.run(run())


def test_post_rejected_while_reload_gate_held(cfg):
    async def run():
        # The apply is a gated route: a reload holding the gate rejects it 503.
        async with _running(cfg, _single()), reload_gate.lock:
            resp = await _post("shout", [["marka"]])
            assert resp.status_code == 503

    asyncio.run(run())


def test_post_on_mcp_provided_tool_applies(cfg, monkeypatch):
    from unittest.mock import AsyncMock

    fake_tool = type(
        "_FakeMcpTool",
        (),
        {
            "name": "ping",
            "description": "ping",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {},
        },
    )()
    monkeypatch.setattr(instance.app, "_probe_mcp", AsyncMock(return_value=[fake_tool]))
    manifest = {
        "extensions_modules": [_EXT_MODULE],
        "mcp": [{"title": "probed", "include": [], "config": {"type": "http", "url": "http://x/mcp"}}],
    }

    async def run():
        async with _running(cfg, manifest):
            # An mcp config provides the tool (normalized ``probed_ping``); the write
            # targets that config's extensions map (the mcp owner-resolution path).
            resp = await _post("probed_ping", [["marka"]])
            assert resp.status_code == 200, _err(resp)
            assert cfg["manifest"]["mcp"][0]["extensions"] == {"probed_ping": [["marka"]]}
            assert "probed_ping_marka" in await _tools()

    asyncio.run(run())


def test_post_apply_local_reload_failure_after_manifest_write_propagates(cfg, monkeypatch):
    async def run():
        async with _running(cfg, _single()):
            # The manifest write lands through the transaction; the LOCAL reload then
            # fails. Per the pipeline's failure discipline the reload is still
            # broadcast (siblings converge on the persisted state) and the call
            # re-raises with the fleet report attached.
            def _boom() -> dict[str, Any]:
                raise RuntimeError("local reload boom")

            monkeypatch.setattr(instance.app, "_reload_config", _boom)
            with pytest.raises(RuntimeError, match="local reload boom"):
                await _post("shout", [["marka"]])
            # The write already landed before the reload/broadcast — the new combos
            # are persisted, so re-running the apply is the recovery.
            assert cfg["manifest"]["tools"][0]["extensions"] == {"shout": [["marka"]]}

    asyncio.run(run())


# -- dict-element (ExtensionElement) door widening ---------------------------


def test_post_dict_element_round_trips(cfg):
    async def run():
        async with _running(cfg, _single()):
            # A config-bearing dict element (the ExtensionElement shape) authors,
            # binds, and round-trips losslessly through the GET union view.
            combo = [{"name": "marka", "config": {"schema": {"type": "object"}}}]
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body={"combos": [combo]})
            )
            assert resp.status_code == 200, _err(resp)
            assert "shout_marka" in await _tools()
            assert _data(await _get("shout"))["combos"] == [combo]

    asyncio.run(run())


def test_post_mixed_string_and_dict_elements_round_trip(cfg):
    async def run():
        async with _running(cfg, _single()):
            combo = ["marka", {"name": "markb", "config": {"k": 1}}]
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body={"combos": [combo]})
            )
            assert resp.status_code == 200, _err(resp)
            assert _data(await _get("shout"))["combos"] == [combo]

    asyncio.run(run())


@pytest.mark.parametrize(
    "element",
    [
        {"config": {"k": 1}},  # missing name
        {"name": "marka"},  # missing config
        {"name": "marka", "config": "nope"},  # non-dict config
        {"name": "marka", "config": {}, "extra": 1},  # extra keys
    ],
)
def test_post_rejects_malformed_dict_element(cfg, element):
    async def run():
        async with _running(cfg, _single()):
            resp = await router.set_tool_extensions(
                _request("POST", "/api/tools/shout/extensions", name="shout", body={"combos": [[element]]})
            )
            assert resp.status_code == 400

    asyncio.run(run())


def test_post_rejects_unregistered_dict_element_name(cfg):
    async def run():
        async with _running(cfg, _single()):
            # Structurally valid (carries a config to get PAST the parser), but the
            # name is not a registered extension — the registry check rejects it.
            resp = await router.set_tool_extensions(
                _request(
                    "POST",
                    "/api/tools/shout/extensions",
                    name="shout",
                    body={"combos": [[{"name": "not_registered", "config": {}}]]},
                )
            )
            assert resp.status_code == 400
            assert "not_registered" in _err(resp)

    asyncio.run(run())
