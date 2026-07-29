"""The preset ``output_schema`` front-door over the REAL engine + store.

Drives the presets route handlers directly (the router-test pattern) inside a live
``app.app_context`` with the true ``PostgresVersionedStore`` + ``PresetStoreView``
over the stateful fake Postgres and the real ``PresetManager``. The manifest
registers the toolbox ``output_schema`` extension (so the field-vs-extension
conflict guard is reachable), a plain-tool base (``weather``, dict-returning), and
two fake agents
— ``structured_agent`` (advertises ``response_format``) and ``plain_agent`` (does
not) — so both dispatch paths and the authoring guards are exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from starlette.requests import Request
from tai42_kit.clients.impl.postgres import PostgresClient

import tai42_skeleton.versioning.store as store_module
from tai42_skeleton.app import instance
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.routers import presets as router
from tests.versioning.conftest import FakeVersioningPg

_MANIFEST = {
    "extensions_modules": ["tests.presets._ext_fixtures", "tai42_toolbox.extensions.output_schema"],
    "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
    "agents": [
        {
            "title": "ag",
            "module": "tests.routers._structured_fixtures",
            "include": ["structured_agent", "plain_agent"],
        }
    ],
}

_OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"echoed_title": {"type": "string"}, "answer": {"type": "string"}},
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
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")
    return fake


@pytest.fixture(autouse=True)
def _emit(monkeypatch) -> None:
    async def spy(kind: str) -> None:
        return None

    monkeypatch.setattr(instance.app, "emit_list_changed", spy)


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    yield
    mgr = instance.app.preset_manager

    async def _clear() -> None:
        for name in list(mgr.registered_names()):
            await mgr.remove(name)
        provider = instance.app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    for name in list(mgr.quarantined_names()):
        mgr.drop_quarantine(name)


async def _create(name: str, base_tool: str, **over: Any):
    body: dict[str, Any] = {"name": name, "base_tool": base_tool}
    body.update(over)
    return await router.create_preset(_request("POST", "/api/presets", body=body))


# -- plain-tool DECLARE + VALIDATE -------------------------------------------


def test_plain_tool_preset_advertises_and_validates(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            schema = {
                "type": "object",
                "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                "required": ["city", "units"],
            }
            resp = await _create("wx", "weather", output_schema=schema)
            assert resp.status_code == 200, _err(resp)
            # The create response, get, and list all surface the authored schema.
            assert _data(resp)["output_schema"] == schema
            got = _data(await router.get_preset(_request("GET", "/api/presets/wx", name="wx")))
            assert got["output_schema"] == schema
            rows = _data(await router.list_presets(_request("GET", "/api/presets")))
            assert next(r for r in rows if r["name"] == "wx")["output_schema"] == schema
            # A conforming result passes through.
            assert await instance.app.tools.run_tool("wx", {"city": "paris"}) == {"city": "paris", "units": "metric"}

    asyncio.run(run())


def test_plain_tool_preset_validate_raises_on_constraint_violation(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``pattern`` is a constraint keyword the faithful validator enforces;
            # ``weather`` returns units="metric", which violates it.
            schema = {
                "type": "object",
                "properties": {"units": {"type": "string", "pattern": "^imperial$"}},
                "required": ["units"],
            }
            resp = await _create("wbad", "weather", output_schema=schema)
            assert resp.status_code == 200, _err(resp)
            from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError

            with pytest.raises(JsonSchemaValidationError):
                await instance.app.tools.run_tool("wbad", {"city": "paris"})

    asyncio.run(run())


# -- agent-base FORCE (bake response_format) ---------------------------------


def test_agent_preset_bakes_response_format_and_injects_preset_name_as_title(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create("sp_auto", "structured_agent", output_schema=_OBJ_SCHEMA)
            assert resp.status_code == 200, _err(resp)
            # Advertised output schema stays the authored (title-free) value.
            got = _data(await router.get_preset(_request("GET", "/api/presets/sp_auto", name="sp_auto")))
            assert got["output_schema"] == _OBJ_SCHEMA
            assert "title" not in got["output_schema"]
            # A real run returns the structured result; the baked response_format
            # carried the preset name injected as its title (no double-validation —
            # the agent's drain is the only validator, so the structured result flows
            # through the preset layer unchecked).
            result = await instance.app.tools.run_tool("sp_auto", {"user_message": "hi"})
            assert result == {"echoed_title": "sp_auto", "answer": "hi"}

    asyncio.run(run())


def test_agent_preset_preserves_an_authored_title(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            schema = {**_OBJ_SCHEMA, "title": "Custom"}
            resp = await _create("sp_titled", "structured_agent", output_schema=schema)
            assert resp.status_code == 200, _err(resp)
            result = await instance.app.tools.run_tool("sp_titled", {"user_message": "hi"})
            # The authored title is preserved, never overwritten by the preset name.
            assert result["echoed_title"] == "Custom"

    asyncio.run(run())


# -- authoring guards (all loud 400s) ----------------------------------------


def test_agent_without_response_format_rejects_output_schema(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create("vp", "plain_agent", output_schema=_OBJ_SCHEMA)
            assert resp.status_code == 400
            assert "does not support forced structured output" in _err(resp)

    asyncio.run(run())


def test_output_schema_field_and_extension_entry_conflict_rejected(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create(
                "cf",
                "weather",
                output_schema=_OBJ_SCHEMA,
                extensions=[[{"name": "output_schema", "config": {"schema": _OBJ_SCHEMA}}]],
            )
            assert resp.status_code == 400
            assert "conflicts with an explicit 'output_schema' extension entry" in _err(resp)

    asyncio.run(run())


def test_meta_schema_invalid_output_schema_rejected(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``type`` must be a string or array of strings — an integer is not a
            # valid draft-2020-12 schema.
            resp = await _create("bad", "weather", output_schema={"type": 123})
            assert resp.status_code == 400
            assert "not a valid JSON Schema" in _err(resp)

    asyncio.run(run())


def test_non_object_output_schema_rejected(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create("scalar", "weather", output_schema={"type": "string"})
            assert resp.status_code == 400
            assert "must be an object schema" in _err(resp)

    asyncio.run(run())


# -- version round-trip + the three carry-forward modes ----------------------


def test_output_schema_version_round_trip_and_carry_forward_modes(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            assert (await _create("v", "weather", output_schema=_OBJ_SCHEMA)).status_code == 200

            # Mode 1 — omitted carries forward (a tags-only save keeps the schema).
            r1 = await router.save_version(_request("POST", "/api/presets/v/versions", body={"tags": ["x"]}, name="v"))
            assert r1.status_code == 200, _err(r1)
            got = _data(await router.get_preset(_request("GET", "/api/presets/v", name="v")))
            assert got["output_schema"] == _OBJ_SCHEMA

            # Mode 2 — an explicit new schema wins, and re-binds on load.
            new_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
            r2 = await router.save_version(
                _request("POST", "/api/presets/v/versions", body={"output_schema": new_schema}, name="v")
            )
            assert r2.status_code == 200, _err(r2)
            got = _data(await router.get_preset(_request("GET", "/api/presets/v", name="v")))
            assert got["output_schema"] == new_schema

            # Mode 3 — an explicit null clears.
            r3 = await router.save_version(
                _request("POST", "/api/presets/v/versions", body={"output_schema": None}, name="v")
            )
            assert r3.status_code == 200, _err(r3)
            got = _data(await router.get_preset(_request("GET", "/api/presets/v", name="v")))
            assert got["output_schema"] is None

            # The version history carries the field on every row.
            versions = _data(await router.list_versions(_request("GET", "/api/presets/v/versions", name="v")))
            assert versions[0]["body"]["output_schema"] == _OBJ_SCHEMA
            assert versions[-1]["body"]["output_schema"] is None

    asyncio.run(run())


# -- HTTP-door dict-element round-trip + rejections (preset doors) ------------


def test_preset_create_dict_extension_element_round_trips(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            combo = [{"name": "output_schema", "config": {"schema": _OBJ_SCHEMA}}]
            resp = await _create("dx", "weather", extensions=[combo])
            assert resp.status_code == 200, _err(resp)
            # The dict element round-trips through create response + get.
            assert _data(resp)["extensions"] == [combo]
            got = _data(await router.get_preset(_request("GET", "/api/presets/dx", name="dx")))
            assert got["extensions"] == [combo]

    asyncio.run(run())


def test_preset_version_save_dict_extension_element_round_trips(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            assert (await _create("dv", "weather")).status_code == 200
            combo = [{"name": "output_schema", "config": {"schema": _OBJ_SCHEMA}}]
            resp = await router.save_version(
                _request("POST", "/api/presets/dv/versions", body={"extensions": [combo]}, name="dv")
            )
            assert resp.status_code == 200, _err(resp)
            got = _data(await router.get_preset(_request("GET", "/api/presets/dv", name="dv")))
            assert got["extensions"] == [combo]

    asyncio.run(run())


@pytest.mark.parametrize(
    "element",
    [
        {"config": {"schema": _OBJ_SCHEMA}},  # missing name
        {"name": "output_schema"},  # missing config
        {"name": "output_schema", "config": "nope"},  # non-dict config
        {"name": "output_schema", "config": {}, "extra": 1},  # extra keys
    ],
)
def test_preset_create_rejects_malformed_dict_element(pg, element):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create("mal", "weather", extensions=[[element]])
            assert resp.status_code == 400

    asyncio.run(run())


@pytest.mark.parametrize(
    "element",
    [
        {"config": {"schema": _OBJ_SCHEMA}},
        {"name": "output_schema"},
        {"name": "output_schema", "config": "nope"},
        {"name": "output_schema", "config": {}, "extra": 1},
    ],
)
def test_preset_version_save_rejects_malformed_dict_element(pg, element):
    async def run():
        async with instance.app.app_context(_manifest()):
            assert (await _create("mv", "weather")).status_code == 200
            resp = await router.save_version(
                _request("POST", "/api/presets/mv/versions", body={"extensions": [[element]]}, name="mv")
            )
            assert resp.status_code == 400

    asyncio.run(run())


def test_preset_create_rejects_unregistered_dict_element_name(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            # Structurally valid (carries a config to get PAST the parser), but the
            # name is not a registered extension — the registry check rejects it.
            resp = await _create("un", "weather", extensions=[[{"name": "not_registered", "config": {}}]])
            assert resp.status_code == 400
            assert "not_registered" in _err(resp)

    asyncio.run(run())


# -- plain-string extension regression ---------------------------------------


def test_plain_string_extension_combo_still_works(pg):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _create("ps", "weather", extensions=[["exta"]])
            assert resp.status_code == 200, _err(resp)
            got = _data(await router.get_preset(_request("GET", "/api/presets/ps", name="ps")))
            assert got["extensions"] == [["exta"]]

    asyncio.run(run())
