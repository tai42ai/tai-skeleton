"""Authored-agent skeleton surface — the ``spec_runnable`` report, the authoring
validation layered on ``POST /api/presets``, and the authored-agent streaming run.

Every case drives the real engine + store inside a live ``app.app_context`` (the
``test_presets`` harness) with FAKE agents registered through the manifest
``agents:`` section: a spec-runnable ``authorable_agent`` (its ``from_tool_input``
renames ``system_prompt`` -> ``system_message``), a non-authorable ``role_agent``,
an ``aliased_agent`` whose ``tool_name`` differs from its registration name, and a
``locked_agent`` (not spec-runnable, declaring ``secret_config`` in
``preset_bakeable_fields``). No concrete agent implementation is referenced — the
surface binds to the contract only.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from tai_kit.clients.impl.postgres import PostgresClient

import tai_skeleton.versioning.store as store_module
from tai_skeleton.app import instance
from tai_skeleton.manifest import Manifest
from tai_skeleton.operations import presets as preset_ops
from tai_skeleton.routers import agents as agents_router
from tai_skeleton.routers import presets as presets_router
from tests.versioning.conftest import FakeVersioningPg

_MANIFEST = {
    "extensions_modules": ["tests.presets._ext_fixtures"],
    "tools": [{"title": "fx", "module": "tests.presets._fixtures", "include": ["weather", "echo"]}],
    "agents": [
        {
            "title": "ag",
            "module": "tests.routers._authoring_fixtures",
            "include": ["authorable_agent", "role_agent", "aliased_agent", "locked_agent"],
        }
    ],
}


def _manifest() -> Manifest:
    return Manifest.model_validate(_MANIFEST)


# -- request / response helpers ----------------------------------------------


def _json_request(method: str, path: str, *, body: Any = None, query: str = "", **path_params: str) -> Request:
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


def _run_request(name: str, body: Any) -> Request:
    """A POST request for the authored-run door whose client never disconnects (the
    monitor's ``is_disconnected`` stays False so the run drains to completion)."""
    payload = json.dumps(body).encode()
    scripted = [{"type": "http.request", "body": payload, "more_body": False}]
    idx = {"i": 0}

    async def receive() -> dict[str, Any]:
        i = idx["i"]
        if i < len(scripted):
            idx["i"] += 1
            return scripted[i]
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/api/agents/authored/{name}/runs",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("1.2.3.4", 1),
        "path_params": {"name": name},
    }
    return Request(scope, receive)


def _data(resp: Response) -> Any:
    return json.loads(bytes(resp.body))["data"]


def _err(resp: Response) -> str:
    return json.loads(bytes(resp.body))["error"]


def _non_role_documents(pg: FakeVersioningPg) -> list[dict[str, Any]]:
    """The versioned documents excluding the admin/editor/viewer role templates the
    access-control startup seeds into the store at every boot, so an assertion over an
    authoring operation's own store writes ignores that boot seed."""
    return [d for d in pg.documents if d["kind"] != "role"]


async def _collect(response: Response) -> list[str]:
    assert isinstance(response, StreamingResponse)
    out: list[str] = []
    async for chunk in response.body_iterator:
        out.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())
    return out


def _data_frames(frames: list[str]) -> list[dict]:
    out: list[dict] = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        assert frame.startswith("data: "), frame
        out.append(json.loads(frame[len("data: ") :].strip()))
    return out


async def _run_authored(name: str, body: Any) -> list[dict]:
    resp = await agents_router.run_authored_agent(_run_request(name, body))
    if not isinstance(resp, StreamingResponse):
        return [{"__status__": resp.status_code, "__error__": _err(resp)}]
    return _data_frames(await _collect(resp))


# -- fixtures (mirror the presets-router harness) ----------------------------


@pytest.fixture
def pg(monkeypatch) -> FakeVersioningPg:
    fake = FakeVersioningPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(store_module, "client_ctx", fake_client_ctx)
    # Faking the store transport models a store-configured deployment, so it must
    # also set the ``VERSIONING_STORE_*`` namespace the create/list/delete routes
    # gate on (versioned_store_configured) — else a versioned author is refused.
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")
    return fake


@pytest.fixture
def emit(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def spy(kind: str) -> None:
        calls.append(kind)

    monkeypatch.setattr(instance.app, "emit_list_changed", spy)
    return calls


@pytest.fixture(autouse=True)
def _reset_preset_registry():
    yield
    mgr = instance.app.preset_manager

    # Clear every remaining preset and base tool: the singleton server +
    # ``PresetManager`` outlive one ``app_context``, so a manifest-bound base or a
    # leaked preset would collide with the next test's bind under
    # ``on_duplicate="error"``.
    async def _clear() -> None:
        for name in list(mgr.registered_names()):
            await mgr.remove(name)
        provider = instance.app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    for name in list(mgr.quarantined_names()):
        mgr.drop_quarantine(name)


# -- authoring helpers -------------------------------------------------------


async def _author(name: str, *, base_tool: str = "authorable_agent", **over: Any) -> Response:
    body = {"name": name, "base_tool": base_tool}
    body.update(over)
    return await presets_router.create_preset(_json_request("POST", "/api/presets", body=body))


# -- spec_runnable report ----------------------------------------------------


def test_list_reports_spec_runnable_marker(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            items = _data(await agents_router.list_agents(_json_request("GET", "/api/agents")))["items"]
            by_name = {it["name"]: it for it in items}
            assert by_name["authorable_agent"]["spec_runnable"] is True
            assert by_name["role_agent"]["spec_runnable"] is False
            # A declared ``preset_bakeable_fields`` does NOT flip the UI-composability
            # marker — the locked agent still reports spec_runnable false.
            assert by_name["locked_agent"]["spec_runnable"] is False
            # tool_name can differ from the registration name — both are reported.
            assert by_name["aliased_agent"]["spec_runnable"] is False
            assert by_name["aliased_agent"]["tool_name"] == "aliased_tool_name"

    asyncio.run(run())


def test_spec_runnable_route_excludes_role_agents(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            request = _json_request("GET", "/api/agents/spec-runnable")
            data = _data(await agents_router.list_spec_runnable_agents(request))
            names = {it["name"] for it in data["items"]}
            assert names == {"authorable_agent"}
            # A declared ``preset_bakeable_fields`` never enters the spec-runnable
            # picker — the locked agent is excluded.
            assert "locked_agent" not in names
            assert data["total"] == 1
            # The picker consumes the input schema.
            assert "input_schema" in data["items"][0]

    asyncio.run(run())


# -- authoring validation (over POST /api/presets) ---------------------------


def test_author_over_non_spec_runnable_agent_baking_undeclared_field_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``role_agent`` is not spec_runnable and declares no preset_bakeable_fields,
            # so baking any existing field (``text``) is rejected as not preset-bakeable —
            # the per-field honored gate, naming the field and the declaration.
            resp = await _author("authored", base_tool="role_agent", fixed_kwargs={"text": "x"})
            assert resp.status_code == 400
            err = _err(resp)
            assert "text" in err
            assert "preset_bakeable_fields" in err

    asyncio.run(run())


def test_author_empty_fixed_kwargs_over_non_spec_runnable_succeeds(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A rename/description wrapper over a non-spec_runnable agent bakes nothing,
            # so there is nothing to gate — it now creates successfully.
            resp = await _author("wrapper", base_tool="role_agent", fixed_kwargs={})
            assert resp.status_code == 200, _err(resp)

    asyncio.run(run())


def test_author_over_locked_agent_bakes_declared_field(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author("locked", base_tool="locked_agent", fixed_kwargs={"secret_config": {"key": "v"}})
            assert resp.status_code == 200, _err(resp)

            # The baked field is HIDDEN in the exposed tool schema.
            tools = await instance.app.tools.get_tools()
            schema = tools["locked"].to_mcp_tool().model_dump()["inputSchema"]
            assert "secret_config" not in schema.get("properties", {})

            # Running forwards the baked value into the agent's recorded kwargs.
            await instance.app.tools.run_tool("locked", {"user_message": "hi"})
            agent = instance.app.agents.get_agent("locked_agent")
            assert cast(Any, agent).received_kwargs["secret_config"] == {"key": "v"}

            # A caller that passes the baked field at run is rejected by the transform's
            # own schema validation (the baked constant is hidden and not overridable).
            with pytest.raises(TypeError, match="unexpected keyword"):
                await instance.app.tools.run_tool("locked", {"user_message": "hi", "secret_config": {"x": 1}})

    asyncio.run(run())


def test_author_over_locked_agent_baking_undeclared_field_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``user_message`` is an existing ToolInput field but NOT declared bakeable.
            resp = await _author("locked", base_tool="locked_agent", fixed_kwargs={"user_message": "x"})
            assert resp.status_code == 400
            err = _err(resp)
            assert "user_message" in err
            assert "preset_bakeable_fields" in err

    asyncio.run(run())


def test_authored_run_over_locked_agent_streams_and_rejects_baked_override(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("locked", base_tool="locked_agent", fixed_kwargs={"secret_config": {"key": "v"}})
            # The authored-run door serves the locked agent (it never checks
            # spec_runnable) — the baked value reaches astream.
            frames = await _run_authored("locked", {"user_message": "hi"})
            assert frames[0]["text"] == "secret={'key': 'v'}"
            assert frames[-1] == {"type": "stream.end"}
            # Naming the baked field in the run body is the existing loud 400.
            rejected = await _run_authored("locked", {"user_message": "hi", "secret_config": {"x": 1}})
            assert rejected[0]["__status__"] == 400
            assert "cannot override the fixed field 'secret_config'" in rejected[0]["__error__"]

    asyncio.run(run())


def test_author_unknown_fixed_kwargs_field_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author("authored", fixed_kwargs={"bogus": 1})
            assert resp.status_code == 400
            assert "bogus" in _err(resp)

    asyncio.run(run())


def test_author_type_invalid_fixed_kwargs_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author("authored", fixed_kwargs={"count": "not-an-int"})
            assert resp.status_code == 400
            assert "count" in _err(resp)

    asyncio.run(run())


def test_author_constraint_violated_fixed_kwargs_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``bounded`` is typed ``int`` with a ``ge=0`` constraint that lives in the
            # field metadata; a negative value is the right TYPE but violates the
            # constraint, so authoring must reject it (not just type-check the value).
            resp = await _author("authored", fixed_kwargs={"bounded": -1})
            assert resp.status_code == 400
            assert "bounded" in _err(resp)

    asyncio.run(run())


def test_author_unknown_tool_name_reference_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author("authored", fixed_kwargs={"tool_names": ["echo", "ghost"]})
            assert resp.status_code == 400
            assert "ghost" in _err(resp)

    asyncio.run(run())


def test_author_unknown_preset_base_reference_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author(
                "authored",
                fixed_kwargs={"presets": [{"name": "p", "base_tool": "nope", "fixed_kwargs": {}}]},
            )
            assert resp.status_code == 400
            assert "nope" in _err(resp)

    asyncio.run(run())


def test_author_preset_typed_base_reference_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A stored preset over ``weather`` — an inline presets entry may NOT point
            # its base_tool at another preset (the flat-preset rule).
            resp0 = await presets_router.create_preset(
                _json_request(
                    "POST",
                    "/api/presets",
                    body={"name": "basep", "base_tool": "weather", "fixed_kwargs": {"units": "v"}},
                )
            )
            assert resp0.status_code == 200, _err(resp0)
            resp = await _author(
                "authored",
                fixed_kwargs={"presets": [{"name": "p", "base_tool": "basep", "fixed_kwargs": {}}]},
            )
            assert resp.status_code == 400
            assert "itself a preset" in _err(resp)

    asyncio.run(run())


def test_author_nested_subagent_unknown_reference_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A bad reference nested two levels deep must still be caught.
            resp = await _author(
                "authored",
                fixed_kwargs={
                    "subagents": [
                        {
                            "name": "outer",
                            "tool_names": ["echo"],
                            "subagents": [{"name": "inner", "tool_names": ["ghost"]}],
                        }
                    ]
                },
            )
            assert resp.status_code == 400
            assert "ghost" in _err(resp)

    asyncio.run(run())


def test_author_valid_spec_registers_and_runs_as_tool(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            resp = await _author(
                "authored",
                fixed_kwargs={
                    "system_prompt": "you are helpful",
                    "tool_names": ["echo", "weather"],
                    "presets": [{"name": "p", "base_tool": "echo", "fixed_kwargs": {}}],
                    "subagents": [{"name": "sub", "tool_names": ["echo"]}],
                },
            )
            assert resp.status_code == 200, _err(resp)
            # The authored agent is a registered tool — runnable via the tool face,
            # with the baked system_prompt mapped to system_message through run.
            result = await instance.app.tools.run_tool("authored", {"user_message": "hi"})
            assert result == "system=you are helpful"

    asyncio.run(run())


# -- name-collision guard union ----------------------------------------------


def test_author_name_collides_with_registration_name_409(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``role_agent`` is a registered agent whose run tool binds under that
            # name — a bound tool, so the existing tool-collision guard fires.
            resp = await _author("role_agent")
            assert resp.status_code == 409

    asyncio.run(run())


def test_author_name_collides_with_differing_tool_name_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # ``aliased_tool_name`` is NOT a bound tool (the run tool binds under the
            # registration name ``aliased_agent``), so only the agent-tool-name guard
            # catches it.
            assert "aliased_tool_name" not in await instance.app.tools.get_tools()
            resp = await _author("aliased_tool_name")
            assert resp.status_code == 400
            assert "agent tool name" in _err(resp)

    asyncio.run(run())


# -- authored streaming run --------------------------------------------------


def test_authored_run_combines_baked_and_request_and_maps(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "baked-sys"})
            frames = await _run_authored("authored", {"user_message": "the query"})
            assert [f["type"] for f in frames] == ["message_final", "stream.end"]
            # The baked system_prompt reached astream MAPPED as system_message (a raw
            # splat would have passed system_prompt through unmapped), and the request
            # supplied the remaining field.
            assert frames[0]["text"] == "system=baked-sys"
            agent = instance.app.agents.get_agent("authorable_agent")
            # The mapped kwargs the live agent's astream actually received — read off
            # the recorded attribute (the manifest importer may re-exec the fixture
            # module, so ``isinstance`` on the fixture class is not identity-stable).
            received = cast(Any, agent).received_kwargs
            assert received == {"system_message": "baked-sys", "user_message": "the query"}

    asyncio.run(run())


def test_authored_run_rejects_baked_field_override_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "baked-sys"})
            frames = await _run_authored("authored", {"user_message": "q", "system_prompt": "sneaky"})
            assert frames[0]["__status__"] == 400
            assert "cannot override the fixed field 'system_prompt'" in frames[0]["__error__"]

    asyncio.run(run())


def test_authored_run_rejects_unknown_input_field_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "baked-sys"})
            # A body key that is not a ``ToolInput`` field is a loud 400 naming it,
            # never a silent drop.
            frames = await _run_authored("authored", {"user_message": "q", "totally_bogus": 1})
            assert frames[0]["__status__"] == 400
            assert "unknown agent input field" in frames[0]["__error__"]
            assert "totally_bogus" in frames[0]["__error__"]

    asyncio.run(run())


def test_authored_run_invalid_input_omits_input_values(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # The baked spec can carry credentials; the runner is not necessarily the
            # author, so a validation error on the combined input must name the failing
            # field WITHOUT echoing input values back (which would leak the baked spec).
            await _author("support_bot", fixed_kwargs={"system_prompt": "s3cr3t-prompt"})

            # Field-level failure: the offending REQUEST value must not be echoed.
            field_err = (await _run_authored("support_bot", {"count": "not-an-int"}))[0]
            assert field_err["__status__"] == 400
            assert "count" in field_err["__error__"]
            assert "not-an-int" not in field_err["__error__"]

            # Model-level failure: its ``input_value`` is the WHOLE combined dict,
            # including the baked ``system_prompt`` — so this is the shape that would
            # leak the baked secret if the error echoed input. The failure is surfaced
            # (its validator message) but the baked secret is NOT.
            model_err = (await _run_authored("support_bot", {"user_message": "boom-model-error"}))[0]
            assert model_err["__status__"] == 400
            assert "model-level rejection" in model_err["__error__"]
            assert "s3cr3t-prompt" not in model_err["__error__"]

    asyncio.run(run())


def test_authored_run_request_field_mapping_to_baked_kwarg_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # The fake agent maps system_prompt -> system_message. Baking system_prompt
            # and also supplying system_message in the run body targets the SAME run
            # kwarg — the field-level override guard misses it (different field names),
            # so from_tool_input rejects the conflict as a loud 400 rather than silently
            # discarding the request's system_message.
            await _author("mapped", fixed_kwargs={"system_prompt": "baked-identity"})
            frames = await _run_authored("mapped", {"user_message": "q", "system_message": "sneaky"})
            assert frames[0]["__status__"] == 400
            assert "only one of system_prompt or system_message" in frames[0]["__error__"]

    asyncio.run(run())


def test_authored_run_streamable_from_live_registry(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # The authored agent is resolved from the PresetManager map (the live
            # registry), so the authored-run door streams it.
            resp = await _author("authored", fixed_kwargs={"system_prompt": "eph-sys"})
            assert resp.status_code == 200, _err(resp)
            assert [d["name"] for d in _non_role_documents(pg)] == ["authored"]
            frames = await _run_authored("authored", {"user_message": "q"})
            assert frames[0]["text"] == "system=eph-sys"
            assert frames[-1] == {"type": "stream.end"}

    asyncio.run(run())


def test_authored_run_unknown_name_404(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            frames = await _run_authored("ghost", {"user_message": "q"})
            assert frames[0]["__status__"] == 404

    asyncio.run(run())


def test_authored_run_over_tool_preset_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A preset over a plain (non-agent) tool is not an authored agent.
            resp = await presets_router.create_preset(
                _json_request(
                    "POST",
                    "/api/presets",
                    body={"name": "toolp", "base_tool": "weather", "fixed_kwargs": {"units": "v"}},
                )
            )
            assert resp.status_code == 200, _err(resp)
            frames = await _run_authored("toolp", {"city": "x"})
            assert frames[0]["__status__"] == 400
            assert "not an authored agent" in frames[0]["__error__"]

    asyncio.run(run())


def test_authored_run_versioning_rollback_reflected(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "v1-sys"})
            await presets_router.save_version(
                _json_request(
                    "POST",
                    "/api/presets/authored/versions",
                    name="authored",
                    body={"fixed_kwargs": {"system_prompt": "v2-sys"}},
                )
            )
            await instance.app.preset_manager.reload("authored")
            assert (await _run_authored("authored", {"user_message": "q"}))[0]["text"] == "system=v2-sys"

            await presets_router.rollback_preset(
                _json_request("POST", "/api/presets/authored/rollback", name="authored", body={"version": 1})
            )
            assert (await _run_authored("authored", {"user_message": "q"}))[0]["text"] == "system=v1-sys"

    asyncio.run(run())


# -- emit reuse (the shared preset create/delete routes, no second emit) -----


def test_author_create_and_delete_emit_exactly_once_each(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "s"})
            assert emit == ["tool"]  # create fires exactly one
            emit.clear()
            resp = await presets_router.delete_preset(_json_request("DELETE", "/api/presets/authored", name="authored"))
            assert resp.status_code == 200
            assert emit == ["tool"]  # delete fires exactly one — no second emit anywhere

    asyncio.run(run())


def test_author_bad_extension_rejected_before_write_emits_nothing(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # A valid spec but an unknown extension combo is rejected BEFORE any
            # store write (validate-then-commit): a 400, nothing persisted, no emit.
            resp = await _author("authored", fixed_kwargs={"system_prompt": "s"}, extensions=[["ghost_ext"]])
            assert resp.status_code == 400
            assert "ghost_ext" in _err(resp)
            assert _non_role_documents(pg) == []
            assert "authored" not in await instance.app.tools.get_tools()
            assert emit == []

    asyncio.run(run())


def test_author_save_version_runs_authoring_validation_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "v1"})
            emit.clear()
            # A save-version edit on an authored agent runs the SAME authoring
            # validation create runs (over the carried-forward base tool): an unknown
            # ``fixed_kwargs`` field is a 400 that commits no new version.
            resp = await presets_router.save_version(
                _json_request(
                    "POST", "/api/presets/authored/versions", name="authored", body={"fixed_kwargs": {"bogus": 1}}
                )
            )
            assert resp.status_code == 400
            assert "bogus" in _err(resp)
            versions = _data(
                await presets_router.list_versions(
                    _json_request("GET", "/api/presets/authored/versions", name="authored")
                )
            )
            assert [v["version"] for v in versions] == [1]  # nothing committed
            assert emit == []

    asyncio.run(run())


def _save_locked_version(body: dict[str, Any]) -> Request:
    return _json_request("POST", "/api/presets/locked/versions", name="locked", body=body)


def test_locked_agent_save_version_gate_rejects_bad_edits(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("locked", base_tool="locked_agent", fixed_kwargs={"secret_config": {"a": 1}})

            # Undeclared existing field -> the per-field honored gate rejects it.
            resp = await presets_router.save_version(_save_locked_version({"fixed_kwargs": {"user_message": "x"}}))
            assert resp.status_code == 400
            assert "user_message" in _err(resp)
            assert "preset_bakeable_fields" in _err(resp)

            # Unknown field.
            resp = await presets_router.save_version(_save_locked_version({"fixed_kwargs": {"bogus": 1}}))
            assert resp.status_code == 400
            assert "bogus" in _err(resp)

            # Type-invalid declared field (``secret_config`` is a dict).
            resp = await presets_router.save_version(_save_locked_version({"fixed_kwargs": {"secret_config": "nope"}}))
            assert resp.status_code == 400
            assert "secret_config" in _err(resp)

            # Nothing committed — still only version 1.
            versions = _data(
                await presets_router.list_versions(_json_request("GET", "/api/presets/locked/versions", name="locked"))
            )
            assert [v["version"] for v in versions] == [1]

    asyncio.run(run())


def test_locked_agent_save_version_valid_edit_reaches_agent(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("locked", base_tool="locked_agent", fixed_kwargs={"secret_config": {"v": 1}})
            resp = await presets_router.save_version(
                _save_locked_version({"fixed_kwargs": {"secret_config": {"v": 2}}})
            )
            assert resp.status_code == 200, _err(resp)
            await instance.app.preset_manager.reload("locked")
            # The new baked value reaches the agent on the next run.
            await instance.app.tools.run_tool("locked", {"user_message": "hi"})
            agent = instance.app.agents.get_agent("locked_agent")
            assert cast(Any, agent).received_kwargs["secret_config"] == {"v": 2}

    asyncio.run(run())


def test_locked_agent_save_version_tags_only_skips_gate(pg, emit, monkeypatch):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("locked", base_tool="locked_agent", fixed_kwargs={"secret_config": {"a": 1}})

            # A tags-only edit (no ``fixed_kwargs`` in the body) must never invoke the
            # authoring gate — pin it with a spy over ``_agent_authoring_error``. The
            # gate lives on the operation, so the seam is spied there.
            calls: list[str] = []
            real = preset_ops._agent_authoring_error

            async def spy(base_tool: str, fixed_kwargs: dict[str, Any]) -> str | None:
                calls.append(base_tool)
                return await real(base_tool, fixed_kwargs)

            monkeypatch.setattr(preset_ops, "_agent_authoring_error", spy)
            resp = await presets_router.save_version(_save_locked_version({"tags": ["blue"]}))
            assert resp.status_code == 200, _err(resp)
            assert calls == []

    asyncio.run(run())


# -- authoring spec-reference validation + authored-run body guards ----------


def _raw_run_request(name: str, raw: bytes) -> Request:
    """An authored-run POST whose body is delivered verbatim, so a malformed JSON
    payload reaches ``request.json()``."""
    scripted = [{"type": "http.request", "body": raw, "more_body": False}]
    idx = {"i": 0}

    async def receive() -> dict[str, Any]:
        i = idx["i"]
        if i < len(scripted):
            idx["i"] += 1
            return scripted[i]
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/api/agents/authored/{name}/runs",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("1.2.3.4", 1),
        "path_params": {"name": name},
    }
    return Request(scope, receive)


def test_author_invalid_preset_spec_reference_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # An inline ``presets`` entry that is not a valid ``PresetSpec`` (missing the
            # required ``base_tool``) is a loud author-time 400, never a silent drop.
            resp = await _author("authored", fixed_kwargs={"presets": [{"name": "p"}]})
            assert resp.status_code == 400
            assert "not a valid preset spec" in _err(resp)

    asyncio.run(run())


def test_authored_run_invalid_json_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "s"})
            resp = await agents_router.run_authored_agent(_raw_run_request("authored", b"not json"))
            assert resp.status_code == 400
            assert _err(resp) == "invalid JSON body"

    asyncio.run(run())


def test_authored_run_non_object_body_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            await _author("authored", fixed_kwargs={"system_prompt": "s"})
            frames = await _run_authored("authored", "scalar")
            assert frames[0]["__status__"] == 400
            assert "JSON object" in frames[0]["__error__"]

    asyncio.run(run())


def test_author_nested_malformed_references_400(pg, emit):
    async def run():
        async with instance.app.app_context(_manifest()):
            # These malformed shapes are nested INSIDE a subagent spec, so they pass the
            # agent ToolInput's typed top-level fields (subagents is list[dict]) and are
            # caught by the recursive reference check — a loud 400 at any depth, never a
            # silent drop.
            resp = await _author("authored", fixed_kwargs={"subagents": [{"name": "s", "tool_names": "notalist"}]})
            assert resp.status_code == 400
            assert "tool_names must be a list" in _err(resp)

            resp = await _author("a2", fixed_kwargs={"subagents": [{"name": "s", "tool_names": [123]}]})
            assert resp.status_code == 400
            assert "entries must be strings" in _err(resp)

            resp = await _author("a3", fixed_kwargs={"subagents": [{"name": "s", "presets": "notalist"}]})
            assert resp.status_code == 400
            assert "presets must be a list" in _err(resp)

            resp = await _author("a4", fixed_kwargs={"subagents": [{"name": "s", "subagents": "notalist"}]})
            assert resp.status_code == 400
            assert "subagents must be a list" in _err(resp)

            resp = await _author("a5", fixed_kwargs={"subagents": [{"name": "s", "subagents": [123]}]})
            assert resp.status_code == 400
            assert "must be an object" in _err(resp)

    asyncio.run(run())
