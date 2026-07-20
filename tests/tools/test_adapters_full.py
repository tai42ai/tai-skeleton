"""Full dispatch-adapter coverage.

``mcp_tool_to_func`` / ``mcp_tool_call_wrapper`` are driven end-to-end with a
faked ``FastMCPClient`` at the adapter's import seam (no network), covering the
schema-building helpers, transport detection, the non-managed call path, and the
error-response annotation branch. ``lc_tool_to_func`` is covered for the model,
schemaless, async/sync, output-schema, and dict-schema-reject branches.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import ClassVar, cast
from unittest.mock import patch

import mcp.types
import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tai_contract.manifest import MCPConfig, TaiMCPConfig

from tai_skeleton.tools.adapters.lc_tool_to_func import lc_tool_to_func
from tai_skeleton.tools.adapters.mcp_tool_to_func import (
    _build_input_model,
    _build_input_value,
    _build_output_schema,
    _detect_transport,
    mcp_tool_call_wrapper,
    mcp_tool_to_func,
)

_CLIENT = "tai_skeleton.tools.adapters.mcp_tool_to_func.FastMCPClient"


# -- faked client -------------------------------------------------------------


class _FakeMcpClient:
    """Yields a stub session per ``current(config=...)`` enter whose
    ``call_tool_mcp`` returns the single scripted response."""

    def __init__(self, response):
        self._response = response
        self.captured_configs: list = []
        self.captured_args: list = []

    def current(self, *, config):
        self.captured_configs.append(config)
        outer = self

        class _Session:
            async def __aenter__(self):
                return self

            @staticmethod
            async def __aexit__(*exc):
                return False

            @staticmethod
            async def call_tool_mcp(name, arguments, meta=None, timeout=None):
                outer.captured_args.append((name, arguments, meta))
                return outer._response

        return _Session()


def _ok_result(text: str = '{"result": "done"}'):
    return mcp.types.CallToolResult(content=[mcp.types.TextContent(type="text", text=text)], isError=False)


def _err_result(text: str = "boom"):
    return mcp.types.CallToolResult(content=[mcp.types.TextContent(type="text", text=text)], isError=True)


def _http_config():
    return TaiMCPConfig(title="srv", config=MCPConfig(type="http", url="http://x/mcp"))


# -- schema helpers -----------------------------------------------------------


class _FakeTool:
    name = "lookup"
    description = "look up"
    inputSchema: ClassVar[dict] = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    outputSchema: ClassVar[dict] = {}


def test_build_input_model_defaults_when_no_schema():
    class _NoSchema:
        inputSchema = None

    model = _build_input_model(_NoSchema(), 50)
    assert issubclass(model, BaseModel)
    # Empty object schema -> a model with no required fields.
    assert model().model_dump() == {}


def test_build_output_schema_unwraps_result_envelope():
    tool_schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "$defs": {"X": {"type": "object"}},
    }

    class _T:
        outputSchema = tool_schema

    out = _build_output_schema(_T())
    # The inner ``result`` schema is lifted out and the parent ``$defs`` is
    # carried onto it.
    assert out["type"] == "string"
    assert out["$defs"] == {"X": {"type": "object"}}


def test_build_output_schema_envelope_without_defs():
    class _T:
        outputSchema: ClassVar[dict] = {"type": "object", "properties": {"result": {"type": "number"}}}

    # Result-enveloped but no parent ``$defs`` -> inner schema lifted as-is.
    assert _build_output_schema(_T()) == {"type": "number"}


def test_build_output_schema_passthrough_when_not_enveloped():
    class _T:
        outputSchema: ClassVar[dict] = {"type": "integer"}

    assert _build_output_schema(_T()) == {"type": "integer"}


def test_build_signature_carries_field_default():
    # A field WITH a default flows its default into the synthesized parameter
    # (the non-PydanticUndefined branch of ``_build_signature``); a required
    # field maps to ``Parameter.empty``.
    from tai_skeleton.tools.adapters.mcp_tool_to_func import _build_signature

    class _M(BaseModel):
        q: str
        n: int = 7

    sig = _build_signature(_M, {}, 50)
    assert sig.parameters["q"].default is inspect.Parameter.empty
    assert sig.parameters["n"].default == 7


def test_build_input_value_applies_alias_and_drops_unset():
    class _M(BaseModel):
        q: str = Field(alias="query")
        n: int | None = None

    # Callers pass by field name; the dump emits the alias and drops the unset
    # Optional ``n``.
    out = _build_input_value(_M, q="hi")
    assert out == {"query": "hi"}


# -- transport detection ------------------------------------------------------


def test_detect_transport_each_kind():
    assert _detect_transport(MCPConfig(type="http", url="http://x")) == "http"
    assert _detect_transport(MCPConfig(type="stdio", command="uvx")) == "stdio"
    assert _detect_transport(MCPConfig(type="uds", uds="/tmp/s.sock")) == "uds"


def test_detect_transport_none_raises():
    with pytest.raises(RuntimeError, match="no transport"):
        _detect_transport(MCPConfig(type=None))


def test_detect_transport_ambiguous_raises():
    cfg = MCPConfig.model_construct(type="http", url="http://x", command="uvx")
    with pytest.raises(RuntimeError, match="ambiguous"):
        _detect_transport(cfg)


# -- mcp_tool_to_func + wrapper (non-managed) ---------------------------------


def test_mcp_tool_to_func_builds_signature():
    # ``mcp.types.Tool`` is a concrete pydantic model the adapter only reads a few
    # attributes off; the fake matches those, so cast it to the declared type.
    func = mcp_tool_to_func(_http_config(), cast(mcp.types.Tool, _FakeTool()), name="lookup", module="srv")
    assert callable(func)
    assert set(inspect.signature(func).parameters) == {"q"}
    assert func.__name__ == "lookup"


def test_wrapper_non_managed_dispatches_and_extracts_output():
    client = _FakeMcpClient(_ok_result())

    async def go():
        with patch(_CLIENT, return_value=client):
            return await mcp_tool_call_wrapper(
                _http_config(), "lookup", _build_input_model(_FakeTool(), 50), {"q": "x"}
            )

    out = asyncio.run(go())
    assert out == {"result": "done"}
    # Non-managed -> no _meta token injected.
    assert client.captured_args[0][2] is None


def test_mcp_tool_to_func_call_drives_wrapper():
    # Calling the built function runs its inner ``func_impl`` -> the wrapper.
    client = _FakeMcpClient(_ok_result('{"result": 1}'))
    func = mcp_tool_to_func(_http_config(), cast(mcp.types.Tool, _FakeTool()), name="lookup", module="srv")
    with patch(_CLIENT, return_value=client):
        out = asyncio.run(func(q="hi"))
    assert out == {"result": 1}
    assert client.captured_args[0][0] == "lookup"


def test_wrapper_managed_token_expired_force_refreshes_and_retries():
    # Managed OAuth entry: a first ``token_expired`` sentinel drives the
    # force-refresh + retry-once branch (the ``is_managed and access_token``
    # gate plus ``handle_token_expired``).
    from unittest.mock import AsyncMock

    from tai_contract.connectors.models import ConnectorRef

    from tai_skeleton.connectors.runtime.resolver import ManagedAuth
    from tai_skeleton.connectors.token_injection import CONNECTOR_ERROR_PREFIX

    conn = "11111111-1111-1111-1111-111111111111"
    managed = TaiMCPConfig(
        title="srv",
        config=MCPConfig(type="http", url="http://x/mcp", headers={}),
        managed=ConnectorRef(connection_id=conn, provider_id="g", sub_service="m"),
    )

    expired = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=f'{CONNECTOR_ERROR_PREFIX}{{"code": "token_expired"}}')],
        isError=True,
    )

    class _RetryClient:
        def __init__(self):
            self._responses = [expired, _ok_result()]
            self.calls = 0

        def current(self, *, config):
            outer = self

            class _S:
                async def __aenter__(self):
                    return self

                @staticmethod
                async def __aexit__(*exc):
                    return False

                @staticmethod
                async def call_tool_mcp(name, arguments, meta=None, timeout=None):
                    outer.calls += 1
                    return outer._responses.pop(0)

            return _S()

    client = _RetryClient()
    resolver = AsyncMock(return_value=ManagedAuth(access_token="t1"))
    refresher = AsyncMock(return_value=ManagedAuth(access_token="t2"))

    async def go():
        with (
            patch(_CLIENT, return_value=client),
            patch("tai_skeleton.connectors.token_injection.resolve_managed_auth", new=resolver),
            patch("tai_skeleton.connectors.token_injection.force_refresh", new=refresher),
        ):
            return await mcp_tool_call_wrapper(managed, "lookup", _build_input_model(_FakeTool(), 50), {"q": "x"})

    out = asyncio.run(go())
    assert out == {"result": "done"}
    assert client.calls == 2  # original + one retry
    refresher.assert_awaited_once()


def test_wrapper_managed_no_expiry_skips_retry():
    # Managed OAuth entry whose first call succeeds: the ``is_managed and
    # access_token`` gate is true but ``is_token_expired`` is false, so the
    # retry branch is skipped (the 144->152 false edge).
    from unittest.mock import AsyncMock

    from tai_contract.connectors.models import ConnectorRef

    from tai_skeleton.connectors.runtime.resolver import ManagedAuth

    managed = TaiMCPConfig(
        title="srv",
        config=MCPConfig(type="http", url="http://x/mcp", headers={}),
        managed=ConnectorRef(
            connection_id="11111111-1111-1111-1111-111111111111",
            provider_id="g",
            sub_service="m",
        ),
    )
    client = _FakeMcpClient(_ok_result())
    resolver = AsyncMock(return_value=ManagedAuth(access_token="t1"))
    refresher = AsyncMock()

    async def go():
        with (
            patch(_CLIENT, return_value=client),
            patch("tai_skeleton.connectors.token_injection.resolve_managed_auth", new=resolver),
            patch("tai_skeleton.connectors.token_injection.force_refresh", new=refresher),
        ):
            return await mcp_tool_call_wrapper(managed, "lookup", _build_input_model(_FakeTool(), 50), {"q": "x"})

    assert asyncio.run(go()) == {"result": "done"}
    refresher.assert_not_awaited()


def test_wrapper_error_response_annotates_span_and_returns():
    client = _FakeMcpClient(_err_result("upstream failed"))
    recorded = {}

    class _Writer:
        def update_current_span(self, level, status_message):
            recorded["level"] = level
            recorded["msg"] = status_message

    class _Mon:
        writer = _Writer()

    async def go():
        with (
            patch(_CLIENT, return_value=client),
            patch("tai_skeleton.tools.adapters.mcp_tool_to_func.get_monitoring", return_value=_Mon()),
        ):
            return await mcp_tool_call_wrapper(
                _http_config(), "lookup", _build_input_model(_FakeTool(), 50), {"q": "x"}
            )

    result = asyncio.run(go())
    # The error path annotates the active span with the extracted error text.
    assert "upstream failed" in recorded["msg"]
    # An error response carries no usable output: it is handed back unchanged.
    assert result is client._response
    assert result.isError is True


# -- lc_tool_to_func ----------------------------------------------------------


class _EchoArgs(BaseModel):
    x: int
    y: str = "hi"
    tags: list[str] = Field(default_factory=list)


class _EchoTool(BaseTool):
    name: str = "echo"
    description: str = "echo args"
    args_schema: type = _EchoArgs

    def _run(self, x: int, y: str = "hi", tags=()):
        return {"x": x, "y": y, "tags": list(tags)}


def test_lc_tool_model_schema_sync_with_default_factory():
    func = lc_tool_to_func(_EchoTool(), async_mode=False)
    params = inspect.signature(func).parameters
    assert set(params) == {"x", "y", "tags"}
    # A default_factory field synthesizes its real default ([]), not None.
    assert params["tags"].default == []
    assert func(x=1) == {"x": 1, "y": "hi", "tags": []}


def test_lc_tool_async_mode_runs():
    func = lc_tool_to_func(_EchoTool(), async_mode=True)
    assert asyncio.run(func(x=5, y="z")) == {"x": 5, "y": "z", "tags": []}


class _NoSchemaTool(BaseTool):
    name: str = "raw"
    description: str = "passes the raw input string"
    # No ``args_schema`` declared -> BaseTool leaves it None (schemaless).

    def _run(self, tool_input):
        return f"got:{tool_input}"


def test_lc_tool_schemaless_uses_input_param():
    func = lc_tool_to_func(_NoSchemaTool(), async_mode=False)
    params = inspect.signature(func).parameters
    assert set(params) == {"input"}
    assert func(input="abc") == "got:abc"


def test_lc_tool_with_output_schema_sets_return_annotation():
    out_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    func = lc_tool_to_func(_EchoTool(), async_mode=False, output_schema=out_schema)
    assert inspect.signature(func).return_annotation is not None


def test_lc_tool_rejects_dict_args_schema():
    class _DictSchemaTool(BaseTool):
        name: str = "bad"
        description: str = "dict schema not supported"
        args_schema: dict = {"type": "object", "properties": {}}  # noqa: RUF012

        def _run(self, **kw):
            return kw

    with pytest.raises(TypeError, match="non-model args_schema"):
        lc_tool_to_func(_DictSchemaTool())


class _AliasArgs(BaseModel):
    q: str = Field(alias="query")


class _AliasTool(BaseTool):
    name: str = "aliased"
    description: str = "echo the aliased field"
    args_schema: type = _AliasArgs

    def _run(self, **kwargs):
        return kwargs


def test_build_signature_aliased_field_keyed_by_field_name():
    # ``build_signature`` iterates ``model_fields`` and keys each parameter by the
    # FIELD name (not the alias), so an aliased field binds correctly.
    from tai_skeleton.tools.adapters.lc_tool_to_func import build_signature

    sig = build_signature(_AliasArgs)
    assert set(sig.parameters) == {"q"}


def test_lc_tool_aliased_field_binds_without_keyerror():
    # Building the adapter over an aliased-schema tool keys the synthesized
    # signature by the FIELD name (not the alias).
    func = lc_tool_to_func(_AliasTool(), async_mode=False)
    assert set(inspect.signature(func).parameters) == {"q"}


class _NoneArgs(BaseModel):
    n: int | None = 5


class _NoneTool(BaseTool):
    name: str = "noner"
    description: str = "echo n"
    args_schema: type = _NoneArgs

    def _run(self, n: int | None = 5):
        return {"n": n}


def test_lc_tool_preserves_explicit_none():
    func = lc_tool_to_func(_NoneTool(), async_mode=False)
    # An explicitly-passed ``None`` (differing from the default 5) is preserved,
    # not silently dropped as ``exclude_none`` would.
    assert func(n=None) == {"n": None}
    # A value equal to the default drops from the forwarded dict; the tool fills it.
    assert func(n=5) == {"n": 5}
