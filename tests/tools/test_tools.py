"""Tools feature: registry register/resolve, the three dispatch adapters, and a
no-per-tenant-scoping guard.
"""

import inspect
from typing import ClassVar, Protocol, cast

import mcp.types
import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from tai42_contract.manifest import MCPConfig, TaiMCPConfig
from tai42_kit.utils.data.json_schema_util import json_schema_to_pydantic_model

from tai42_skeleton.tools import (
    ToolRegistry,
    lc_tool_to_func,
    mcp_tool_to_func,
)

# -- registry -----------------------------------------------------------------


def test_registry_registers_and_resolves_tool():
    # A base tool selected with an attached extension combo (seeded through the
    # constructor's ``tool_extensions`` map): the bare present-check combo plus
    # the attachment combo.
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})

    assert list(reg.tool_extensions_iterator("foo")) == [[], ["ext1"]]
    assert reg.used_extensions == frozenset({"ext1"})

    # ``foo`` has no extend-tool backing it, so it is reported missing until one
    # is registered (drives ``validation``).
    assert "foo" in reg.missing_tools
    reg.register_extend_tool(tool_name="foo", extend_tool_name="foo_ext")
    assert "foo" not in reg.missing_tools


def test_registry_unregister_drops_tool():
    reg = ToolRegistry({"foo"}, {"foo": [["ext1"]]})
    reg.unregister_tool_base("foo")
    assert reg.missing_tools == frozenset()


# -- json-schema -> pydantic (via the kit util) -------------------------------


class _SampleInstance(Protocol):
    name: str
    count: int | None


def test_json_schema_to_pydantic_via_kit_util():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name"],
    }
    model = json_schema_to_pydantic_model(schema, "Sample")

    assert issubclass(model, BaseModel)
    # The model is built dynamically from the schema, so its fields aren't
    # statically known; narrow to the shape the schema declares.
    instance = cast(_SampleInstance, model(name="x"))
    assert instance.name == "x"
    assert instance.count is None
    with pytest.raises(ValidationError):
        model()  # ``name`` is required


# -- lc tool -> func ----------------------------------------------------------


class _EchoArgs(BaseModel):
    x: int
    y: str = "hi"


class _EchoTool(BaseTool):
    name: str = "echo"
    description: str = "echo the args back"
    args_schema: type = _EchoArgs

    def _run(self, x: int, y: str = "hi"):
        return {"x": x, "y": y}


def test_lc_tool_to_func_converts_and_runs():
    func = lc_tool_to_func(_EchoTool(), async_mode=False)

    params = inspect.signature(func).parameters
    assert set(params) == {"x", "y"}

    assert func(x=1) == {"x": 1, "y": "hi"}
    assert func(x=2, y="bye") == {"x": 2, "y": "bye"}


# -- mcp tool -> func ---------------------------------------------------------


class _FakeMcpTool:
    name = "lookup"
    description = "look something up"
    inputSchema: ClassVar[dict] = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    outputSchema: ClassVar[dict] = {}


def test_mcp_tool_to_func_builds_callable():
    config = TaiMCPConfig(title="srv", config=MCPConfig(url="http://example/mcp"))
    # ``mcp.types.Tool`` is a concrete pydantic model; the fake supplies the
    # attributes the adapter reads, so cast it to the declared parameter type.
    func = mcp_tool_to_func(config, cast(mcp.types.Tool, _FakeMcpTool()), name="lookup", module="srv")

    assert callable(func)
    params = inspect.signature(func).parameters
    assert set(params) == {"q"}


# -- no per-tenant scoping -----------------------------------------------------


def test_no_per_tenant_scoping():
    # The registry keys only on tool name + extension, never a tenant/client/org
    # partition.
    reg_params = set(inspect.signature(ToolRegistry.__init__).parameters) - {"self"}
    assert reg_params == {"requested_tools", "tool_extensions"}

    # The dispatch adapters take only the tool + its plumbing, no tenant routing.
    for adapter in (lc_tool_to_func, mcp_tool_to_func):
        names = set(inspect.signature(adapter).parameters)
        assert not (names & {"tenant", "tenant_id", "client", "client_id", "org", "org_id"})
