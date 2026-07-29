"""``TAI_MCP_*`` config for the skeleton's MCP dispatch seam.

Shared across the two layers of a managed MCP tool call: the connector call layer
(``connectors.token_injection.call_with_auth`` hands ``call_timeout_seconds`` to
fastmcp so a downstream that accepts a request then stalls cannot hang the caller
forever) and the tool-adapter layer (``tools.adapters.mcp_tool_to_func`` and
``tools.binding`` read ``schema_max_depth`` when converting an advertised tool's
JSON schema). It lives in the neutral ``settings`` package so both layers depend
on it downward. Read at call time through the ``settings_cache`` accessor so a
``.env`` override applied by the CLI bootstrap is honoured.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings, settings_cache


class MCPDispatchSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_MCP_")

    # Wall-clock budget for one downstream MCP tool call. Bounds a downstream that
    # accepts the request then stalls; a long-running tool needs the operator to
    # raise this, never an unbounded wait. Must be positive.
    call_timeout_seconds: float = Field(default=300, gt=0)

    # Max JSON-schema nesting depth the converter accepts when adapting an
    # advertised MCP tool's input/output schema. A hostile/buggy server can
    # advertise a deeply-nested schema that blows the recursion; the converter
    # raises past this bound and the single tool is skipped (see
    # ``ToolBinding.mcp_tools``), never taking down the binding pass. Must be
    # positive.
    schema_max_depth: int = Field(default=50, gt=0)


@settings_cache
def mcp_dispatch_settings() -> MCPDispatchSettings:
    return MCPDispatchSettings()
