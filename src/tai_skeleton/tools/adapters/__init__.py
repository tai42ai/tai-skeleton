"""Tool-dispatch glue: convert a vendor tool (langchain ``BaseTool``, MCP
``mcp.Tool``) into a callable with a synthesized signature. The generic
JSON-Schema → pydantic converter these adapters use lives in
``tai_kit.utils.data.json_schema_util``.
"""

from tai_skeleton.tools.adapters.lc_tool_to_func import lc_tool_to_func
from tai_skeleton.tools.adapters.mcp_tool_to_func import (
    mcp_tool_call_wrapper,
    mcp_tool_to_func,
)

__all__ = [
    "lc_tool_to_func",
    "mcp_tool_call_wrapper",
    "mcp_tool_to_func",
]
