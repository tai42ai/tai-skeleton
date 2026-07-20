"""Callback glue — chain a tool after a backend tool runs.

:class:`CallbackSchema` is the impl half of :class:`tai_contract.backend.CallbackSchema`:
it adds the ``rendered_*`` methods (via the template render mixins) to the contract
field shape. ``callback_execution`` evaluates the rendered condition over a tool
result and, when it passes (an empty condition always passes), renders an
expression and optionally runs a follow-up tool. It returns the follow-up tool's
result, or the rendered expression output when no tool is set, or ``None`` when
the condition fails. ``prepare_backend_kwargs`` strips the FastMCP context from a tool's kwargs and
injects the tool name before a backend dispatch.
"""

from __future__ import annotations

from typing import Any

from tai_contract.app import TaiApp
from tai_contract.backend import CallbackSchema as CallbackFields
from tai_kit.utils.data import run_jq_first
from tai_kit.utils.lc.signature_util import exclude_fastmcp_ctx_from_kwargs

from tai_skeleton.template import ConditionMixin, ExprMixin


class CallbackSchema(CallbackFields, ConditionMixin, ExprMixin):
    """Impl half of the contract ``CallbackSchema``: the contract field shape
    (including ``tool``) plus the skeleton render mixins' ``rendered_*`` methods."""


async def prepare_backend_kwargs(func, tool_name_arg, tool_name, kwargs):
    kwargs = exclude_fastmcp_ctx_from_kwargs(func, kwargs)
    kwargs[tool_name_arg] = tool_name
    return kwargs


async def callback_execution(
    result: Any,
    callback: CallbackSchema,
    app: TaiApp,
) -> Any:
    cond = await callback.rendered_condition()
    if cond:
        cond_output = await run_jq_first(cond, result)
        if not cond_output:
            return None

    expr = await callback.rendered_expr()
    # An absent/empty expr is not an error — ``get_compiled_jq("")`` would raise,
    # so guard like the hooks path and yield an empty mapping.
    expr_output = (await run_jq_first(expr, result)) if expr else {}

    if callback.tool:
        return await app.tools.run_tool(callback.tool, expr_output)
    return expr_output
