"""The preset bind kernel — the single point every preset builds its live tool
through.

``preset_bind`` transforms a base tool into a new named tool via ONE FastMCP
``Tool.from_tool`` call: each ``fixed_kwargs`` key is baked as a HIDDEN, FIXED
constant (``ArgTransform(hide=True, default=<value>)`` — removed from the exposed
input schema, and a caller that passes it is rejected; it cannot be overridden at
runtime), while the REMAINING arguments keep the base tool's real typed schema
(names, types, descriptions), NOT one opaque ``params`` blob. The preset's
``description`` and categorization ``tags`` are set on the transformed tool, so a
bind re-applies them from the stored body every time.

Bake through the PROGRAMMATIC ``transform_args`` path (whose ``default`` accepts
any value, incl. dict/list) rather than a declarative scalar-only path, so a
non-scalar baked value is preserved.

An author-set ``output_schema`` (an object JSON Schema) dispatches on the base's
kind. When the base is an AGENT run tool, the schema is baked into the run tool's
``response_format`` (a hidden, fixed constant, with the preset ``name`` injected as
its ``title`` when absent — the agent run seam requires a title, and the preset
name is validated to the provider structured-output name charset) so the agent
FORCES a structured output; the preset advertises the authored (title-free)
schema. When the base is a plain tool, the schema is advertised as the bound
tool's output schema and every result is validated against it at run time with
tai42-kit's faithful draft-2020-12 validator, raising loudly on any mismatch — no
forcing is possible for a non-LLM tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform, forward
from tai42_kit.utils.data.json_schema_util import validate_against_json_schema

if TYPE_CHECKING:
    from tai42_contract.app import TaiApp


async def preset_bind(
    app: TaiApp,
    base_tool: str,
    fixed_kwargs: dict[str, Any],
    *,
    name: str,
    description: str = "",
    tags: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Tool:
    """Return a FastMCP tool transform of ``base_tool`` as the new named tool ``name``.

    Resolves the base ``Tool`` object (``app.tools.get_tool`` — hence async), then
    builds the transform in one call. Each ``fixed_kwargs`` key becomes a hidden,
    fixed constant; the remaining arguments keep the base tool's typed schema. An
    ``output_schema`` dispatches on the base kind (agent → bake ``response_format``;
    plain tool → advertise + validate-and-raise).
    """
    base = await app.tools.get_tool(base_tool)
    transform_args = {key: ArgTransform(hide=True, default=value) for key, value in fixed_kwargs.items()}

    if output_schema is None:
        return Tool.from_tool(
            base,
            name=name,
            description=description,
            tags=set(tags or []),
            transform_args=transform_args,
        )

    if base_tool in app.agents.all_agents():
        # Agent base — FORCE structured output: bake ``response_format`` from the
        # authored schema. The agent run seam requires a top-level ``title``; when
        # the author left it off, inject the preset name. The advertised output
        # schema stays the authored (title-free) value; the agent's own drain
        # validates the forced result, so no second validation wrapper is attached.
        baked_response_format = dict(output_schema)
        baked_response_format.setdefault("title", name)
        transform_args["response_format"] = ArgTransform(hide=True, default=baked_response_format)
        return Tool.from_tool(
            base,
            name=name,
            description=description,
            tags=set(tags or []),
            transform_args=transform_args,
            output_schema=output_schema,
        )

    # Plain tool — DECLARE + VALIDATE: advertise the authored schema and validate
    # every result against it, raising loudly on any mismatch (a non-LLM tool
    # cannot be forced).
    async def _enforce_output_schema(**kwargs: Any) -> Any:
        result = await forward(**kwargs)
        validate_against_json_schema(result.structured_content, output_schema)
        return result

    return Tool.from_tool(
        base,
        name=name,
        description=description,
        tags=set(tags or []),
        transform_args=transform_args,
        output_schema=output_schema,
        transform_fn=_enforce_output_schema,
    )
