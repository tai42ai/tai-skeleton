"""``tai sub-mcp`` — register and inspect sub-MCP apps.

Thin wrappers over the ``/api/sub-mcp*`` routes.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_result

app = typer.Typer(
    name="sub-mcp",
    help="Register and inspect sub-MCP apps.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/sub-mcp"))
def list_sub_mcp(ctx: typer.Context) -> None:
    """List the registered sub-MCP apps and their tools.

    Example: ``tai sub-mcp list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/sub-mcp")
    emit_result(ctx_obj, data)


@app.command("register")
@covers(("POST", "/api/sub-mcp"))
def register_sub_mcp(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Slug to mount the sub-MCP app under.")],
    tool: Annotated[list[str], typer.Option("--tool", help="A tool name to expose (repeatable).")],
) -> None:
    """Register (or reload) a sub-MCP app exposing the named tools under a slug.

    Example: ``tai sub-mcp register billing --tool invoice --tool refund``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/sub-mcp", json={"slug": slug, "tools": list(tool)})
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/sub-mcp/{slug}"))
def delete_sub_mcp(ctx: typer.Context, slug: Annotated[str, typer.Argument(help="Sub-MCP app slug.")]) -> None:
    """Unregister a sub-MCP app.

    Example: ``tai sub-mcp delete billing``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/sub-mcp/{slug}")
    emit_result(ctx_obj, data)
