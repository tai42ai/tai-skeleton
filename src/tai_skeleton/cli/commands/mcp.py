"""``tai mcp`` — inspect and configure mounted MCP servers.

Thin wrappers over the ``/api/mcp-config*`` and ``/api/mcp-status*`` routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import app_context, covers, emit_result

app = typer.Typer(
    name="mcp",
    help="Inspect and configure mounted MCP servers.",
    no_args_is_help=True,
)


@app.command("status")
@covers(("GET", "/api/mcp-status"))
def mcp_status(ctx: typer.Context) -> None:
    """Snapshot the live MCP binding status.

    Example: ``tai mcp status``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/mcp-status")
    emit_result(ctx_obj, data)


@app.command("schema")
@covers(("GET", "/api/mcp-config/schema"))
def mcp_config_schema(ctx: typer.Context) -> None:
    """Get the JSON schema for one MCP-config entry.

    Example: ``tai mcp schema``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/mcp-config/schema")
    emit_result(ctx_obj, data)


@app.command("set")
@covers(("POST", "/api/mcp-config"))
def set_mcp_config(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Option(
            "--file", exists=True, dir_okay=False, readable=True, help="A JSON file with the full 'mcp' list."
        ),
    ],
) -> None:
    """Replace the MCP config section from a JSON file and hot-reload.

    The file is the full replacement MCP list, either a bare JSON array or an
    object carrying an ``"mcp"`` list.

    Example: ``tai mcp set --file mcp.json``
    """
    ctx_obj = app_context(ctx)
    try:
        parsed = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"file must be valid JSON: {exc}", param_hint="--file") from exc
    if isinstance(parsed, dict) and "mcp" in parsed:
        mcp = parsed["mcp"]
    elif isinstance(parsed, list):
        mcp = parsed
    else:
        raise typer.BadParameter("file must be a JSON list or an object with an 'mcp' list", param_hint="--file")
    with ctx_obj.client() as client:
        data = client.post("/api/mcp-config", json={"mcp": mcp})
    emit_result(ctx_obj, data)


def _targets_body(target: list[str] | None) -> dict[str, list[str] | None]:
    return {"targets": list(target) if target else None}


@app.command("reload")
@covers(("POST", "/api/mcp-status/{title}/reload"))
def reload_mcp(
    ctx: typer.Context,
    title: Annotated[str, typer.Argument(help="MCP server title.")],
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the fan-out to (repeatable).")
    ] = None,
) -> None:
    """Reload a single MCP server by title.

    Without ``--target`` the response is the single-worker status; with one or more
    the confirmed fleet fan-out is awaited and per-worker results are returned.

    Example: ``tai mcp reload my-server``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/mcp-status/{title}/reload", json=_targets_body(target))
    emit_result(ctx_obj, data)


@app.command("failed")
@covers(("GET", "/api/mcp-status/failed"))
def list_failed_mcps(
    ctx: typer.Context,
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the listing to (repeatable).")
    ] = None,
) -> None:
    """List the MCP servers skipped by the viability check (down/slow at boot or reload).

    Example: ``tai mcp failed``
    """
    ctx_obj = app_context(ctx)
    params = {"targets": list(target)} if target else None
    with ctx_obj.client() as client:
        data = client.get("/api/mcp-status/failed", params=params)
    emit_result(ctx_obj, data)


@app.command("reload-failed")
@covers(("POST", "/api/mcp-status/reload-failed"))
def reload_failed_mcps(
    ctx: typer.Context,
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the fan-out to (repeatable).")
    ] = None,
) -> None:
    """Re-probe every failed MCP server and attach the ones now viable.

    Example: ``tai mcp reload-failed``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/mcp-status/reload-failed", json=_targets_body(target))
    emit_result(ctx_obj, data)


@app.command("deregister")
@covers(("POST", "/api/mcp-status/{title}/deregister"))
def deregister_mcp(
    ctx: typer.Context,
    title: Annotated[str, typer.Argument(help="MCP server title.")],
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the fan-out to (repeatable).")
    ] = None,
) -> None:
    """Detach a single MCP server's tools by title (the removal counterpart of reload).

    Example: ``tai mcp deregister my-server``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/mcp-status/{title}/deregister", json=_targets_body(target))
    emit_result(ctx_obj, data)
