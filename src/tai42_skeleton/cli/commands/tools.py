"""``tai tools`` — inspect and run registered tools, and manage their extensions.

Thin wrappers over the ``/api/tools*``, ``/api/run-tool``, ``/api/tool-runs*`` and
per-tool ``/api/tools/{name}/extensions`` routes. Per-tool extension operations
live here (the global extension catalog is ``tai extensions list``).
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
    parse_extension_combo,
    parse_kwargs,
)

app = typer.Typer(
    name="tools",
    help="Inspect and run registered tools.",
    no_args_is_help=True,
)

_KWARGS_HELP = "Tool arguments as a JSON object."
_KW_HELP = "A key=value tool argument (repeatable; value parsed as JSON)."


@app.command("list")
@covers(("GET", "/api/tools"))
def list_tools(ctx: typer.Context) -> None:
    """List the registered tool names.

    Example: ``tai tools list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/tools")
    emit_records(ctx_obj, data, ["name"])


@app.command("tags")
@covers(("GET", "/api/tools/tags"))
def tool_tags(ctx: typer.Context) -> None:
    """List each tool's native tags.

    Example: ``tai tools tags``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/tools/tags")
    emit_records(ctx_obj, data, ["name", "tags"])


@app.command("schema")
@covers(("GET", "/api/tools/{tool_name}/schema"))
def tool_schema(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Tool name.")]) -> None:
    """Get one tool's input/output schema.

    Example: ``tai tools schema my_tool``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/tools/{name}/schema")
    emit_result(ctx_obj, data)


@app.command("schemas")
@covers(("GET", "/api/tools-schema"))
def tools_schema(ctx: typer.Context) -> None:
    """Get the input/output schema of every registered tool.

    Example: ``tai tools schemas --json``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/tools-schema")
    emit_result(ctx_obj, data)


@app.command("run")
@covers(("POST", "/api/run-tool"))
def run_tool(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Tool name.")],
    kwargs: Annotated[str | None, typer.Option("--kwargs", help=_KWARGS_HELP)] = None,
    kw: Annotated[list[str] | None, typer.Option("--kw", help=_KW_HELP)] = None,
) -> None:
    """Run a registered tool synchronously and print its result.

    Example: ``tai tools run add --kw a=1 --kw b=2``
    """
    ctx_obj = app_context(ctx)
    arguments = parse_kwargs(kwargs, kw)
    with ctx_obj.client() as client:
        data = client.post("/api/run-tool", json={"tool": name, "kwargs": arguments})
    emit_result(ctx_obj, data)


_TARGET_HELP = "A worker to restrict the fan-out to (repeatable)."


@app.command("reload")
@covers(("POST", "/api/tools/reload"))
def reload_tool(
    ctx: typer.Context,
    kind: Annotated[str, typer.Argument(help='Tool kind (e.g. "flow").')],
    name: Annotated[str, typer.Argument(help="Tool name.")],
    target: Annotated[list[str] | None, typer.Option("--target", help=_TARGET_HELP)] = None,
) -> None:
    """Re-register one app tool from its stored definition, fanning out to the fleet.

    Example: ``tai tools reload flow my_flow``
    """
    ctx_obj = app_context(ctx)
    targets = list(target) if target else None
    with ctx_obj.client() as client:
        data = client.post("/api/tools/reload", json={"kind": kind, "name": name, "targets": targets})
    emit_result(ctx_obj, data)


@app.command("remove")
@covers(("POST", "/api/tools/remove"))
def remove_tool(
    ctx: typer.Context,
    kind: Annotated[str, typer.Argument(help='Tool kind (e.g. "flow").')],
    name: Annotated[str, typer.Argument(help="Tool name.")],
    target: Annotated[list[str] | None, typer.Option("--target", help=_TARGET_HELP)] = None,
) -> None:
    """Remove one app tool from the live registry, fanning out to the fleet.

    Example: ``tai tools remove flow my_flow``
    """
    ctx_obj = app_context(ctx)
    targets = list(target) if target else None
    with ctx_obj.client() as client:
        data = client.post("/api/tools/remove", json={"kind": kind, "name": name, "targets": targets})
    emit_result(ctx_obj, data)


@app.command("extensions")
@covers(("GET", "/api/tools/{name}/extensions"))
def tool_extensions(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Tool name.")]) -> None:
    """Show a tool's applied extension combos and the available catalog.

    Example: ``tai tools extensions my_tool``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/tools/{name}/extensions")
    emit_result(ctx_obj, data)


@app.command("apply")
@covers(("POST", "/api/tools/{name}/extensions"))
def apply_extensions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Tool name.")],
    combo: Annotated[
        list[str] | None,
        typer.Option(
            "--combo",
            help=(
                "One extension combo as a JSON array of elements (repeatable). Each element is an "
                'extension name (\'"chain"\') or a {"name","config"} object binding config to it '
                '(e.g. \'[{"name":"output_schema","config":{"schema":{"type":"object"}}}]\'). '
                "Passing no --combo clears the tool's extensions."
            ),
        ),
    ] = None,
) -> None:
    """Set a tool's full list of extension combos (lossless multi-combo write).

    Each ``--combo`` is one combo, written as a JSON array of extension elements;
    repeat it to author several combos at once. An element is a bare extension
    name or a ``{"name", "config"}`` object binding author config, so per-element
    config (e.g. an ``output_schema`` combo's schema) round-trips losslessly.
    Passing no ``--combo`` clears every combo.

    Example: ``tai tools apply my_tool --combo '["chain","batch"]' --combo '["chain"]'``
    """
    ctx_obj = app_context(ctx)
    combos = [parse_extension_combo(chain, param_hint="--combo") for chain in (combo or [])]
    with ctx_obj.client() as client:
        data = client.post(f"/api/tools/{name}/extensions", json={"combos": combos})
    emit_result(ctx_obj, data)


# -- Background tool runs (``tai tools runs ...``) ---------------------------

runs_app = typer.Typer(
    name="runs",
    help="Submit and inspect background (detached) tool runs.",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")


@runs_app.command("submit")
@covers(("POST", "/api/tool-runs"))
def submit_run(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Tool name.")],
    kwargs: Annotated[str | None, typer.Option("--kwargs", help=_KWARGS_HELP)] = None,
    kw: Annotated[list[str] | None, typer.Option("--kw", help=_KW_HELP)] = None,
) -> None:
    """Submit a tool for background execution and print its run id.

    Example: ``tai tools runs submit slow_tool --kw n=100``
    """
    ctx_obj = app_context(ctx)
    arguments = parse_kwargs(kwargs, kw)
    with ctx_obj.client() as client:
        data = client.post("/api/tool-runs", json={"tool_name": name, "arguments": arguments})
    emit_result(ctx_obj, data)


@runs_app.command("get")
@covers(("GET", "/api/tool-runs/{run_id}"))
def get_run(ctx: typer.Context, run_id: Annotated[str, typer.Argument(help="Run id.")]) -> None:
    """Get a background tool run's status and result.

    Example: ``tai tools runs get abc123``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/tool-runs/{run_id}")
    emit_result(ctx_obj, data)


@runs_app.command("list")
@covers(("GET", "/api/tool-runs"))
def list_runs(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Tool name.")]) -> None:
    """List the recent background runs for a tool.

    Example: ``tai tools runs list slow_tool``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/tool-runs", params={"tool_name": name})
    emit_records(ctx_obj, data, ["run_id", "tool_name", "status", "started_at", "finished_at"])
