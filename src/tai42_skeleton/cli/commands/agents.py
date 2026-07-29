"""``tai agents`` — list and run agents.

Thin wrappers over the ``/api/agents*`` routes. The run doors are SSE streams, so
``run`` and ``authored-run`` render each ``StreamEvent`` frame as it arrives.
Creating an authored agent is a preset over a spec-runnable agent — ``tai presets
create --base-tool <agent>`` writes it through the shared preset route.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    parse_json_object,
    stream_frames,
)

app = typer.Typer(
    name="agents",
    help="List and run agents.",
    no_args_is_help=True,
)

_LIST_COLUMNS = ["name", "tool_name", "spec_runnable", "description"]


@app.command("list")
@covers(("GET", "/api/agents"))
def list_agents(ctx: typer.Context) -> None:
    """List every registered agent.

    Example: ``tai agents list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/agents")
    emit_records(ctx_obj, data, _LIST_COLUMNS, items_key="items")


@app.command("spec-runnable")
@covers(("GET", "/api/agents/spec-runnable"))
def list_spec_runnable(ctx: typer.Context) -> None:
    """List the spec-runnable (authorable) agents.

    Example: ``tai agents spec-runnable``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/agents/spec-runnable")
    emit_records(ctx_obj, data, _LIST_COLUMNS, items_key="items")


@app.command("run")
@covers(("POST", "/api/agents/{name}/runs"))
def run_agent(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Agent registration name.")],
    input_json: Annotated[str, typer.Option("--input", help="The agent input as a JSON object.")],
) -> None:
    """Stream a run of an agent, one event frame at a time.

    Example: ``tai agents run researcher --input '{"query":"weather"}'``
    """
    ctx_obj = app_context(ctx)
    body = parse_json_object(input_json, param_hint="--input")
    stream_frames(ctx_obj, "POST", f"/api/agents/{name}/runs", json_body=body)


@app.command("authored-run")
@covers(("POST", "/api/agents/authored/{name}/runs"))
def run_authored_agent(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Authored-agent (preset) name.")],
    input_json: Annotated[str, typer.Option("--input", help="The non-baked agent input as a JSON object.")],
) -> None:
    """Stream a run of an authored agent (a preset baked over a spec-runnable agent).

    Example: ``tai agents authored-run my_researcher --input '{"query":"weather"}'``
    """
    ctx_obj = app_context(ctx)
    body = parse_json_object(input_json, param_hint="--input")
    stream_frames(ctx_obj, "POST", f"/api/agents/authored/{name}/runs", json_body=body)
