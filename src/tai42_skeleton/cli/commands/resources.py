"""``tai resources`` — load stored resources by id.

Thin wrapper over the ``/api/resources/get`` route.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_result,
    parse_kwargs,
)

app = typer.Typer(
    name="resources",
    help="Load stored resources by id.",
    no_args_is_help=True,
)


@app.command("get")
@covers(("GET", "/api/resources/get"), ("POST", "/api/resources/get"))
def get_resource_by_id(
    ctx: typer.Context,
    resource_id: Annotated[str, typer.Argument(help="Resource id (path) to load.")],
    render: Annotated[
        bool, typer.Option("--render", help="Render the resource as a Jinja template (with any --kw/--kwargs vars).")
    ] = False,
    kwargs: Annotated[
        str | None, typer.Option("--kwargs", help="Render kwargs as a JSON object (implies --render).")
    ] = None,
    kw: Annotated[
        list[str] | None, typer.Option("--kw", help="A key=value render kwarg (repeatable; implies --render).")
    ] = None,
) -> None:
    """Load a stored resource by id, optionally rendering it as a template.

    Without ``--render``/``--kw``/``--kwargs`` the loaded content is returned as-is.
    Any render var (or a bare ``--render``) renders text as a Jinja template.

    Example: ``tai resources get prompts/greeting.md --kw name=Ada``
    """
    ctx_obj = app_context(ctx)
    wants_render = render or kwargs is not None or kw
    with ctx_obj.client() as client:
        if wants_render:
            # The render path carries arbitrary nested ``template_kwargs``, which needs
            # a request body — the write-classed POST door.
            body = {"resource_id": resource_id, "template_kwargs": parse_kwargs(kwargs, kw)}
            data = client.post("/api/resources/get", json=body)
        else:
            # The plain fetch-as-is path — the read-classed GET door, so a resources
            # READ grant can fetch it.
            data = client.get("/api/resources/get", params={"resource_id": resource_id})
    emit_result(ctx_obj, data)
