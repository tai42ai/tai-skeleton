"""``tai templates`` — manage prompt and resource templates.

Thin wrappers over the ``/api/templates``, ``/api/template``, ``/api/upload-template``,
``/api/delete-template``, ``/api/render-template`` and ``/api/clear-templates-cache``
routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_result,
    parse_kwargs,
)

app = typer.Typer(
    name="templates",
    help="Manage prompt and resource templates.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/templates"))
def list_templates(ctx: typer.Context) -> None:
    """List the available templates.

    Example: ``tai templates list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/templates")
    emit_result(ctx_obj, data)


@app.command("get")
@covers(("POST", "/api/template"))
def get_template(ctx: typer.Context, template_id: Annotated[str, typer.Argument(help="Template id.")]) -> None:
    """Fetch a template's content and its input schema.

    Example: ``tai templates get prompts/greeting.md``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/template", json={"template_id": template_id})
    emit_result(ctx_obj, data)


@app.command("upload")
@covers(("POST", "/api/upload-template"))
def upload_template(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Template key to write.")],
    file: Annotated[
        Path,
        typer.Option("--file", exists=True, dir_okay=False, readable=True, help="Local file whose content to upload."),
    ],
) -> None:
    """Upload (create or overwrite) a template from a local file.

    Example: ``tai templates upload prompts/greeting.md --file greeting.md``
    """
    ctx_obj = app_context(ctx)
    content = file.read_text(encoding="utf-8")
    with ctx_obj.client() as client:
        data = client.post("/api/upload-template", json={"path": path, "content": content})
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("POST", "/api/delete-template"))
def delete_template(ctx: typer.Context, path: Annotated[str, typer.Argument(help="Template key to delete.")]) -> None:
    """Delete a template.

    Example: ``tai templates delete prompts/greeting.md``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/delete-template", json={"path": path})
    emit_result(ctx_obj, data)


@app.command("render")
@covers(("POST", "/api/render-template"))
def render_template(
    ctx: typer.Context,
    template_id: Annotated[str | None, typer.Option("--template-id", help="A stored template id to render.")] = None,
    content: Annotated[str | None, typer.Option("--content", help="Inline template content to render.")] = None,
    kwargs: Annotated[str | None, typer.Option("--kwargs", help="Render kwargs as a JSON object.")] = None,
    kw: Annotated[
        list[str] | None, typer.Option("--kw", help="A key=value render kwarg (repeatable; value parsed as JSON).")
    ] = None,
) -> None:
    """Render a template by id or inline content with kwargs.

    Example: ``tai templates render --template-id prompts/greeting.md --kw name=Ada``
    """
    ctx_obj = app_context(ctx)
    if (template_id is None) == (content is None):
        raise typer.BadParameter("provide exactly one of --template-id or --content")
    body: dict = {"kwargs": parse_kwargs(kwargs, kw)}
    if template_id is not None:
        body["template_id"] = template_id
    else:
        body["content"] = content
    with ctx_obj.client() as client:
        data = client.post("/api/render-template", json=body)
    emit_result(ctx_obj, data)


@app.command("clear-cache")
@covers(("POST", "/api/clear-templates-cache"))
def clear_cache(ctx: typer.Context) -> None:
    """Clear the template render cache.

    Example: ``tai templates clear-cache``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/clear-templates-cache")
    emit_result(ctx_obj, data)
