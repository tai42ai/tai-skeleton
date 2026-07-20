"""``tai presets`` — manage preset tools and their versions.

Thin wrappers over the ``/api/presets*`` routes. A preset whose ``base_tool`` is a
spec-runnable agent's run tool is an authored agent; ``create`` writes both.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
    parse_extension_combos,
    parse_json_object,
)

app = typer.Typer(
    name="presets",
    help="Manage preset tools and their versions.",
    no_args_is_help=True,
)

_LIST_COLUMNS = ["name", "base_tool", "active_version", "conflicted"]

_EXTENSIONS_HELP = (
    "Extension combos as a JSON array of combos. Each combo is a non-empty array of elements, and an "
    'element is an extension name (\'"chain"\') or a {"name","config"} object binding config to it '
    '(e.g. \'[[{"name":"output_schema","config":{"schema":{"type":"object"}}}]]\').'
)
_EXTENSIONS_HELP_CLEARABLE = f"{_EXTENSIONS_HELP} '[]' clears them."


@app.command("list")
@covers(("GET", "/api/presets"))
def list_presets(ctx: typer.Context) -> None:
    """List the store-backed (versioned) presets.

    Example: ``tai presets list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/presets")
    emit_records(ctx_obj, data, _LIST_COLUMNS)


@app.command("get")
@covers(("GET", "/api/presets/{name}"))
def get_preset(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Preset name.")]) -> None:
    """Get a store-backed preset's record and active body.

    Example: ``tai presets get my_preset``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/presets/{name}")
    emit_result(ctx_obj, data)


@app.command("create")
@covers(("POST", "/api/presets"))
def create_preset(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    base_tool: Annotated[str, typer.Option("--base-tool", help="The registered non-preset tool the preset wraps.")],
    kwargs: Annotated[str | None, typer.Option("--kwargs", help="Baked fixed kwargs as a JSON object.")] = None,
    description: Annotated[str, typer.Option("--description", help="Human description.")] = "",
    tags: Annotated[list[str] | None, typer.Option("--tag", help="A tag (repeatable).")] = None,
    extensions: Annotated[str | None, typer.Option("--extensions", help=_EXTENSIONS_HELP)] = None,
) -> None:
    """Create a versioned preset (or an authored agent, when the base tool is an agent).

    Example: ``tai presets create greet --base-tool echo --kwargs '{"prefix":"hi"}'``
    """
    ctx_obj = app_context(ctx)
    body: dict = {
        "name": name,
        "base_tool": base_tool,
        "description": description,
        "fixed_kwargs": parse_json_object(kwargs, param_hint="--kwargs") if kwargs is not None else {},
        "tags": list(tags or []),
    }
    if extensions is not None:
        body["extensions"] = parse_extension_combos(extensions, param_hint="--extensions")
    with ctx_obj.client() as client:
        data = client.post("/api/presets", json=body)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/presets/{name}"))
def delete_preset(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Preset name.")]) -> None:
    """Delete a preset.

    Example: ``tai presets delete my_preset``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/presets/{name}")
    emit_result(ctx_obj, data)


@app.command("versions")
@covers(("GET", "/api/presets/{name}/versions"))
def list_versions(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Preset name.")]) -> None:
    """List a preset's version history.

    Example: ``tai presets versions my_preset``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/presets/{name}/versions")
    emit_records(ctx_obj, data, ["version", "created_at"])


@app.command("get-version")
@covers(("GET", "/api/presets/{name}/versions/{version}"))
def get_version(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    version: Annotated[int, typer.Argument(help="Version number.")],
) -> None:
    """Get a specific preset version.

    Example: ``tai presets get-version my_preset 3``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/presets/{name}/versions/{version}")
    emit_result(ctx_obj, data)


@app.command("save-version")
@covers(("POST", "/api/presets/{name}/versions"))
def save_version(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    kwargs: Annotated[
        str | None, typer.Option("--kwargs", help="New fixed kwargs (JSON object); omit to carry forward.")
    ] = None,
    tags: Annotated[
        list[str] | None, typer.Option("--tag", help="A tag (repeatable); replaces the tag set when given.")
    ] = None,
    clear_tags: Annotated[
        bool, typer.Option("--clear-tags", help="Clear all tags (send an explicit empty tag list).")
    ] = False,
    extensions: Annotated[str | None, typer.Option("--extensions", help=_EXTENSIONS_HELP_CLEARABLE)] = None,
) -> None:
    """Save a new preset version. Omitted fields carry forward; ``--clear-tags`` and
    ``--extensions '[]'`` send the explicit clear sentinel.

    Example: ``tai presets save-version my_preset --kwargs '{"n":2}' --clear-tags``
    """
    ctx_obj = app_context(ctx)
    if clear_tags and tags:
        raise typer.BadParameter("pass either --tag or --clear-tags, not both", param_hint="--clear-tags")
    body: dict = {}
    if kwargs is not None:
        body["fixed_kwargs"] = parse_json_object(kwargs, param_hint="--kwargs")
    if clear_tags:
        body["tags"] = []
    elif tags:
        body["tags"] = list(tags)
    if extensions is not None:
        body["extensions"] = parse_extension_combos(extensions, param_hint="--extensions")
    if not body:
        raise typer.BadParameter("provide at least one of --kwargs, --tag/--clear-tags, or --extensions")
    with ctx_obj.client() as client:
        data = client.post(f"/api/presets/{name}/versions", json=body)
    emit_result(ctx_obj, data)


@app.command("rollback")
@covers(("POST", "/api/presets/{name}/rollback"))
def rollback_preset(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    version: Annotated[int, typer.Argument(help="Target version to make active.")],
) -> None:
    """Roll a preset back to a prior version.

    Example: ``tai presets rollback my_preset 2``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/presets/{name}/rollback", json={"version": version})
    emit_result(ctx_obj, data)


@app.command("rename")
@covers(("POST", "/api/presets/{name}/rename"))
def rename_preset(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Current preset name.")],
    new_name: Annotated[str, typer.Argument(help="New preset name.")],
) -> None:
    """Rename a preset — its name is its live tool name, so the tool is rebound.

    Example: ``tai presets rename old_name new_name``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/presets/{name}/rename", json={"new_name": new_name})
    emit_result(ctx_obj, data)


@app.command("referees")
@covers(("GET", "/api/presets/{name}/referees"))
def preset_referees(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Preset name.")]) -> None:
    """List the presets that reference this preset (a rename would strand them).

    Example: ``tai presets referees my_preset``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/presets/{name}/referees")
    emit_result(ctx_obj, data)


@app.command("validate")
@covers(("POST", "/api/presets/validate"))
def validate_preset(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    base_tool: Annotated[
        str | None, typer.Option("--base-tool", help="The base tool (required for a new preset).")
    ] = None,
    kwargs: Annotated[str | None, typer.Option("--kwargs", help="Baked fixed kwargs as a JSON object.")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Human description.")] = None,
    tags: Annotated[list[str] | None, typer.Option("--tag", help="A tag (repeatable).")] = None,
    extensions: Annotated[str | None, typer.Option("--extensions", help=_EXTENSIONS_HELP)] = None,
) -> None:
    """Dry-run a preset draft — report whether it would be accepted as a create (a
    new name) or a new version (an existing name), without writing anything.

    Example: ``tai presets validate greet --base-tool echo --kwargs '{"prefix":"hi"}'``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"name": name}
    if base_tool is not None:
        body["base_tool"] = base_tool
    if kwargs is not None:
        body["fixed_kwargs"] = parse_json_object(kwargs, param_hint="--kwargs")
    if description is not None:
        body["description"] = description
    if tags:
        body["tags"] = list(tags)
    if extensions is not None:
        body["extensions"] = parse_extension_combos(extensions, param_hint="--extensions")
    with ctx_obj.client() as client:
        data = client.post("/api/presets/validate", json=body)
    emit_result(ctx_obj, data)


@app.command("set-version-tags")
@covers(("PUT", "/api/presets/{name}/versions/{version}/tags"))
def set_version_tags(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Preset name.")],
    version: Annotated[int, typer.Argument(help="Version number.")],
    tags: Annotated[list[str] | None, typer.Argument(help="The tags to set; none clears them to [].")] = None,
) -> None:
    """Replace a preset version's tags (labels only — no rebind). Zero tag arguments
    clears them to ``[]``.

    Example: ``tai presets set-version-tags my_preset 2 stable reviewed``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.put(f"/api/presets/{name}/versions/{version}/tags", json={"tags": list(tags or [])})
    emit_result(ctx_obj, data)
