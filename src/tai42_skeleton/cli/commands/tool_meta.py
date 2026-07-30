"""``tai tool-meta`` — manage the tool-metadata overlay (folders + per-tool rows).

Thin wrappers over the ``/api/tool-meta*`` routes. The overlay is unversioned
organizational metadata over ANY tool (native, preset, flow): a display name, a
folder placement, user tags, and a tri-state visibility. The per-tool ``set``
command mirrors the API's MERGE-PATCH: only the flags you pass are sent, so a
field you omit is left unchanged. A CLEAR flag sends the explicit reset — a
present-null (``--clear-display-name`` / ``--clear-folder``) or a present-empty
array (``--clear-tags``) — distinct from omitting the flag entirely.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_result,
)

app = typer.Typer(
    name="tool-meta",
    help="Manage the tool-metadata overlay (folders, display names, tags, visibility).",
    no_args_is_help=True,
)

# ``--visibility`` maps a three-value choice onto the tri-state ``hidden`` overlay:
# defer to the plugin declaration, force shown, or force hidden.
_VISIBILITY_TO_HIDDEN: dict[str, bool | None] = {"default": None, "shown": False, "hidden": True}


@app.command("list")
@covers(("GET", "/api/tool-meta"))
def list_tool_meta(ctx: typer.Context) -> None:
    """Show the whole overlay — every folder and every per-tool row.

    Example: ``tai tool-meta list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/tool-meta")
    emit_result(ctx_obj, data)


@app.command("set")
@covers(("PATCH", "/api/tool-meta/tools/{tool_name}"))
def set_tool_meta(
    ctx: typer.Context,
    tool_name: Annotated[str, typer.Argument(help="The tool whose overlay to edit.")],
    display_name: Annotated[
        str | None, typer.Option("--display-name", help="Set the display label (non-empty).")
    ] = None,
    clear_display_name: Annotated[
        bool, typer.Option("--clear-display-name", help="Clear the display label (defer to the tool name).")
    ] = False,
    folder: Annotated[str | None, typer.Option("--folder", help="Place the tool in this folder id.")] = None,
    clear_folder: Annotated[bool, typer.Option("--clear-folder", help="Remove the tool from its folder.")] = False,
    tags: Annotated[
        list[str] | None, typer.Option("--tag", help="A user tag (repeatable); replaces the whole set when given.")
    ] = None,
    clear_tags: Annotated[bool, typer.Option("--clear-tags", help="Clear all user tags (send an empty set).")] = False,
    visibility: Annotated[
        str | None,
        typer.Option("--visibility", help="Visibility override: default (defer) | shown | hidden."),
    ] = None,
) -> None:
    """Merge-patch a tool's overlay. Only the flags you pass are sent; omit a field
    to leave it unchanged.

    Example: ``tai tool-meta set web_search --display-name 'Web Search' --tag research``
    """
    ctx_obj = app_context(ctx)
    if display_name is not None and clear_display_name:
        raise typer.BadParameter("pass either --display-name or --clear-display-name, not both")
    if folder is not None and clear_folder:
        raise typer.BadParameter("pass either --folder or --clear-folder, not both")
    if tags and clear_tags:
        raise typer.BadParameter("pass either --tag or --clear-tags, not both")

    body: dict[str, Any] = {}
    if display_name is not None:
        body["display_name"] = display_name
    elif clear_display_name:
        body["display_name"] = None
    if folder is not None:
        body["folder_id"] = folder
    elif clear_folder:
        body["folder_id"] = None
    if tags:
        body["tags"] = list(tags)
    elif clear_tags:
        body["tags"] = []
    if visibility is not None:
        if visibility not in _VISIBILITY_TO_HIDDEN:
            raise typer.BadParameter("--visibility must be one of: default, shown, hidden")
        body["hidden"] = _VISIBILITY_TO_HIDDEN[visibility]

    if not body:
        raise typer.BadParameter(
            "provide at least one field to set (--display-name/--clear-display-name, "
            "--folder/--clear-folder, --tag/--clear-tags, or --visibility)"
        )
    with ctx_obj.client() as client:
        data = client.patch(f"/api/tool-meta/tools/{tool_name}", json=body)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/tool-meta/tools/{tool_name}"))
def delete_tool_meta(
    ctx: typer.Context, tool_name: Annotated[str, typer.Argument(help="The tool whose overlay row to drop.")]
) -> None:
    """Delete a tool's whole overlay row (idempotent).

    Example: ``tai tool-meta delete web_search``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/tool-meta/tools/{tool_name}")
    emit_result(ctx_obj, data)


@app.command("folder-create")
@covers(("POST", "/api/tool-meta/folders"))
def create_folder(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Folder name.")],
    parent: Annotated[str | None, typer.Option("--parent", help="Parent folder id; omit for a root folder.")] = None,
) -> None:
    """Create a folder (root, or under ``--parent``).

    Example: ``tai tool-meta folder-create Research``
    """
    ctx_obj = app_context(ctx)
    body: dict[str, Any] = {"name": name}
    if parent is not None:
        body["parent_id"] = parent
    with ctx_obj.client() as client:
        data = client.post("/api/tool-meta/folders", json=body)
    emit_result(ctx_obj, data)


@app.command("folder-rename")
@covers(("POST", "/api/tool-meta/folders/{folder_id}/rename"))
def rename_folder(
    ctx: typer.Context,
    folder_id: Annotated[str, typer.Argument(help="Folder id.")],
    name: Annotated[str, typer.Argument(help="New folder name.")],
) -> None:
    """Rename a folder.

    Example: ``tai tool-meta folder-rename <id> Archive``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/tool-meta/folders/{folder_id}/rename", json={"name": name})
    emit_result(ctx_obj, data)


@app.command("folder-move")
@covers(("POST", "/api/tool-meta/folders/{folder_id}/move"))
def move_folder(
    ctx: typer.Context,
    folder_id: Annotated[str, typer.Argument(help="Folder id.")],
    parent: Annotated[
        str | None, typer.Option("--parent", help="New parent folder id; omit to move to the root.")
    ] = None,
) -> None:
    """Re-parent a folder (omit ``--parent`` to move it to the root).

    Example: ``tai tool-meta folder-move <id> --parent <parent-id>``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/tool-meta/folders/{folder_id}/move", json={"parent_id": parent})
    emit_result(ctx_obj, data)


@app.command("folder-delete")
@covers(("DELETE", "/api/tool-meta/folders/{folder_id}"))
def delete_folder(
    ctx: typer.Context, folder_id: Annotated[str, typer.Argument(help="Folder id (must be empty).")]
) -> None:
    """Delete an empty folder.

    Example: ``tai tool-meta folder-delete <id>``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/tool-meta/folders/{folder_id}")
    emit_result(ctx_obj, data)
