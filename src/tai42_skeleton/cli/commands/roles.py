"""``tai roles`` — manage access-control roles.

Thin wrappers over the authed ``/api/auth/roles*`` routes. Create/edit take the per-tag
grant map as repeatable ``--grant <tag>=<level>`` options; the server validates the tags
+ levels + base tier (a bad grant fails loudly server-side).
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_records, emit_result

app = typer.Typer(
    name="roles",
    help="Manage access-control roles.",
    no_args_is_help=True,
)


def _parse_grants(grants: list[str]) -> dict[str, str]:
    """Parse repeatable ``--grant tag=level`` options into a grant map. A malformed
    entry fails loudly here (the server also validates the tag/level)."""
    parsed: dict[str, str] = {}
    for item in grants:
        tag, sep, level = item.partition("=")
        if not sep or not tag:
            raise typer.BadParameter(f"--grant must be 'tag=level', got {item!r}")
        parsed[tag] = level
    return parsed


@app.command("list")
@covers(("GET", "/api/auth/roles"))
def list_roles(ctx: typer.Context) -> None:
    """List every role with its base tier, grant map, and allow_all flag.

    Example: ``tai roles list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/roles")
    emit_records(ctx_obj, data, ["name", "base_tier", "allow_all", "grants", "description"])


@app.command("show")
def show_role(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Role name.")]) -> None:
    """Show one role's full definition (filtered from the roles listing).

    Example: ``tai roles show editor``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/roles")
    match = next((role for role in data if role.get("name") == name), None)
    if match is None:
        raise typer.BadParameter(f"unknown role: {name!r}")
    emit_result(ctx_obj, match)


@app.command("create")
@covers(("POST", "/api/auth/roles"))
def create_role(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="New role name.")],
    base_tier: Annotated[str, typer.Option("--base-tier", help="Base tier: editor or viewer.")],
    grant: Annotated[list[str] | None, typer.Option("--grant", help="Repeatable tag=level (none/read/write).")] = None,
    description: Annotated[str, typer.Option("--description", help="Role description.")] = "",
) -> None:
    """Create a role over a base tier with a per-tag grant map.

    Example: ``tai roles create ops --base-tier editor --grant presets=write --grant hooks=read``
    """
    ctx_obj = app_context(ctx)
    body = {
        "name": name,
        "base_tier": base_tier,
        "description": description,
        "grants": _parse_grants(grant or []),
    }
    with ctx_obj.client() as client:
        data = client.post("/api/auth/roles", json=body)
    emit_result(ctx_obj, data)


@app.command("edit")
@covers(("PUT", "/api/auth/roles/{name}"))
def edit_role(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Role name.")],
    grant: Annotated[list[str] | None, typer.Option("--grant", help="Repeatable tag=level (none/read/write).")] = None,
    description: Annotated[str | None, typer.Option("--description", help="New description.")] = None,
) -> None:
    """Edit a role's per-tag grant map (and optionally its description). LIVE — every
    holder's reach changes on their next request.

    Example: ``tai roles edit ops --grant hooks=write``
    """
    ctx_obj = app_context(ctx)
    # Omit-means-keep: send ``grants`` only when ``--grant`` was passed (else the server
    # preserves the stored grant map), and ``description`` only when ``--description`` was
    # passed. A description-only edit therefore never wipes the grant map.
    body: dict = {}
    if grant is not None:
        body["grants"] = _parse_grants(grant)
    if description is not None:
        body["description"] = description
    with ctx_obj.client() as client:
        data = client.request("PUT", f"/api/auth/roles/{name}", json=body)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/auth/roles/{name}"))
def delete_role(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Role name.")]) -> None:
    """Delete a role (rejected while any principal still holds it).

    Example: ``tai roles delete ops``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/auth/roles/{name}")
    emit_result(ctx_obj, data)


@app.command("versions")
@covers(("GET", "/api/auth/roles/{name}/versions"))
def list_versions(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Role name.")]) -> None:
    """Show a role's version history and its who/when/before→after audit trail.

    Example: ``tai roles versions ops``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/auth/roles/{name}/versions")
    emit_result(ctx_obj, data)


@app.command("rollback")
@covers(("POST", "/api/auth/roles/{name}/rollback"))
def rollback_role(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Role name.")],
    version: Annotated[int, typer.Argument(help="Version number to roll back to.")],
) -> None:
    """Roll a role back to a prior version (LIVE — holders follow on their next request).

    Example: ``tai roles rollback ops 2``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/auth/roles/{name}/rollback", json={"version": version})
    emit_result(ctx_obj, data)
