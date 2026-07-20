"""``tai scopes`` — manage access-control scopes.

Thin wrappers over the authed ``/api/auth/scopes*`` routes.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import app_context, covers, emit_records, emit_result

app = typer.Typer(
    name="scopes",
    help="Manage access-control scopes.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/auth/scopes"))
def list_scopes(ctx: typer.Context) -> None:
    """List every non-public route mapping as ``{url: scope_id}``.

    Example: ``tai scopes list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/scopes")
    emit_result(ctx_obj, data)


@app.command("add")
@covers(("POST", "/api/auth/scopes"))
def add_scope_url(
    ctx: typer.Context,
    scope_id: Annotated[str, typer.Argument(help="Scope id.")],
    url: Annotated[str, typer.Argument(help="URL to map into the scope.")],
    pattern: Annotated[str | None, typer.Option("--pattern", help="Optional dynamic match pattern.")] = None,
) -> None:
    """Map a URL to a scope (optionally with a dynamic match pattern).

    Example: ``tai scopes add read /api/tools``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"scope_id": scope_id, "url": url}
    if pattern is not None:
        body["pattern"] = pattern
    with ctx_obj.client() as client:
        data = client.post("/api/auth/scopes", json=body)
    emit_result(ctx_obj, data)


@app.command("remove-url")
@covers(("DELETE", "/api/auth/scopes/urls"))
def remove_scope_url(
    ctx: typer.Context, url: Annotated[str, typer.Argument(help="URL to unmap from every scope.")]
) -> None:
    """Remove a URL from every scope that references it.

    Example: ``tai scopes remove-url /api/tools``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.request("DELETE", "/api/auth/scopes/urls", json={"url": url})
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/auth/scopes/{scope_id}"))
def delete_scope(ctx: typer.Context, scope_id: Annotated[str, typer.Argument(help="Scope id.")]) -> None:
    """Delete a scope.

    Example: ``tai scopes delete read``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/auth/scopes/{scope_id}")
    emit_result(ctx_obj, data)


@app.command("routes")
@covers(("GET", "/api/auth/routes"))
def list_routes(ctx: typer.Context) -> None:
    """List the app's HTTP routes with each route's scope mapping.

    A ``mapped`` of ``null`` marks an unassigned route. Example: ``tai scopes routes``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/routes")
    emit_records(ctx_obj, data, ["path", "methods", "mapped"])


@app.command("public-list")
@covers(("GET", "/api/auth/public-routes"))
def list_public_routes(ctx: typer.Context) -> None:
    """List every route pinned to the public marker.

    Example: ``tai scopes public-list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/public-routes")
    emit_result(ctx_obj, data)


@app.command("public-pin")
@covers(("POST", "/api/auth/public-routes"))
def pin_public_route(
    ctx: typer.Context,
    url: Annotated[str, typer.Argument(help="URL to pin public.")],
    pattern: Annotated[str | None, typer.Option("--pattern", help="Optional dynamic match pattern.")] = None,
) -> None:
    """Pin a URL public (optionally with a dynamic match pattern).

    Example: ``tai scopes public-pin /universal_webhook/orders``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"url": url}
    if pattern is not None:
        body["pattern"] = pattern
    with ctx_obj.client() as client:
        data = client.post("/api/auth/public-routes", json=body)
    emit_result(ctx_obj, data)


@app.command("public-unpin")
@covers(("DELETE", "/api/auth/public-routes"))
def unpin_public_route(ctx: typer.Context, url: Annotated[str, typer.Argument(help="URL to unpin.")]) -> None:
    """Unpin a public URL.

    Example: ``tai scopes public-unpin /universal_webhook/orders``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.request("DELETE", "/api/auth/public-routes", json={"url": url})
    emit_result(ctx_obj, data)
