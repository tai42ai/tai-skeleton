"""``tai plugins`` — browse and install marketplace plugins.

Thin wrappers over the eight ``/api/marketplace/*`` routes: the five reads
(``search``, ``info``, ``categories``, ``installed``, ``advisories``) and the
three environment-mutating flows (``install``, ``uninstall``, ``update``). Each
command declares the exact registered route it invokes via ``@covers`` so the
CLI↔route parity gate proves every marketplace route is reachable from the
terminal.

``install``/``uninstall``/``update`` run arbitrary third-party code in the serving
environment by design; typed :class:`ApiError` results (a 409 collision, a 502
registry failure, a 503 in-progress) surface through the root group's handler, so
these commands add no error handling of their own.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from tai_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
)

app = typer.Typer(
    name="plugins",
    help="Browse and install marketplace plugins.",
    no_args_is_help=True,
)


def _split_ref(ref: str) -> tuple[str, str]:
    """Split a ``namespace/name`` ref into its two halves, raising a usage error
    on anything that is not exactly one non-empty namespace and one non-empty
    name separated by a single ``/``."""
    namespace, sep, name = ref.partition("/")
    if not sep or not namespace or not name or "/" in name:
        raise typer.BadParameter("REF must be 'namespace/name'", param_hint="REF")
    return namespace, name


@app.command("search")
@covers(("GET", "/api/marketplace/search"))
def search(
    ctx: typer.Context,
    query: Annotated[str | None, typer.Argument(help="Free-text search query; omit to browse all.")] = None,
    kind: Annotated[str | None, typer.Option("--kind", help="Filter by provides kind (tool, agent, ...).")] = None,
    category: Annotated[str | None, typer.Option("--category", help="Filter by category.")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Filter by tag (repeatable).")] = None,
    namespace: Annotated[str | None, typer.Option("--namespace", help="Filter by publisher namespace.")] = None,
    tier: Annotated[str | None, typer.Option("--tier", help="Filter by trust tier.")] = None,
    contract: Annotated[str | None, typer.Option("--contract", help="Filter by compatible contract version.")] = None,
    sort: Annotated[str | None, typer.Option("--sort", help="Sort order (downloads, updated, name).")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Result page (1-based).")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Results per page.")] = None,
) -> None:
    """Search the marketplace registry, optionally filtered by facets.

    Example: ``tai plugins search uuid --kind tool``
    """
    ctx_obj = app_context(ctx)
    params: dict[str, Any] = {}
    if query is not None:
        params["q"] = query
    if kind is not None:
        params["kind"] = kind
    if category is not None:
        params["category"] = category
    if tag:
        # Each ``--tag`` becomes its own repeated ``tags`` query param — the client
        # encodes a list value as repeated params, never comma-joined.
        params["tags"] = tag
    if namespace is not None:
        params["namespace"] = namespace
    if tier is not None:
        params["tier"] = tier
    if contract is not None:
        params["contract"] = contract
    if sort is not None:
        params["sort"] = sort
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    with ctx_obj.client() as client:
        data = client.get("/api/marketplace/search", params=params)
    emit_records(
        ctx_obj,
        data,
        ["ref", "display_name", "latest_version", "trust_tier", "downloads"],
        items_key="items",
    )


@app.command("info")
@covers(("GET", "/api/marketplace/plugins/{ns}/{name}"))
def info(ctx: typer.Context, ref: Annotated[str, typer.Argument(help="Plugin ref 'namespace/name'.")]) -> None:
    """Show one listing's detail and its published versions.

    Example: ``tai plugins info tai42/toolbox``
    """
    ctx_obj = app_context(ctx)
    namespace, name = _split_ref(ref)
    with ctx_obj.client() as client:
        data = client.get(f"/api/marketplace/plugins/{namespace}/{name}")
    emit_result(ctx_obj, data)


@app.command("categories")
@covers(("GET", "/api/marketplace/categories"))
def categories(ctx: typer.Context) -> None:
    """List the marketplace's controlled category vocabulary.

    Example: ``tai plugins categories``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/marketplace/categories")
    emit_records(ctx_obj, data, ["category"])


@app.command("installed")
@covers(("GET", "/api/marketplace/installed"))
def installed(ctx: typer.Context) -> None:
    """List the installed marketplace plugins and their update availability.

    Example: ``tai plugins installed``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/marketplace/installed")
    emit_records(ctx_obj, data, ["ref", "version", "latest", "update_available", "installed_at"])


@app.command("install")
@covers(("POST", "/api/marketplace/install"))
def install(
    ctx: typer.Context,
    ref: Annotated[str, typer.Argument(help="Plugin ref 'namespace/name'.")],
    version: Annotated[str | None, typer.Option("--version", help="Pin a specific version; omit for latest.")] = None,
) -> None:
    """Install a marketplace plugin by ref, optionally pinning a version.

    Example: ``tai plugins install tai42/toolbox``
    """
    ctx_obj = app_context(ctx)
    body: dict[str, Any] = {"ref": ref}
    if version is not None:
        body["version"] = version
    with ctx_obj.client() as client:
        data = client.post("/api/marketplace/install", json=body)
    emit_result(ctx_obj, data)


@app.command("uninstall")
@covers(("POST", "/api/marketplace/uninstall"))
def uninstall(ctx: typer.Context, ref: Annotated[str, typer.Argument(help="Plugin ref 'namespace/name'.")]) -> None:
    """Uninstall a marketplace-installed plugin by ref.

    Example: ``tai plugins uninstall tai42/toolbox``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post("/api/marketplace/uninstall", json={"ref": ref})
    emit_result(ctx_obj, data)


@app.command("update")
@covers(("POST", "/api/marketplace/update"))
def update(
    ctx: typer.Context,
    ref: Annotated[str, typer.Argument(help="Plugin ref 'namespace/name'.")],
    version: Annotated[str | None, typer.Option("--version", help="Target version; omit for latest.")] = None,
) -> None:
    """Update an installed plugin to a newer (or named) version.

    Example: ``tai plugins update tai42/toolbox``
    """
    ctx_obj = app_context(ctx)
    body: dict[str, Any] = {"ref": ref}
    if version is not None:
        body["version"] = version
    with ctx_obj.client() as client:
        data = client.post("/api/marketplace/update", json=body)
    emit_result(ctx_obj, data)


@app.command("advisories")
@covers(("GET", "/api/marketplace/advisories"))
def advisories(ctx: typer.Context) -> None:
    """Show the cached advisory snapshot for the installed plugins.

    Example: ``tai plugins advisories``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/marketplace/advisories")
    emit_records(
        ctx_obj,
        data,
        ["listing", "severity", "summary", "affected_versions", "withdrawn_at"],
        items_key="advisories",
    )
