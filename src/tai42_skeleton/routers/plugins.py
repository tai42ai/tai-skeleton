"""Studio-plugin registry + asset serving + SPA hosting.

Three surfaces, split by trust (the unauthenticated surface is limited to
callback doors + static Studio UI assets; everything carrying data stays authed):

- ``GET /api/plugins`` (AUTHED) — lists installed Studio plugins' manifest
  CONTENTS from the startup-built registry. ``Cache-Control: no-cache``.
- ``GET /api/plugins/{name}/studio/{path}`` (PUBLIC) — serves a plugin's
  ``studio/`` dist files, but ONLY those in the plugin's integrity set: strict
  realpath-under-root traversal defense, the served path checked against the
  plugin's SRI-pinned served URLs (an unlisted, un-scanned file 404s), explicit
  content-type mapping (never OS guessing, never ``text/html``), the
  ``studio-manifest.json`` file excluded on the realpath basename, long-lived
  caching for content-hashed filenames.
- ``GET /{path}`` (PUBLIC, SPA history fallback) — serves the built Studio SPA:
  static files as-is, every other GET falling back to ``index.html`` with the
  security headers, the per-response CSP nonce, and the injected import map.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.

ORDERING (load-bearing): the SPA catch-all matches ANY path, and FastMCP matches
custom routes in registration (import) order, so this router MUST be listed LAST
in ``manifest.routers_modules`` — otherwise its catch-all shadows sibling routes
registered after it. The catch-all guards ``/api`` / ``/mcp`` paths (returns 404,
never index.html) so a misorder surfaces as a visible 404 on those routes rather
than a silent wrong-serve, but correct ordering is the contract.
"""

from __future__ import annotations

from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.plugins import list_studio_plugins as _list_studio_plugins_op
from tai42_skeleton.plugins.registry import (
    STUDIO_MANIFEST_FILENAME,
    StudioPluginError,
    current_registry,
    resolve_under,
)
from tai42_skeleton.plugins.serving import (
    HTML_CONTENT_TYPE,
    asset_content_type,
    generate_nonce,
    inject_importmap,
    security_headers,
)
from tai42_skeleton.plugins.settings import plugins_settings

_HASHED_ASSET_CACHE = "public, max-age=31536000, immutable"
_REVALIDATE_CACHE = "no-cache"
_NO_STORE = "no-store"
_INDEX_FILENAME = "index.html"
_VENDOR_PREFIX = "vendor/"


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


# -- Registry listing (AUTHED) -----------------------------------------------
#
# A thin adapter over ``operations.plugins.list_studio_plugins``. The revalidate
# cache directive rides the success response via the adapter's ``response_headers``
# seam (an HTTP-edge caching concern), so the operation stays a plain data read.


list_studio_plugins = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_studio_plugins_op),
    path="/api/plugins",
    method="GET",
    response_headers={"cache-control": _REVALIDATE_CACHE},
)


# -- Plugin asset serving (PUBLIC) -------------------------------------------


@http_surface().custom_route(
    "/api/plugins/{name}/studio/{path:path}",
    methods=["GET"],
    summary="Serve a studio plugin asset",
    tags=["plugins"],
    response_model=None,
    authed=False,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(404, 500),
        success_status=200,
    ),
)
async def serve_studio_asset(request: Request) -> Response:
    name = request.path_params["name"]
    rel = request.path_params["path"]
    try:
        registry = current_registry()
    except StudioPluginError as exc:
        return _error(str(exc), 500)
    plugin = registry.plugins.get(name)
    if plugin is None:
        return _error(f"unknown studio plugin: {name!r}", 404)
    try:
        target = resolve_under(plugin.dist_root, rel)
    except StudioPluginError:
        return _error("not found", 404)
    # Exclude the structured manifest on the REALPATH basename, so a
    # normalization bypass (``sub/../studio-manifest.json``) that stays under-root
    # cannot leak it.
    if target.name == STUDIO_MANIFEST_FILENAME:
        return _error("not found", 404)
    # Serve ONLY integrity-pinned assets: reachable files are exactly the ones in
    # this plugin's served-URL integrity set (the same map the import map SRI-pins
    # and the load-time byte-scan covers). An unlisted file shipped in the dist —
    # un-hashed and un-scanned — resolves under-root but is not served, so the
    # "every served byte is SRI-pinned and specifier-scanned" invariant holds.
    served_url = f"/api/plugins/{name}/studio/{target.relative_to(plugin.dist_root).as_posix()}"
    if served_url not in plugin.integrity_by_url:
        return _error("not found", 404)
    if not target.is_file():
        return _error("not found", 404)
    return Response(
        target.read_bytes(),
        media_type=asset_content_type(target.name),
        headers={"cache-control": _HASHED_ASSET_CACHE},
    )


# -- SPA hosting (PUBLIC) — history fallback + static files ------------------
#
# Defined LAST in this module so its catch-all is the lowest-precedence route
# registered here (see the module docstring's ORDERING note).


def _serve_index(dist_root: Path) -> Response:
    """Serve ``index.html`` with the injected import map, a fresh CSP nonce, the
    security headers, and ``no-store`` (a cached copy would desync its inline
    nonce from the live CSP header)."""
    index = dist_root / _INDEX_FILENAME
    if not index.is_file():
        return _error("studio SPA index.html not found in the configured dist path", 500)
    try:
        registry = current_registry()
    except StudioPluginError as exc:
        return _error(str(exc), 500)
    nonce = generate_nonce()
    try:
        html = inject_importmap(index.read_text(encoding="utf-8"), registry, nonce)
    except StudioPluginError as exc:
        return _error(str(exc), 500)
    headers = security_headers(nonce)
    headers["cache-control"] = _NO_STORE
    headers["content-type"] = HTML_CONTENT_TYPE
    return Response(html, headers=headers)


def _serve_static(dist_root: Path, rel: str, target: Path) -> Response:
    """Serve an existing static file. HTML files (the OAuth pages) also carry the
    security headers + a nonce — the app's ``script-src 'self'`` protection depends
    on the header being present here, not only on the fallback branch."""
    if target.name.lower().endswith((".html", ".htm")):
        nonce = generate_nonce()
        headers = security_headers(nonce)
        headers["cache-control"] = _NO_STORE
        headers["content-type"] = HTML_CONTENT_TYPE
        return Response(target.read_text(encoding="utf-8"), headers=headers)
    cache = _REVALIDATE_CACHE if rel.startswith(_VENDOR_PREFIX) else _HASHED_ASSET_CACHE
    return Response(
        target.read_bytes(),
        media_type=asset_content_type(target.name),
        headers={"cache-control": cache},
    )


@tai42_app.http.custom_route(
    "/{spa_path:path}",
    methods=["GET"],
    summary="Serve the studio SPA (history fallback + static files)",
    tags=["plugins"],
    response_model=None,
    authed=False,
)
async def serve_spa(request: Request) -> Response:
    spa_path = request.path_params["spa_path"]
    # The catch-all must never serve index.html over an API/MCP path: those own
    # their own routes (an unknown one is a genuine 404, not the SPA shell).
    if spa_path in ("api", "mcp") or spa_path.startswith(("api/", "mcp/")):
        return _error("not found", 404)
    dist_path = plugins_settings().studio_dist_path
    if dist_path is None:
        return _error("not found", 404)
    dist_root = Path(dist_path)
    if spa_path:
        try:
            target = resolve_under(dist_root, spa_path)
        except StudioPluginError:
            return _error("not found", 404)
        # index.html always goes through injection (byte-constant serving would
        # ship it without the import map — a dead app).
        if target.is_file() and target.name != _INDEX_FILENAME:
            return _serve_static(dist_root, spa_path, target)
    return _serve_index(dist_root)
