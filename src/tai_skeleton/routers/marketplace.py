"""The marketplace HTTP surface — ``/api/marketplace/*`` (all AUTHED).

Eight thin adapters over the operations in
``tai_skeleton.operations.marketplace``:

* ``GET  /api/marketplace/search``                 — proxy the registry search (multi-value ``tags``).
* ``GET  /api/marketplace/plugins/{ns}/{name}``    — one listing's detail composed with its versions.
* ``GET  /api/marketplace/categories``             — the controlled category vocabulary.
* ``GET  /api/marketplace/installed``              — the installed inventory + update availability.
* ``POST /api/marketplace/install``                — install a plugin by ref.
* ``POST /api/marketplace/uninstall``              — uninstall a plugin by ref.
* ``POST /api/marketplace/update``                 — update a plugin to a newer/named version.
* ``GET  /api/marketplace/advisories``             — the cached advisory snapshot for installed plugins.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.

Security model: install/uninstall/update mutate the running environment by design
(they run arbitrary third-party code) and are scope-guardable through the existing
scopes mechanism like any other route; the operations also carry
``authority_changing`` so they are off the default projected MCP tool surface. The
registry is reached OUTBOUND only — this is a server-side proxy, so Studio needs no
second origin. A registry that cannot be reached or answers garbage is a failed
UPSTREAM dependency of this proxying surface, so it maps to a 502 Bad Gateway (a
500 would blame this server; a 503 would falsely promise that a retry fixes it) —
the translation lives in the operations layer.

The advisory poll is the ONE background outbound call the feature makes and is a
documented, default-on setting (``MARKETPLACE_ADVISORIES_POLL``); its lifecycle is
wired at the bottom of this module, so the poll runs exactly when this surface is
opted into the deployment's ``routers_modules``.

The search route reads its whitelisted query facets at the HTTP edge (``tags`` is
multi-value and must survive as repeated params, which the adapter's plain
query-param parse would collapse), so it uses a context extractor; the other reads
take path params or none, and the three writes parse their small JSON body through
the adapter's request-model parse.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.marketplace import advisories
from tai_skeleton.operations import operation_metadata_of, register_operation_route
from tai_skeleton.operations.marketplace import marketplace_advisories as _marketplace_advisories_op
from tai_skeleton.operations.marketplace import marketplace_categories as _marketplace_categories_op
from tai_skeleton.operations.marketplace import marketplace_install as _marketplace_install_op
from tai_skeleton.operations.marketplace import marketplace_installed as _marketplace_installed_op
from tai_skeleton.operations.marketplace import marketplace_plugin_detail as _marketplace_plugin_detail_op
from tai_skeleton.operations.marketplace import marketplace_search as _marketplace_search_op
from tai_skeleton.operations.marketplace import marketplace_uninstall as _marketplace_uninstall_op
from tai_skeleton.operations.marketplace import marketplace_update as _marketplace_update_op

# The single-valued search facets forwarded to the registry; ``tags`` is handled
# separately as a multi-value param. Unknown query params are ignored (they never
# reach the registry).
_SEARCH_SINGLE_PARAMS = ("q", "kind", "category", "namespace", "tier", "contract", "sort", "page", "page_size")


async def _extract_search(request: Request) -> dict[str, Any]:
    """Derive the search operation's flat facet kwargs from the query string.

    ``tags`` is read with ``getlist`` so repeated ``tags`` params survive as a list
    (the adapter's default query-param parse would keep only the last); every other
    whitelisted facet is single-valued, and absent facets are simply omitted so the
    operation's defaults apply.
    """
    params = request.query_params
    kwargs: dict[str, Any] = {}
    for key in _SEARCH_SINGLE_PARAMS:
        value = params.get(key)
        if value is not None:
            kwargs[key] = value
    tags = params.getlist("tags")
    if tags:
        kwargs["tags"] = tags
    return kwargs


marketplace_search = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_search_op),
    path="/api/marketplace/search",
    method="GET",
    context_extractor=_extract_search,
)

marketplace_plugin_detail = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_plugin_detail_op),
    path="/api/marketplace/plugins/{ns}/{name}",
    method="GET",
)

marketplace_categories = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_categories_op),
    path="/api/marketplace/categories",
    method="GET",
)

marketplace_installed = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_installed_op),
    path="/api/marketplace/installed",
    method="GET",
)

marketplace_install = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_install_op),
    path="/api/marketplace/install",
    method="POST",
)

marketplace_uninstall = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_uninstall_op),
    path="/api/marketplace/uninstall",
    method="POST",
)

marketplace_update = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_update_op),
    path="/api/marketplace/update",
    method="POST",
)

marketplace_advisories = register_operation_route(
    tai_app,
    operation_metadata_of(_marketplace_advisories_op),
    path="/api/marketplace/advisories",
    method="GET",
)


@tai_app.lifecycle.on_startup
def _start_advisories_poll() -> None:
    """Start the advisory poll on the serving loop (the startup hook runs inside
    the lifespan). It is a no-op when ``MARKETPLACE_ADVISORIES_POLL`` is off."""
    advisories.start_poll()


@tai_app.lifecycle.on_reload
def _restart_advisories_poll() -> None:
    """Re-pace the advisory poll after a reload. Reload handlers run under
    ``asyncio.run`` on a throwaway worker-thread loop — a task spawned there would
    die with that loop — so this marshals the restart onto the remembered serving
    loop, where it re-reads ``MARKETPLACE_*`` (a reload resets the settings caches)
    and re-paces/starts/stops the poll to match."""
    advisories.restart_poll_from_reload()


@tai_app.lifecycle.on_shutdown
async def _stop_advisories_poll() -> None:
    """Cancel and await the advisory poll task on the serving loop it lives on."""
    await advisories.stop_poll()
