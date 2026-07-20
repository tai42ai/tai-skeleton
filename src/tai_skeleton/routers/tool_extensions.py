"""HTTP routes for attaching tool extensions — ``/api/tools/{name}/extensions``
(all AUTHED).

Applying an extension to a MANIFEST-provided tool is a manifest edit + reload:
the ``extensions`` MAP on a ``tools``/``mcp`` config is the single source of
truth for which clip-on powers (chain/batch/monitor/…) a tool carries, decoupled
from ``include``/``exclude`` (selection). These two doors read and write that map
through the live-manifest edit path, mirroring ``POST /api/mcp-config``
(``routers/manifest.py``).

The doors:

* ``GET /api/tools/{name}/extensions`` — the FULL lossless list-of-combos the
  ``extensions`` map holds for the base tool (the UNION across every ``tools`` +
  ``mcp`` config that maps it) plus the catalog of available extensions. 404 when
  ``name`` is not a registered tool.
* ``POST /api/tools/{name}/extensions`` — author ALL of the tool's combos at
  once (``{"combos": [[ext, …], …]}``). The write targets the SINGLE config that
  provides the tool; an ambiguous or missing owner raises loudly (400), and a
  mapping that also lives in another config is a 409 consolidation ambiguity
  (nothing written). An empty ``combos: []`` clears the tool's extensions.

Both doors are thin adapters over operations in
``tai_skeleton.operations.tool_extensions``. The POST body carries a nested
combo structure the operation validates itself with typed errors, so it is parsed
here at the HTTP edge (producing the operation's flat ``combos`` argument) rather
than by the adapter's plain request-model parse. Success bodies are
``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai_contract.app import tai_app
from tai_contract.manifest import ExtensionElement

from tai_skeleton.manifest import Manifest  # noqa: F401 — the model-validate seam tests patch
from tai_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai_skeleton.operations.tool_extensions import get_tool_extensions as _get_tool_extensions_op
from tai_skeleton.operations.tool_extensions import set_tool_extensions as _set_tool_extensions_op


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object") from None
    return body


def _read_element(element: Any) -> ExtensionElement:
    """One combo element, structurally validated: a non-empty extension NAME (bare
    string), or a ``{"name": <non-empty str>, "config": <dict>}`` mapping binding
    author config (``config`` REQUIRED — a config-less selection is the bare-string
    form, so a config-free dict is malformed) with no other keys. Anything else is
    a loud 400. Registration of the name is checked later against the live
    registry."""
    if isinstance(element, str):
        if not element:
            raise BadRequestError("an extension name must be a non-empty string")
        return element
    if isinstance(element, dict):
        name = element.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequestError("an extension element must have a non-empty string 'name'")
        config = element.get("config")
        if not isinstance(config, dict):
            raise BadRequestError(f"extension element {name!r} must carry a 'config' mapping")
        extra = set(element) - {"name", "config"}
        if extra:
            raise BadRequestError(f"extension element {name!r} has unexpected keys: {sorted(extra)!r}")
        return {"name": name, "config": dict(config)}
    raise BadRequestError("each combo element must be an extension name or a {'name', 'config'} mapping")


def _read_combos(body: dict[str, Any]) -> list[list[ExtensionElement]]:
    """The full list-of-combos to author. Each inner combo must be a non-empty
    list of combo elements — the empty inner combo (``[[]]`` or any ``[]`` member
    of a non-empty list) is rejected because the bare branch is seeded implicitly
    and is never requested as an empty combo. A top-level ``combos: []`` is legal —
    it CLEARS the tool's extensions (drops the map key)."""
    if "combos" not in body:
        raise BadRequestError("body must contain a 'combos' list")
    combos = body["combos"]
    if not isinstance(combos, list):
        raise BadRequestError("'combos' must be a list of combos")
    result: list[list[ExtensionElement]] = []
    for combo in combos:
        if not isinstance(combo, list) or not combo:
            raise BadRequestError("each combo must be a non-empty list of extension elements")
        result.append([_read_element(element) for element in combo])
    return result


async def _extract_combos(request: Request) -> dict:
    """Parse the POST body into the operation's flat ``combos`` argument, rejecting
    a malformed structure with a loud 400 before the operation runs."""
    body = await _json_object(request)
    return {"combos": _read_combos(body)}


get_tool_extensions = register_operation_route(
    tai_app,
    operation_metadata_of(_get_tool_extensions_op),
    path="/api/tools/{name}/extensions",
    method="GET",
)

set_tool_extensions = register_operation_route(
    tai_app,
    operation_metadata_of(_set_tool_extensions_op),
    path="/api/tools/{name}/extensions",
    method="POST",
    context_extractor=_extract_combos,
)
