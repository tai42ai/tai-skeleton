"""Templates HTTP surface for the Studio templates feature.

AUTHED thin adapters over operations in ``tai_skeleton.operations.templates``:

- ``GET  /api/templates``              — list template ids/paths.
- ``POST /api/template``               — fetch one template's content + schema.
- ``POST /api/upload-template``        — write a template.
- ``POST /api/delete-template``        — delete a template.
- ``POST /api/render-template``        — render by id or inline content.
- ``POST /api/clear-templates-cache``  — drop the compile cache.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``. The
logical-key containment guard and the ``404``/``400``/``500`` error mapping live in
the operations (so the MCP tool edge and the CLI carry them too); each extractor
here only parses the request body, preserving the door's hand-authored malformed-body
``400`` messages that a plain request-model parse would answer ``422`` for.
"""

from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai_skeleton.operations.templates import clear_templates_cache as _clear_templates_cache_op
from tai_skeleton.operations.templates import delete_template as _delete_template_op
from tai_skeleton.operations.templates import get_template as _get_template_op
from tai_skeleton.operations.templates import list_templates as _list_templates_op
from tai_skeleton.operations.templates import render_template as _render_template_op
from tai_skeleton.operations.templates import upload_template as _upload_template_op


async def _json_body(request: Request) -> dict:
    """Parse a JSON-object body, preserving the door's hand-authored ``400``s."""
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise BadRequestError(f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise BadRequestError("request body must be a JSON object")
    return body


async def _extract_fetch(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    return {"template_id": body.get("template_id")}


async def _extract_upload(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    return {"path": body.get("path"), "content": body.get("content")}


async def _extract_delete(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    return {"path": body.get("path")}


async def _extract_render(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    return {
        "content": body.get("content"),
        "template_id": body.get("template_id"),
        "kwargs": body.get("kwargs"),
    }


list_templates = register_operation_route(
    tai_app,
    operation_metadata_of(_list_templates_op),
    path="/api/templates",
    method="GET",
)

get_template = register_operation_route(
    tai_app,
    operation_metadata_of(_get_template_op),
    path="/api/template",
    method="POST",
    context_extractor=_extract_fetch,
)

upload_template = register_operation_route(
    tai_app,
    operation_metadata_of(_upload_template_op),
    path="/api/upload-template",
    method="POST",
    context_extractor=_extract_upload,
)

delete_template = register_operation_route(
    tai_app,
    operation_metadata_of(_delete_template_op),
    path="/api/delete-template",
    method="POST",
    context_extractor=_extract_delete,
)

render_template = register_operation_route(
    tai_app,
    operation_metadata_of(_render_template_op),
    path="/api/render-template",
    method="POST",
    context_extractor=_extract_render,
)

clear_templates_cache = register_operation_route(
    tai_app,
    operation_metadata_of(_clear_templates_cache_op),
    path="/api/clear-templates-cache",
    method="POST",
)
