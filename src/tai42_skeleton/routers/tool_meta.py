"""HTTP routes for the tool-metadata overlay — ``/api/tool-meta*`` (all AUTHED).

Thin adapters over the operations in :mod:`tai42_skeleton.operations.tool_meta` —
no overlay logic lives here. Each mutating door parses and validates its body at
the HTTP edge into the operation's flat arguments (an explicit 400 surface); the
operation owns the store and the merge-patch resolution.

The overlay upsert is a MERGE-PATCH (RFC-7396): the extractor forwards ONLY the
keys the caller sent, so a field's ABSENCE (unchanged) is told apart from a
present-null (clear) at the edge — the same presence-driven pattern the preset
save-version door uses for ``output_schema``. A present ``tags`` array replaces the
set. At least one field must be present.

The doors:

* ``GET  /api/tool-meta`` — the whole overlay (folders + per-tool rows).
* ``PATCH  /api/tool-meta/tools/{tool_name}`` — merge-patch a tool's overlay.
* ``DELETE /api/tool-meta/tools/{tool_name}`` — drop a tool's overlay row (idempotent).
* ``POST /api/tool-meta/folders`` — create a folder.
* ``POST /api/tool-meta/folders/{folder_id}/rename`` — rename a folder.
* ``POST /api/tool-meta/folders/{folder_id}/move`` — re-parent a folder.
* ``DELETE /api/tool-meta/folders/{folder_id}`` — delete an empty folder.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.tool_meta import (
    create_folder as _create_folder_op,
)
from tai42_skeleton.operations.tool_meta import (
    delete_folder as _delete_folder_op,
)
from tai42_skeleton.operations.tool_meta import (
    delete_tool_meta as _delete_tool_meta_op,
)
from tai42_skeleton.operations.tool_meta import (
    list_tool_meta as _list_tool_meta_op,
)
from tai42_skeleton.operations.tool_meta import (
    move_folder as _move_folder_op,
)
from tai42_skeleton.operations.tool_meta import (
    rename_folder as _rename_folder_op,
)
from tai42_skeleton.operations.tool_meta import (
    upsert_tool_meta as _upsert_tool_meta_op,
)

# -- request-body readers (HTTP-edge parsing; raise the byte-stable 400) ------

# The overlay's merge-patch fields, in a fixed order for a stable "at least one of"
# message.
_PATCH_FIELDS = ("display_name", "folder_id", "tags", "hidden")


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object") from None
    return body


def _require_str(body: dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise BadRequestError(f"body must contain a non-empty string {field!r}")
    return value


def _opt_str_or_null(body: dict[str, Any], field: str) -> str | None:
    """A field that must be a string or ``null`` when present; ``null`` (or absent)
    reads as ``None``."""
    value = body.get(field)
    if value is not None and not isinstance(value, str):
        raise BadRequestError(f"{field!r} must be a string or null")
    return value


# -- HTTP-edge extractors -----------------------------------------------------


async def _extract_upsert(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    patch: dict[str, Any] = {}
    # Only the keys the caller sent enter the patch, so absence (unchanged) stays
    # distinct from a present-null (clear). Each present value is type-checked here.
    if "display_name" in body:
        value = body["display_name"]
        if value is not None and not isinstance(value, str):
            raise BadRequestError("'display_name' must be a string or null")
        patch["display_name"] = value
    if "folder_id" in body:
        value = body["folder_id"]
        if value is not None and not isinstance(value, str):
            raise BadRequestError("'folder_id' must be a string or null")
        patch["folder_id"] = value
    if "tags" in body:
        value = body["tags"]
        if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
            raise BadRequestError("'tags' must be a list of strings")
        patch["tags"] = value
    if "hidden" in body:
        value = body["hidden"]
        if value is not None and not isinstance(value, bool):
            raise BadRequestError("'hidden' must be a boolean or null")
        patch["hidden"] = value
    if not patch:
        raise BadRequestError(f"body must provide at least one of {list(_PATCH_FIELDS)}")
    return {"patch": patch}


async def _extract_folder_create(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    return {"name": _require_str(body, "name"), "parent_id": _opt_str_or_null(body, "parent_id")}


async def _extract_folder_rename(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    return {"name": _require_str(body, "name")}


async def _extract_folder_move(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    return {"parent_id": _opt_str_or_null(body, "parent_id")}


# -- route registrations (thin adapters over the operations) -----------------


list_tool_meta = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_tool_meta_op),
    path="/api/tool-meta",
    method="GET",
    action="read",
)

upsert_tool_meta = register_operation_route(
    tai42_app,
    operation_metadata_of(_upsert_tool_meta_op),
    path="/api/tool-meta/tools/{tool_name}",
    method="PATCH",
    context_extractor=_extract_upsert,
    action="write",
)

delete_tool_meta = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_tool_meta_op),
    path="/api/tool-meta/tools/{tool_name}",
    method="DELETE",
    action="write",
)

create_folder = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_folder_op),
    path="/api/tool-meta/folders",
    method="POST",
    context_extractor=_extract_folder_create,
    action="write",
)

rename_folder = register_operation_route(
    tai42_app,
    operation_metadata_of(_rename_folder_op),
    path="/api/tool-meta/folders/{folder_id}/rename",
    method="POST",
    context_extractor=_extract_folder_rename,
    action="write",
)

move_folder = register_operation_route(
    tai42_app,
    operation_metadata_of(_move_folder_op),
    path="/api/tool-meta/folders/{folder_id}/move",
    method="POST",
    context_extractor=_extract_folder_move,
    action="write",
)

delete_folder = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_folder_op),
    path="/api/tool-meta/folders/{folder_id}",
    method="DELETE",
    action="write",
)
