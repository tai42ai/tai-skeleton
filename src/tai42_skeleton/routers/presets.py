"""HTTP routes for the presets surface — ``/api/presets*`` (all AUTHED).

A preset's ``fixed_kwargs`` can carry credentials, so every door here is treated
like the tool surface: authed. The routes are thin adapters over the operations in
:mod:`tai42_skeleton.operations.presets` — no preset logic lives here. Each mutating
door's body is parsed and validated at the HTTP edge into the operation's flat
arguments (producing an explicit 400 surface); the operation owns the
register/reload engine, the versioned store, the ``list_changed`` emit, and the
worker-bus fan-out.

The doors:

* ``GET /api/presets`` — one row per STORE-BACKED record (the presets plus the
  ``conflicted`` quarantined ones).
* ``POST /api/presets`` — create, ATOMIC (ordered name pre-checks → base rule →
  agent-authoring → combo/schema/bind validation → store write THEN register →
  ``list_changed`` → bus rebind).
* ``GET /api/presets/{name}`` — the store record + the active ``fixed_kwargs``.
* ``GET /api/presets/{name}/versions`` / ``.../versions/{version}`` — the version
  history / one version.
* ``POST /api/presets/{name}/versions`` — save a new version (carry-forward
  sentinels) then reload; ``list_changed`` GUARDED on a real wire/extension change.
* ``POST /api/presets/{name}/rollback`` — re-point the active version then reload.
* ``POST /api/presets/{name}/rename`` — rename, ATOMIC (a preset's name IS its live
  tool name; new bound BEFORE the old is torn down).
* ``DELETE /api/presets/{name}`` — soft-delete a live record (or HARD-delete a
  conflicted one) then tear down + fan out.
* ``GET /api/presets/{name}/referees`` — the presets a rename would strand.
* ``POST /api/presets/validate`` — dry-run a create/version draft (200 verdict).
* ``PUT /api/presets/{name}/versions/{version}/tags`` — relabel a version.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.presets import (
    create_preset as _create_preset_op,
)
from tai42_skeleton.operations.presets import (
    delete_preset as _delete_preset_op,
)
from tai42_skeleton.operations.presets import (
    get_preset as _get_preset_op,
)
from tai42_skeleton.operations.presets import (
    get_version as _get_version_op,
)
from tai42_skeleton.operations.presets import (
    list_presets as _list_presets_op,
)
from tai42_skeleton.operations.presets import (
    list_versions as _list_versions_op,
)
from tai42_skeleton.operations.presets import (
    preset_referees as _preset_referees_op,
)
from tai42_skeleton.operations.presets import (
    read_create_extensions,
    read_edit_extensions,
    read_output_schema,
)
from tai42_skeleton.operations.presets import (
    rename_preset as _rename_preset_op,
)
from tai42_skeleton.operations.presets import (
    rollback_preset as _rollback_preset_op,
)
from tai42_skeleton.operations.presets import (
    save_version as _save_version_op,
)
from tai42_skeleton.operations.presets import (
    set_preset_version_tags as _set_preset_version_tags_op,
)
from tai42_skeleton.operations.presets import (
    validate_preset as _validate_preset_op,
)

# -- request-body readers (HTTP-edge parsing; raise the byte-stable 400) ------


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


def _optional_str(body: dict[str, Any], field: str, default: str = "") -> str:
    value = body.get(field, default)
    if not isinstance(value, str):
        raise BadRequestError(f"{field!r} must be a string")
    return value


def _optional_dict(body: dict[str, Any], field: str) -> dict[str, Any]:
    value = body.get(field, {})
    if not isinstance(value, dict):
        raise BadRequestError(f"{field!r} must be a JSON object")
    return value


def _optional_str_list(body: dict[str, Any], field: str) -> list[str]:
    value = body.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BadRequestError(f"{field!r} must be a list of strings")
    return value


# -- HTTP-edge extractors (validate the body; produce the op's flat kwargs) ----


async def _extract_create(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    return {
        "name": _require_str(body, "name"),
        "base_tool": _require_str(body, "base_tool"),
        "description": _optional_str(body, "description"),
        "fixed_kwargs": _optional_dict(body, "fixed_kwargs"),
        "tags": _optional_str_list(body, "tags"),
        "extensions": read_create_extensions("extensions" in body, body.get("extensions")),
        "output_schema": read_output_schema(body.get("output_schema")),
    }


async def _extract_save_version(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    if not any(field in body for field in ("fixed_kwargs", "tags", "extensions", "output_schema")):
        raise BadRequestError("body must provide at least one of 'fixed_kwargs', 'tags', 'extensions', 'output_schema'")
    fixed_kwargs = body.get("fixed_kwargs")
    if fixed_kwargs is not None and not isinstance(fixed_kwargs, dict):
        raise BadRequestError("'fixed_kwargs' must be a JSON object")
    tags = body.get("tags")
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
        raise BadRequestError("'tags' must be a list of strings")
    extensions = read_edit_extensions("extensions" in body, body.get("extensions"))
    # ``output_schema`` carry-forward is distinct: PRESENT (even as ``null``) is a
    # deliberate value (a ``null`` CLEARS the field); ABSENT carries the active value
    # forward. ``None`` alone cannot signal both, so a presence flag drives the
    # store's carry-forward sentinel in the operation.
    output_schema_provided = "output_schema" in body
    output_schema = read_output_schema(body.get("output_schema")) if output_schema_provided else None
    return {
        "fixed_kwargs": fixed_kwargs,
        "tags": tags,
        "extensions": extensions,
        "output_schema": output_schema,
        "output_schema_provided": output_schema_provided,
    }


async def _extract_rollback(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    version = body.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise BadRequestError("body must contain an integer 'version'")
    return {"version": version}


async def _extract_rename(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    return {"new_name": _require_str(body, "new_name")}


async def _extract_validate(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    name = _require_str(body, "name")
    base_tool_present = "base_tool" in body and body.get("base_tool") is not None
    description_present = "description" in body and body.get("description") is not None
    fixed_kwargs_present = "fixed_kwargs" in body and body.get("fixed_kwargs") is not None
    tags_present = "tags" in body and body.get("tags") is not None
    return {
        "name": name,
        "base_tool": _require_str(body, "base_tool") if base_tool_present else None,
        "description": _optional_str(body, "description") if description_present else None,
        "fixed_kwargs": _optional_dict(body, "fixed_kwargs") if fixed_kwargs_present else None,
        "tags": _optional_str_list(body, "tags") if tags_present else None,
        "extensions_present": "extensions" in body,
        "extensions_value": body.get("extensions"),
        "output_schema_present": "output_schema" in body,
        "output_schema_value": body.get("output_schema"),
    }


async def _extract_version_tags(request: Request) -> dict[str, Any]:
    body = await _json_object(request)
    tags = body.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise BadRequestError("body must contain a 'tags' list of strings")
    return {"tags": tags}


# -- route registrations (thin adapters over the operations) -----------------


list_presets = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_presets_op),
    path="/api/presets",
    method="GET",
    action="read",
)

create_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_preset_op),
    path="/api/presets",
    method="POST",
    context_extractor=_extract_create,
    action="write",
)

get_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_preset_op),
    path="/api/presets/{name}",
    method="GET",
    action="read",
)

list_versions = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_versions_op),
    path="/api/presets/{name}/versions",
    method="GET",
    action="read",
)

get_version = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_version_op),
    path="/api/presets/{name}/versions/{version}",
    method="GET",
    action="read",
)

save_version = register_operation_route(
    tai42_app,
    operation_metadata_of(_save_version_op),
    path="/api/presets/{name}/versions",
    method="POST",
    context_extractor=_extract_save_version,
    action="write",
)

rollback_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_rollback_preset_op),
    path="/api/presets/{name}/rollback",
    method="POST",
    context_extractor=_extract_rollback,
    action="write",
)

rename_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_rename_preset_op),
    path="/api/presets/{name}/rename",
    method="POST",
    context_extractor=_extract_rename,
    action="write",
)

delete_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_preset_op),
    path="/api/presets/{name}",
    method="DELETE",
    action="write",
)

preset_referees = register_operation_route(
    tai42_app,
    operation_metadata_of(_preset_referees_op),
    path="/api/presets/{name}/referees",
    method="GET",
    action="read",
)

validate_preset = register_operation_route(
    tai42_app,
    operation_metadata_of(_validate_preset_op),
    path="/api/presets/validate",
    method="POST",
    context_extractor=_extract_validate,
    action="write",
)

set_preset_version_tags = register_operation_route(
    tai42_app,
    operation_metadata_of(_set_preset_version_tags_op),
    path="/api/presets/{name}/versions/{version}/tags",
    method="PUT",
    context_extractor=_extract_version_tags,
    action="write",
)
