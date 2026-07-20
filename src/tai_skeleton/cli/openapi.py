"""The OpenAPI 3.1 emitter — turns the route-metadata registry into a spec.

The app is a FastMCP + Starlette ``custom_route`` server, so it emits no schema
of its own. This module walks the shared route-enumeration primitive
(:func:`tai_skeleton.app.route_registry.load_api_routes`) and builds a valid
OpenAPI 3.1 document: one operation per method, api-key ``security`` for authed
routes, request bodies from ``request_model.model_json_schema()``, and responses
that wrap the ``{"data": ...}`` success envelope and the ``{"error": ...}``
failure envelope — including the retriable ``503`` reloading response every
reload-gated route declares.

Emission is OFFLINE by construction: the registry is populated purely by
importing the router modules, so no database, Redis, or live config/manifest is
required. The docs pipeline emits the spec with no environment booted, so
emission must never need one.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from typing import Any

from pydantic import BaseModel

from tai_skeleton.app.reload_gate import REJECT_MESSAGE
from tai_skeleton.app.route_registry import RouteMetadata, load_api_routes

_SECURITY_SCHEME = "ApiKeyAuth"
_API_KEY_HEADER = "x-api-key"

# Shared response-envelope component schemas.
_ERROR_SCHEMA = "Error"
_RELOADING_ERROR_SCHEMA = "ReloadingError"

_PATH_PARAM = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")

_NON_JSON_DESCRIPTIONS: dict[str, str] = {
    "text/event-stream": "Server-sent event stream.",
    "text/csv": "CSV export.",
    "application/octet-stream": "Asset bytes.",
    "text/html": "HTML page.",
}

_STATUS_DESCRIPTIONS: dict[int, str] = {
    400: "Malformed request.",
    401: "Missing or invalid api key.",
    403: "Forbidden.",
    404: "Resource not found.",
    409: "Conflict with the current resource state.",
    410: "Resource no longer available.",
    413: "Request body too large.",
    415: "Unsupported media type.",
    422: "Request failed validation.",
    500: "Internal server error.",
    503: "The server is applying a config reload; retry shortly.",
}


def _openapi_path(path: str) -> str:
    """Rewrite Starlette path params to OpenAPI form, dropping the ``:path``
    converter suffix (``/x/{p:path}`` -> ``/x/{p}``)."""
    return _PATH_PARAM.sub(lambda m: "{" + m.group(1) + "}", path)


def _path_parameters(path: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
        }
        for name in _PATH_PARAM.findall(path)
    ]


def _operation_id(method: str, path: str) -> str:
    slug = _PATH_PARAM.sub(lambda m: m.group(1), path)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_")
    return f"{method.lower()}_{slug}"


# Envelope schema names the emitter reserves for the shared ``{"error": ...}`` and
# reloading responses; a request/response model may never claim one.
_RESERVED_SCHEMA_NAMES = frozenset({_ERROR_SCHEMA, _RELOADING_ERROR_SCHEMA})


def _assign_component(components: dict[str, Any], name: str, schema: dict[str, Any]) -> None:
    """Write ``schema`` under ``name`` in ``components``, raising LOUDLY on a name
    collision that would otherwise silently keep or overwrite the wrong schema — a
    reserved envelope name, or two distinct models (or ``$defs``) sharing a
    ``__name__`` with differing schemas. Re-registering an identical schema (the
    same model reached twice) is a no-op."""
    if name in _RESERVED_SCHEMA_NAMES:
        raise ValueError(f"schema name {name!r} collides with a reserved response-envelope component")
    existing = components.get(name)
    if existing is not None and existing != schema:
        raise ValueError(f"schema name {name!r} maps to two distinct schemas — a component-name collision")
    components[name] = schema


def _register_model(model: type[BaseModel], components: dict[str, Any]) -> str:
    """Merge ``model``'s JSON schema (and its ``$defs``) into ``components`` and
    return the component name to ``$ref``."""
    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
    for def_name, def_schema in schema.pop("$defs", {}).items():
        _assign_component(components, def_name, def_schema)
    _assign_component(components, model.__name__, schema)
    return model.__name__


def _json_envelope_schema(meta: RouteMetadata, components: dict[str, Any]) -> dict[str, Any]:
    if meta.response_model is None:
        data_schema: dict[str, Any] = {}
    else:
        data_schema = {"$ref": f"#/components/schemas/{_register_model(meta.response_model, components)}"}
    return {
        "type": "object",
        "properties": {"data": data_schema},
        "required": ["data"],
    }


def _success_response(meta: RouteMetadata, method: str, components: dict[str, Any]) -> dict[str, Any]:
    """The 200/2xx response for ``method``, documenting every content type it serves.
    ``application/json`` carries the ``{"data": ...}`` envelope; a streaming, CSV,
    HTML, or asset/download type answers its own media type instead. A method that
    serves more than one type (the runs export: CSV or a JSON download) lists them
    all under ``content``."""
    media_types = meta.success_media_types[method]
    content: dict[str, Any] = {}
    for media_type in media_types:
        if media_type == "application/json":
            content[media_type] = {"schema": _json_envelope_schema(meta, components)}
        else:
            content[media_type] = {"schema": {"type": "string"}}
    if len(media_types) == 1 and media_types[0] != "application/json":
        description = _NON_JSON_DESCRIPTIONS.get(media_types[0], "Success.")
    else:
        description = "Success."
    return {"description": description, "content": content}


def _error_response(status: int) -> dict[str, Any]:
    if status == 503:
        return {
            "description": _STATUS_DESCRIPTIONS[503],
            "headers": {
                "Retry-After": {
                    "description": "Seconds to wait before retrying.",
                    "schema": {"type": "integer"},
                }
            },
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{_RELOADING_ERROR_SCHEMA}"}}},
        }
    return {
        "description": _STATUS_DESCRIPTIONS.get(status, "Error."),
        "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{_ERROR_SCHEMA}"}}},
    }


def _operation(meta: RouteMetadata, method: str, components: dict[str, Any]) -> dict[str, Any]:
    responses: dict[str, Any] = {str(meta.success_status): _success_response(meta, method, components)}
    for status in meta.error_statuses:
        responses[str(status)] = _error_response(status)

    operation: dict[str, Any] = {
        "operationId": _operation_id(method, meta.path),
        "summary": meta.summary,
        "tags": list(meta.tags),
        "responses": responses,
    }
    if meta.description:
        operation["description"] = meta.description

    parameters = _path_parameters(meta.path)
    if parameters:
        operation["parameters"] = parameters

    if meta.request_model is not None:
        ref = _register_model(meta.request_model, components)
        operation["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{ref}"}}},
        }

    if meta.authed:
        operation["security"] = [{_SECURITY_SCHEME: []}]

    # A destructive route (an operation flagged ``destructive`` or a DELETE the
    # adapter auto-forced) advertises it so a client can gate the call; a
    # non-destructive route emits nothing.
    if meta.destructive:
        operation["x-destructive"] = True

    return operation


def build_openapi_spec() -> dict[str, Any]:
    """Build the OpenAPI 3.1 document for the ``/api/*`` surface.

    Offline: reads the route-metadata registry only. Every registered ``/api/*``
    route appears; reload-gated routes carry the retriable ``503`` response.
    """
    components: dict[str, Any] = {
        _ERROR_SCHEMA: {
            "type": "object",
            "properties": {"error": {"type": "string"}},
            "required": ["error"],
        },
        _RELOADING_ERROR_SCHEMA: {
            "type": "object",
            "properties": {
                "error": {"type": "string", "const": REJECT_MESSAGE},
                "reloading": {"type": "boolean", "const": True},
            },
            "required": ["error", "reloading"],
        },
    }

    paths: dict[str, dict[str, Any]] = {}
    for meta in load_api_routes():
        oapath = paths.setdefault(_openapi_path(meta.path), {})
        for method in meta.methods:
            oapath[method.lower()] = _operation(meta, method, components)

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "tai-skeleton API",
            "version": version("tai-skeleton"),
            "description": "The operator HTTP surface served under /api/*.",
        },
        "paths": paths,
        "components": {
            "schemas": components,
            "securitySchemes": {_SECURITY_SCHEME: {"type": "apiKey", "in": "header", "name": _API_KEY_HEADER}},
        },
    }
