"""HTTP surface for the storage provider — ``/api/storage*`` (all AUTHED).

AUTHED thin adapters over operations in ``tai42_skeleton.operations.storage`` — a thin
skin over the registered :class:`~tai42_contract.storage.Storage` provider (the app's
content store). Storage is dead by default (the skeleton ships no provider); a
backend registers one as a manifest-loaded plugin:

- ``GET /api/storage`` — provider identity (or ``present: false``, 200).
- ``GET /api/storage/resources`` — the sorted resource ids.
- ``GET /api/storage/resources/{id}/stat`` — the object's inferred content type.
- ``GET /api/storage/resources/{id}/content`` — the raw object bytes, as a file
  download. THIS door stays a native handler: it answers raw bytes + a
  Content-Disposition attachment header, not the ``{"data": ...}`` envelope.
- ``POST /api/storage/resources`` — upload text OR base64 bytes under an id.
- ``DELETE /api/storage/resources/{id}`` — remove one object.
- ``DELETE /api/storage/dirs/{path}`` — remove a directory subtree.

With no provider registered every door except ``GET /api/storage`` answers a loud
501. Success bodies are ``{"data": ...}`` (or the raw download for content); failures
are ``{"error": "<message>"}``. The id/path containment guard and the provider-error
mapping live in the operations (so the MCP tool edge and the CLI carry them too).
"""

from __future__ import annotations

import posixpath

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.storage import (
    _NO_PROVIDER_MESSAGE,
    _UNSAFE_ID_MESSAGE,
    _content_disposition,
    _is_unsafe_path,
    _provider,
)
from tai42_skeleton.operations.storage import delete_dir as _delete_dir_op
from tai42_skeleton.operations.storage import delete_resource as _delete_resource_op
from tai42_skeleton.operations.storage import list_resources as _list_resources_op
from tai42_skeleton.operations.storage import stat_resource as _stat_resource_op
from tai42_skeleton.operations.storage import storage_info as _storage_info_op
from tai42_skeleton.operations.storage import upload_resource as _upload_resource_op


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _extract_upload(request: Request) -> dict:
    """Parse the upload body at the HTTP edge, preserving the door's hand-authored
    malformed-body ``400`` messages; the operation validates the field shapes."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object")
    return {
        "resource_id": body.get("id"),
        "content_text": body.get("content_text"),
        "content_base64": body.get("content_base64"),
    }


storage_info = register_operation_route(
    tai42_app,
    operation_metadata_of(_storage_info_op),
    path="/api/storage",
    method="GET",
)

list_resources = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_resources_op),
    path="/api/storage/resources",
    method="GET",
)

stat_resource = register_operation_route(
    tai42_app,
    operation_metadata_of(_stat_resource_op),
    path="/api/storage/resources/{resource_id:path}/stat",
    method="GET",
)

upload_resource = register_operation_route(
    tai42_app,
    operation_metadata_of(_upload_resource_op),
    path="/api/storage/resources",
    method="POST",
    context_extractor=_extract_upload,
)

delete_resource = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_resource_op),
    path="/api/storage/resources/{resource_id:path}",
    method="DELETE",
)

delete_dir = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_dir_op),
    path="/api/storage/dirs/{dir_path:path}",
    method="DELETE",
)


@http_surface().custom_route(
    "/api/storage/resources/{resource_id:path}/content",
    methods=["GET"],
    summary="Download a storage resource",
    tags=["storage"],
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=False,
        error_statuses=(400, 401, 404, 500, 501),
        success_status=200,
    ),
)
async def download_resource(request: Request) -> Response:
    """Serve the raw object bytes as a file download.

    A content server (raw bytes + a Content-Disposition attachment header, not the
    ``{"data": ...}`` envelope), so it stays a native handler rather than an
    operation adapter."""
    resource_id = request.path_params["resource_id"]
    if _is_unsafe_path(resource_id):
        return _error(f"resource id {resource_id!r} {_UNSAFE_ID_MESSAGE}", 400)
    provider = _provider()
    if provider is None:
        return _error(_NO_PROVIDER_MESSAGE, 501)
    try:
        data = await provider.load_bytes(resource_id)
        content_type = (await provider.stat(resource_id)).content_type or "application/octet-stream"
    except FileNotFoundError:
        return _error(f"resource {resource_id!r} not found", 404)
    except ValueError as exc:
        # A provider-reported boundary violation is a client error (400), never a 500.
        return _error(str(exc), 400)
    filename = posixpath.basename(resource_id)
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": _content_disposition(filename)},
    )
