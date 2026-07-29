"""Connectors HTTP surface for the Studio connectors feature.

Routes (all AUTHED), prefixed ``/api/connectors``:

- ``GET    /providers``                        — provider catalog.
- ``GET    /connections``                      — installed connections (no secrets).
- ``GET    /connections/{id}``                 — one connection.
- ``POST   /connections/start``                — begin a Connect (OAuth or no-auth).
- ``DELETE /connections/{id}``                 — disconnect.
- ``POST   /connections/{id}/reconnect``       — re-run the OAuth flow.
- ``PATCH  /connections/{id}/sub-services``    — toggle enabled sub-services.
- ``POST   /oauth/complete``                   — finalize a callback (code + signed state).

The first seven doors are thin adapters over the operations in
``tai42_skeleton.operations.connectors`` — no connector logic lives here. Each
mutating door's body is parsed and validated at the HTTP edge into the operation's
flat arguments (producing an explicit 400 surface), and the
request-derived ``redirect_uri`` / ``origin`` are computed here and handed to the
operation so it stays request-free.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}`` — except
``/oauth/complete``, whose ``{"data": {...}}`` body carries a ``kind`` discriminator AND
a load-bearing status code (a failed exchange is HTTP 400 with a ``kind`` body). That
shape is not expressible in the ``{"data": ...}`` / ``{"error": ...}`` adapter envelope,
so ``/oauth/complete`` stays a native handler here.

``/oauth/complete`` decodes the HMAC-signed ``state`` envelope via ``oauth.state.decode``:
the single-use ``flow_id`` is recovered from that signed ``state``, and the token
exchange reuses the redirect URI stored in the flow state at authorize-start (so it
stays byte-identical per RFC 6749). Connection views NEVER carry tokens/config
secrets — only the stored, non-secret record fields.
"""

from __future__ import annotations

from json import JSONDecodeError

from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app
from tai42_contract.connectors.models import (
    PatchSubServicesRequest,
    StartConnectRequest,
    StartReconnectRequest,
)

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.connectors.oauth import client as oauth_client
from tai42_skeleton.connectors.oauth import state
from tai42_skeleton.connectors.oauth.redirect import compute_deployment_origin, compute_redirect_uri
from tai42_skeleton.connectors.service import connection_service
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.connectors import disconnect as _disconnect_op
from tai42_skeleton.operations.connectors import get_connection as _get_connection_op
from tai42_skeleton.operations.connectors import list_connections as _list_connections_op
from tai42_skeleton.operations.connectors import list_connector_providers as _list_connector_providers_op
from tai42_skeleton.operations.connectors import patch_sub_services as _patch_sub_services_op
from tai42_skeleton.operations.connectors import reconnect as _reconnect_op
from tai42_skeleton.operations.connectors import start_connect as _start_connect_op


class _BadRequest(ValueError):
    """A malformed request body — surfaced as a 400."""


class OAuthComplete(BaseModel):
    """The OAuth provider callback payload — the signed ``state`` and the
    authorization ``code``, or a provider-reported ``error``."""

    state: str | None = None
    code: str | None = None
    error: str | None = None


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise _BadRequest(f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise _BadRequest("request body must be a JSON object")
    return body


# -- HTTP-edge extractors (validate the body; compute request-derived args) ---


async def _extract_start_connect(request: Request) -> dict:
    try:
        body = await _json_body(request)
        req = StartConnectRequest.model_validate(body)
    except _BadRequest as exc:
        raise BadRequestError(str(exc)) from exc
    except ValidationError as exc:
        raise BadRequestError(f"invalid request body: {exc}") from exc
    return {
        "provider_id": req.provider_id,
        "alias": req.alias,
        "enabled_sub_services": req.enabled_sub_services,
        "config_values": req.config_values,
        "return_url": req.return_url,
        "redirect_uri": compute_redirect_uri(request),
        "origin": compute_deployment_origin(request),
    }


async def _extract_reconnect(request: Request) -> dict:
    try:
        body = await _json_body(request)
        req = StartReconnectRequest.model_validate(body)
    except _BadRequest as exc:
        raise BadRequestError(str(exc)) from exc
    except ValidationError as exc:
        raise BadRequestError(f"invalid request body: {exc}") from exc
    return {
        "enabled_sub_services": req.enabled_sub_services,
        "return_url": req.return_url,
        "redirect_uri": compute_redirect_uri(request),
        "origin": compute_deployment_origin(request),
    }


async def _extract_patch_sub_services(request: Request) -> dict:
    try:
        body = await _json_body(request)
        req = PatchSubServicesRequest.model_validate(body)
    except _BadRequest as exc:
        raise BadRequestError(str(exc)) from exc
    except ValidationError as exc:
        raise BadRequestError(f"invalid request body: {exc}") from exc
    return {
        "enabled_sub_services": req.enabled_sub_services,
        "return_url": req.return_url,
        "redirect_uri": compute_redirect_uri(request),
        "origin": compute_deployment_origin(request),
    }


# -- Providers / connections / connect flows (operation-backed) --------------


providers = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_connector_providers_op),
    path="/api/connectors/providers",
    method="GET",
    action="read",
)

connections = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_connections_op),
    path="/api/connectors/connections",
    method="GET",
    action="read",
)

get_connection = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_connection_op),
    path="/api/connectors/connections/{connection_id}",
    method="GET",
    action="read",
)

start_connect = register_operation_route(
    tai42_app,
    operation_metadata_of(_start_connect_op),
    path="/api/connectors/connections/start",
    method="POST",
    context_extractor=_extract_start_connect,
    action="write",
)

disconnect = register_operation_route(
    tai42_app,
    operation_metadata_of(_disconnect_op),
    path="/api/connectors/connections/{connection_id}",
    method="DELETE",
    action="write",
)

reconnect = register_operation_route(
    tai42_app,
    operation_metadata_of(_reconnect_op),
    path="/api/connectors/connections/{connection_id}/reconnect",
    method="POST",
    context_extractor=_extract_reconnect,
    action="write",
)

patch_sub_services = register_operation_route(
    tai42_app,
    operation_metadata_of(_patch_sub_services_op),
    path="/api/connectors/connections/{connection_id}/sub-services",
    method="PATCH",
    context_extractor=_extract_patch_sub_services,
    action="write",
)


# -- OAuth completion (native: discriminated body + load-bearing status) ------


@http_surface().custom_route(
    "/api/connectors/oauth/complete",
    methods=["POST"],
    summary="Complete an OAuth connection flow",
    tags=["connectors"],
    request_model=OAuthComplete,
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=True,
        error_statuses=(400, 401, 500),
        success_status=200,
    ),
    action="write",
)
async def oauth_complete(request: Request) -> Response:
    try:
        body = await _json_body(request)
    except _BadRequest as exc:
        return _error(str(exc), 400)

    # A provider-reported error (user cancelled / denied) short-circuits.
    if body.get("error"):
        return JSONResponse({"data": {"kind": "cancelled", "message": "Sign-in cancelled."}})

    state_param = body.get("state")
    code = body.get("code")
    if not isinstance(state_param, str) or not state_param:
        return _error("state is required", 400)
    if not isinstance(code, str) or not code:
        return _error("code is required", 400)

    # Recover the single-use flow_id from the HMAC-signed state. A tampered
    # or forged envelope is a loud 400 with a discriminated body.
    try:
        decoded = state.decode(state_param)
    except state.StateInvalidError:
        return JSONResponse({"data": {"kind": "failed", "reason": "StateInvalid"}}, status_code=400)

    try:
        result = await connection_service.complete_connect(
            flow_id=decoded.flow_id,
            code=code,
        )
    except (
        connection_service.AliasInUseError,
        connection_service.ConnectionNotFoundError,
        connection_service.ConcurrentConnectionUpdateError,
        oauth_client.OAuthError,
    ) as exc:
        # A post-exchange completion failure (alias collision, CAS miss, the
        # connection vanished, or a removed provider surfaced as OAuthError) is a
        # 4xx discriminated failure, never a raw 500.
        return JSONResponse({"data": {"kind": "failed", "reason": type(exc).__name__}}, status_code=400)

    return JSONResponse(
        {
            "data": {
                "kind": "success",
                "connection_id": result.connection_id,
                "return_url": result.return_url,
                "fanout": result.fanout,
            }
        }
    )
