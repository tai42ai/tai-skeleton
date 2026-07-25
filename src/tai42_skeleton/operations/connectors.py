"""Connector operations for the Studio connectors feature.

Seven operations over the connector service — the provider catalog, the installed
connections (secret-free views), and the connect/reconnect/patch/disconnect
mutations. The OAuth-completion door is NOT here: its ``{"data": {...}}`` body
carries a ``kind`` discriminator AND a load-bearing status code (a failed exchange
is HTTP 400 with a ``kind`` body), which the ``{"data": ...}`` / ``{"error": ...}``
adapter envelope cannot express — it stays a native handler in the router.

Requests parse and responses serialize THROUGH the tai42-contract connector wire
models; the mutations map a malformed body / off-list origin to a 400, an alias
collision to a 409, and an absent connection to a 404. The connect/reconnect/patch
operations take the request-derived ``redirect_uri`` / ``origin`` as flat arguments
the route's extractor computes at the HTTP edge, so the operation stays request-free.
Connection views NEVER carry tokens/config secrets — only the stored, non-secret
record fields.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.connectors.models import (
    ConnectedAccountView,
    ConnectionRecord,
    ConnectionsListResponse,
    DisconnectResponse,
    PatchSubServicesRequest,
    PatchSubServicesResponse,
    ProviderCatalogEntry,
    StartConnectNoAuthResponse,
    StartConnectRequest,
    StartConnectResponse,
    StartReconnectRequest,
    SubServiceView,
)
from tai42_contract.connectors.providers import ConfigFieldSpec, ProviderDescriptor
from tai42_contract.connectors.service import (
    AliasInUseError,
    NoAuthConnectResult,
    StartConnectResult,
)

from tai42_skeleton.connectors.oauth import client as oauth_client
from tai42_skeleton.connectors.providers.registry import list_providers
from tai42_skeleton.connectors.service import connection_service
from tai42_skeleton.connectors.store import token_store
from tai42_skeleton.connectors.store.persistence import load_record_or_none
from tai42_skeleton.operations import BadRequestError, ConflictError, NotFoundError, operation

# -- Serialization (through the contract wire models; never leaks secrets) ----


def _provider_view(provider: ProviderDescriptor) -> dict[str, Any]:
    return ProviderCatalogEntry(
        id=provider.id,
        display_name=provider.display_name,
        description=provider.description,
        icon_url=provider.icon_url,
        kind=provider.kind,
        origin=provider.origin,
        category=provider.category,
        sub_services=[
            SubServiceView(id=s.id, display_name=s.display_name, description=s.description, scopes=list(s.scopes))
            for s in provider.sub_services.values()
        ],
        config_fields=[
            ConfigFieldSpec(key=f.key, label=f.label, target=f.target, required=f.required, secret=f.secret)
            for f in provider.config_fields
        ],
    ).model_dump(mode="json")


def _connection_account(record: ConnectionRecord) -> ConnectedAccountView:
    # Tokens (access/refresh/expiry) and no-auth config_values are SecretStr and
    # are DELIBERATELY excluded — this view is UI-facing.
    return ConnectedAccountView(
        connection_id=record.connection_id,
        provider_id=record.provider_id,
        alias=record.alias,
        kind=record.kind,
        account_identity=record.account_identity,
        enabled_sub_services=record.enabled_sub_services,
        granted_scopes=record.granted_scopes,
        auth_health_state=record.auth_health_state,
        created_at=record.created_at,
    )


def _start_result_view(result: StartConnectResult | NoAuthConnectResult) -> dict[str, Any]:
    if isinstance(result, StartConnectResult):
        return StartConnectResponse(flow_id=result.flow_id, authorize_url=result.authorize_url).model_dump(mode="json")
    return StartConnectNoAuthResponse(
        connection_id=result.connection_id,
        added_manifest_entries=result.added_manifest_entries,
        fanout=result.fanout,
    ).model_dump(mode="json")


# -- Providers + connections (reads) -----------------------------------------


@operation(summary="List OAuth connector providers", tags=["connectors"])
async def list_connector_providers() -> list[dict[str, Any]]:
    """The provider catalog — one entry per registered connector provider."""
    return [_provider_view(p) for p in list_providers()]


@operation(summary="List connections", tags=["connectors"])
async def list_connections() -> dict[str, Any]:
    """The installed connections as secret-free views."""
    ids = await token_store().list()
    records = [await load_record_or_none(cid) for cid in ids]
    items = [_connection_account(r) for r in records if r is not None]
    return ConnectionsListResponse(items=items, total=len(items)).model_dump(mode="json")


@operation(summary="Get a connection", tags=["connectors"], errors=[NotFoundError])
async def get_connection(connection_id: str) -> dict[str, Any]:
    """One connection's secret-free view; an unknown id is a loud 404."""
    record = await load_record_or_none(connection_id)
    if record is None:
        raise NotFoundError("connection not found")
    return _connection_account(record).model_dump(mode="json")


# -- Connect / reconnect / patch / disconnect (mutations) --------------------


@operation(
    summary="Start a connection flow",
    tags=["connectors"],
    destructive=True,
    errors=[BadRequestError, ConflictError],
    request_model=StartConnectRequest,
)
async def start_connect(
    provider_id: str,
    alias: str,
    enabled_sub_services: list[str],
    config_values: dict[str, str],
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> dict[str, Any]:
    """Begin a Connect (OAuth authorize URL, or an immediate no-auth connection)."""
    try:
        result = await connection_service.start_connect(
            provider_id=provider_id,
            alias=alias,
            enabled_sub_services=enabled_sub_services,
            config_values=config_values,
            return_url=return_url,
            redirect_uri=redirect_uri,
            origin=origin,
        )
    except AliasInUseError as exc:
        raise ConflictError(str(exc)) from exc
    except (ValueError, oauth_client.OAuthError) as exc:
        # An off-list Origin surfaces as RedirectUriNotAllowedError (an OAuthError);
        # map it to a clean 400 rather than let it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    return _start_result_view(result)


@operation(summary="Disconnect a connection", tags=["connectors"], errors=[NotFoundError])
async def disconnect(connection_id: str) -> dict[str, Any]:
    """Disconnect (purge) a connection; a genuinely-absent connection is a 404."""
    # Let the service load the record with include_expired so a lapsed connection is
    # still purgeable (a serving-filtered pre-check would 404 an expired connection and
    # strand its blob + manifest entries). A genuinely-absent connection surfaces as
    # ConnectionNotFoundError.
    try:
        result = await connection_service.disconnect(connection_id=connection_id)
    except connection_service.ConnectionNotFoundError as exc:
        raise NotFoundError("connection not found") from exc
    return DisconnectResponse(
        connection_id=result.connection_id,
        upstream_revoke_outcome=result.upstream_revoke_outcome,
        upstream_revoke_status=result.upstream_revoke_status,
        removed_manifest_entries=result.removed_manifest_entries,
        fanout=result.fanout,
    ).model_dump(mode="json")


@operation(
    summary="Start a reconnect flow",
    tags=["connectors"],
    destructive=True,
    errors=[BadRequestError, NotFoundError],
    request_model=StartReconnectRequest,
)
async def reconnect(
    connection_id: str,
    enabled_sub_services: list[str],
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> dict[str, Any]:
    """Re-run the OAuth flow for an existing connection; an unknown id is a 404."""
    if await load_record_or_none(connection_id) is None:
        raise NotFoundError("connection not found")
    try:
        result = await connection_service.start_reconnect(
            connection_id=connection_id,
            enabled_sub_services=enabled_sub_services,
            return_url=return_url,
            redirect_uri=redirect_uri,
            origin=origin,
        )
    except (ValueError, oauth_client.OAuthError) as exc:
        # An off-list Origin surfaces as RedirectUriNotAllowedError (an OAuthError);
        # map it to a clean 400 rather than let it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    return StartConnectResponse(flow_id=result.flow_id, authorize_url=result.authorize_url).model_dump(mode="json")


@operation(
    summary="Patch a connection's enabled sub-services",
    tags=["connectors"],
    destructive=True,
    errors=[BadRequestError, NotFoundError],
    request_model=PatchSubServicesRequest,
)
async def patch_sub_services(
    connection_id: str,
    enabled_sub_services: list[str],
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> dict[str, Any]:
    """Toggle a connection's enabled sub-services; an unknown id is a 404."""
    if await load_record_or_none(connection_id) is None:
        raise NotFoundError("connection not found")
    try:
        result = await connection_service.patch_sub_services(
            connection_id=connection_id,
            desired=enabled_sub_services,
            return_url=return_url,
            redirect_uri=redirect_uri,
            origin=origin,
        )
    except (ValueError, oauth_client.OAuthError) as exc:
        # An off-list Origin surfaces as RedirectUriNotAllowedError (an OAuthError);
        # map it to a clean 400 rather than let it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    return PatchSubServicesResponse(
        connection_id=result.connection_id,
        enabled_sub_services=result.enabled_sub_services,
        consent_required=result.consent_required,
        flow_id=result.flow_id,
        authorize_url=result.authorize_url,
        added_manifest_entries=result.added_manifest_entries,
        removed_manifest_entries=result.removed_manifest_entries,
        fanout=result.fanout,
    ).model_dump(mode="json")
