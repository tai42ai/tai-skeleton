"""Connector-specific auth glue for managed MCP tool calls.

The general tool-call dispatch lives in
``tai_skeleton.tools.adapters.mcp_tool_to_func``; this module holds the parts that
are specific to connector-managed entries: resolving a ``ManagedAuth`` from the
runtime store and injecting it into the request — an OAuth token (http Bearer
header or stdio ``_meta`` field) or, for a no-auth connection, the client's
static config (http headers or stdio ``env``) — recognising the hub's structured
error sentinel, and the ``token_expired`` force-refresh-and-retry-once flow.

``is_managed`` (immutable on the config) gates every connector branch. The
empty-token guard (OAuth-only — a no-auth credential legitimately has no token),
the anchored error-framing regex (anti-forgery), the transport allowlist, and
the single-retry semantics are load-bearing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import mcp
from tai_contract.connectors.models import ConnectorRef
from tai_contract.manifest import TaiMCPConfig
from tai_kit.clients.impl.mcp import FastMCPClient

from tai_skeleton.connectors.runtime.launch import SUPPORTED_MANAGED_TRANSPORTS
from tai_skeleton.connectors.runtime.resolver import (
    ConnectorAuthExpiredError,
    ConnectorConnectionError,
    ConnectorReconnectRequiredError,
    ConnectorRefreshFailingError,
    ManagedAuth,
    force_refresh,
    resolve_managed_auth,
)
from tai_skeleton.connectors.settings import connector_adapter_settings
from tai_skeleton.settings.mcp_settings import mcp_dispatch_settings

logger = logging.getLogger(__name__)


_TOKEN_EXPIRED_CODE = "token_expired"

# Outbound structured-error codes — the client-facing half of the connector-error
# wire contract. When a managed tool call cannot proceed until the user acts, the
# adapter frames one of these (same prefix + ``{"code": ...}`` JSON the connector
# servers use) so a client recovers a machine-actionable signal via
# :func:`extract_connector_error_payload` instead of only a generic error string.
_RECONNECT_REQUIRED_CODE = "reconnect_required"
_REFRESH_FAILING_CODE = "refresh_failing"
_AUTH_EXPIRED_CODE = "auth_expired"


# The ``_meta`` token key and error prefix are the cross-repo wire contract with
# the connector-launched servers. They are read from settings lazily (never at
# import), so a ``.env`` override applied by the CLI bootstrap before the first
# tool call takes effect — settings_cache memoises the value after that.
def _meta_token_key() -> str:
    return connector_adapter_settings().meta_token_key


def _error_prefix() -> str:
    return connector_adapter_settings().error_prefix


_FRAMING_RE_CACHE: re.Pattern[str] | None = None


def _connector_error_framing_re() -> re.Pattern[str]:
    """Compiled (and memoised) framing regex for the error sentinel.

    Match the error prefix only at start-of-text or right after fastmcp's
    ``Error calling tool '<name>': `` framing — so a managed tool echoing user
    input can't forge the sentinel and trigger a phantom retry. Built on first
    use, not at import, so the prefix is read post-bootstrap.
    """
    global _FRAMING_RE_CACHE
    if _FRAMING_RE_CACHE is None:
        _FRAMING_RE_CACHE = re.compile(
            rf"^(?:Error calling tool '[^']+': )?{re.escape(_error_prefix())}",
        )
    return _FRAMING_RE_CACHE


def __getattr__(name: str) -> object:
    # Expose the settings-backed wire values as module attributes for importers
    # (``from token_injection import CONNECTOR_ERROR_PREFIX``) while deferring the
    # settings read / regex build to first access (after the CLI env bootstrap)
    # rather than running it at import — mirrors ``app.instance``'s lazy ``app``.
    if name == "CONNECTOR_META_TOKEN_KEY":
        return _meta_token_key()
    if name == "CONNECTOR_ERROR_PREFIX":
        return _error_prefix()
    if name == "_CONNECTOR_ERROR_FRAMING_RE":
        return _connector_error_framing_re()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# -- Managed auth -------------------------------------------------------------


async def resolve_managed_auth_for_config(config: TaiMCPConfig) -> ManagedAuth | None:
    """Resolve a ManagedAuth for a managed config, or None.

    None means "inject nothing": a non-managed config, or a no-auth managed entry
    with no client config. A no-auth entry WITH config returns a ManagedAuth
    carrying headers/env (no token). Only an OAuth result (``access_token`` set)
    is checked for an empty token — a no-auth credential legitimately has none.
    """
    m = config.managed
    if m is None:  # is_managed ≡ managed is not None
        return None
    auth = await resolve_managed_auth(m.connection_id, m.provider_id, m.sub_service)
    if auth is not None and auth.access_token is not None:
        _reject_empty_token(auth, m)
    return auth


async def _force_refresh(config: TaiMCPConfig, failed_access_token: str | None) -> ManagedAuth:
    m = _require_managed(config)
    auth = await force_refresh(m.connection_id, failed_access_token=failed_access_token)
    _reject_empty_token(auth, m)
    return auth


def _require_managed(config: TaiMCPConfig) -> ConnectorRef:
    """The connector ref of a managed entry; the connector branches are gated on
    ``is_managed``, so a missing ref here is a caller bug that fails loudly."""
    if config.managed is None:
        raise RuntimeError(f"config {config.title!r} is not connector-managed")
    return config.managed


def _reject_empty_token(auth: ManagedAuth, m) -> None:
    """Refuse a ManagedAuth with no access_token — never send an unauthenticated
    request as if it were authenticated."""
    if not auth.access_token:
        raise RuntimeError(
            "resolved ManagedAuth has empty access_token for "
            f"connection_id={m.connection_id} sub_service={m.sub_service}",
        )


# -- Token injection ----------------------------------------------------------


def _merge_http_auth(config: TaiMCPConfig, auth: ManagedAuth) -> TaiMCPConfig:
    """Return a copy of ``config`` with ``Authorization: Bearer …`` merged in.

    Header keys are lowercased to dedupe against any manifest-supplied header
    (HTTP names are case-insensitive, dicts are not).
    """
    raw_headers = config.config.headers or {}
    merged = {k.lower(): v for k, v in raw_headers.items()}
    merged["authorization"] = f"Bearer {auth.access_token}"
    new_inner = config.config.model_copy(update={"headers": merged})
    return config.model_copy(update={"config": new_inner})


def _merge_http_headers(config: TaiMCPConfig, headers: dict[str, str]) -> TaiMCPConfig:
    """Return a copy of ``config`` with the client's no-auth ``headers`` merged in
    (http transport). Keys lowercased to dedupe against manifest-supplied headers."""
    raw_headers = config.config.headers or {}
    merged = {k.lower(): v for k, v in raw_headers.items()}
    for key, value in headers.items():
        merged[key.lower()] = value
    new_inner = config.config.model_copy(update={"headers": merged})
    return config.model_copy(update={"config": new_inner})


def _merge_stdio_env(config: TaiMCPConfig, env: dict[str, str]) -> TaiMCPConfig:
    """Return a copy of ``config`` with the client's no-auth ``env`` merged into
    the stdio launch env (client values override static descriptor env)."""
    raw_env = config.config.env or {}
    merged = {**raw_env, **env}
    new_inner = config.config.model_copy(update={"env": merged})
    return config.model_copy(update={"config": new_inner})


def extract_connector_error_payload(response: mcp.types.CallToolResult) -> dict | None:
    """Return the connector structured-error payload, or None.

    Connector-launched servers wrap a JSON ``{"code": ...}`` payload with
    ``CONNECTOR_ERROR_PREFIX``. Only the anchored framing is trusted, so an
    echoed-back prefix in a tool's error string can't forge a match. Caller
    MUST gate on ``config.is_managed``.
    """
    if not response.isError:
        return None
    for block in response.content or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        m = _connector_error_framing_re().match(text)
        if m is None:
            continue
        suffix = text[m.end() :].strip()
        try:
            payload = json.loads(suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("code"), str):
            return payload
    return None


def managed_auth_error_result(
    exc: ConnectorConnectionError | ConnectorAuthExpiredError,
) -> mcp.types.CallToolResult:
    """Build an error ``CallToolResult`` for a managed tool call blocked on user
    action — ``invalid_grant`` (reconnect), refresh budget exhausted, or auth
    still expired after a forced refresh.

    The text carries the connector-error prefix + a ``{"code": ...}`` payload, so
    it surfaces through the same error channel as a sub-server connector error and
    a client recovers the reconnect signal via
    :func:`extract_connector_error_payload`; it also reads cleanly as prose.
    """
    if isinstance(exc, ConnectorReconnectRequiredError):
        code = _RECONNECT_REQUIRED_CODE
    elif isinstance(exc, ConnectorRefreshFailingError):
        code = _REFRESH_FAILING_CODE
    elif isinstance(exc, ConnectorAuthExpiredError):
        code = _AUTH_EXPIRED_CODE
    else:  # a bare ConnectorConnectionError (e.g. the connection is missing)
        code = _RECONNECT_REQUIRED_CODE
    payload: dict[str, str] = {"code": code, "message": str(exc), "connection_id": exc.connection_id}
    for attr in ("provider_id", "sub_service"):  # present on ConnectorAuthExpiredError
        value = getattr(exc, attr, None)
        if value is not None:
            payload[attr] = value
    text = f"{_error_prefix()}{json.dumps(payload)}"
    return mcp.types.CallToolResult(isError=True, content=[mcp.types.TextContent(type="text", text=text)])


def check_managed_transport(config: TaiMCPConfig, transport: str) -> None:
    """Allow managed entries only on transports with a token-injection path."""
    if config.is_managed and transport not in SUPPORTED_MANAGED_TRANSPORTS:
        raise RuntimeError(
            f"managed transport {transport!r} is not supported; connector-"
            "managed entries must be http (header auth) or stdio (_meta auth)."
        )


def _prepare_request(
    config: TaiMCPConfig,
    auth: ManagedAuth | None,
    transport: str,
) -> tuple[TaiMCPConfig, dict | None]:
    """Return ``(effective_config, meta)`` for one dispatch.

    ``auth is None`` → inject nothing: a non-managed entry OR a no-auth managed
    entry with no client config. ``auth.access_token`` set → OAuth: stdio ``_meta``
    token / http Bearer header. ``auth`` with headers/env but no token → no-auth
    managed with client config: merge the client's headers (http) or env (stdio).
    """
    if auth is None:
        return config, None
    if auth.access_token is not None:
        # OAuth managed entry.
        if transport == "stdio":
            return config, {_meta_token_key(): auth.access_token}
        # transport == "http" — unsupported transports rejected by the pre-flight.
        return _merge_http_auth(config, auth), None
    # No-auth managed entry with client config — transport-matched (env for stdio,
    # headers for http); MCPConfig rejects the cross combination.
    if transport == "stdio":
        return _merge_stdio_env(config, auth.env), None
    return _merge_http_headers(config, auth.headers), None


async def call_with_auth(
    config: TaiMCPConfig,
    auth: ManagedAuth | None,
    transport: str,
    tool_name: str,
    arguments: Any,
    mcp_client: FastMCPClient,
) -> mcp.types.CallToolResult:
    """One MCP dispatch.

    On the retry path :func:`_merge_http_auth` returns a fresh config (rotated
    Authorization → distinct ``model_dump`` → fresh client).
    """
    effective_config, meta = _prepare_request(config, auth, transport)
    async with mcp_client.current(config=effective_config.model_dump()) as client:
        return await client.call_tool_mcp(
            tool_name,
            arguments,
            meta=meta,
            timeout=mcp_dispatch_settings().call_timeout_seconds,
        )


async def handle_token_expired(
    config: TaiMCPConfig,
    transport: str,
    tool_name: str,
    arguments: Any,
    mcp_client: FastMCPClient,
    superseded_auth: ManagedAuth,
    failed_access_token: str | None = None,
) -> mcp.types.CallToolResult:
    """Force-refresh the token then retry the call exactly once.

    ``failed_access_token`` is the access token the token_expired'd call used; it
    fences the force-refresh so N concurrent 401s do not drive N upstream
    refreshes when a peer already rotated the token.

    ``superseded_auth`` is the auth the token_expired'd call was made with. Once
    the retry has run against the freshly-refreshed auth, the superseded auth's
    pooled session is evicted by its exact pool key so a rotation does not leak an
    open MCP session — but ONLY when its effective config actually differs from
    the fresh one. On http the rotated ``Authorization`` header yields a distinct
    key; on stdio the token travels in ``_meta`` (not the config), so the key is
    unchanged and evicting it would tear down the live just-retried session.

    Raises :class:`ConnectorAuthExpiredError` if a second token_expired arrives.
    """
    managed = _require_managed(config)
    logger.info(
        "connectors: token_expired from %s/%s (connection_id=%s) — force-refreshing and retrying once",
        managed.provider_id,
        managed.sub_service,
        managed.connection_id,
    )
    refreshed = await _force_refresh(config, failed_access_token)
    try:
        response = await call_with_auth(
            config,
            refreshed,
            transport,
            tool_name,
            arguments,
            mcp_client,
        )
        retry_payload = extract_connector_error_payload(response)
        if retry_payload is not None and retry_payload.get("code") == _TOKEN_EXPIRED_CODE:
            raise ConnectorAuthExpiredError(
                connection_id=managed.connection_id,
                provider_id=managed.provider_id,
                sub_service=managed.sub_service,
            )
        return response
    finally:
        # The fresh client now exists and served the retry, so the superseded
        # session is evicted regardless of the retry's outcome — success, an
        # error result, or the second-token_expired raise above. Both configs
        # come from the same pure ``_prepare_request``, so equal dumps mean the
        # same pool key (the stdio _meta case): closing it would evict the live
        # session, so only a genuinely distinct key (rotated http header) is closed.
        # The whole eviction — computing the effective configs AND closing the
        # superseded session — is guarded as one unit: a failure anywhere in here
        # is logged and swallowed so it can never replace the tool call's real
        # exception propagating through this ``finally``.
        try:
            superseded_config = _prepare_request(config, superseded_auth, transport)[0]
            fresh_config = _prepare_request(config, refreshed, transport)[0]
            superseded_dump = superseded_config.model_dump()
            if superseded_dump != fresh_config.model_dump():
                await mcp_client.close(config=superseded_dump)
        except Exception:
            logger.warning(
                "failed to close superseded MCP session after token rotation",
                exc_info=True,
            )


def is_token_expired(payload: dict | None) -> bool:
    """True if a structured-error payload carries the token_expired code."""
    return payload is not None and payload.get("code") == _TOKEN_EXPIRED_CODE
