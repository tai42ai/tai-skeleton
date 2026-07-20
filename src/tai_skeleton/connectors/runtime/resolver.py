"""Runtime token resolution for connector-managed manifest entries.

Returns a :class:`ManagedAuth` (a bearer access token) for every outbound MCP
call against a managed entry, refreshing proactively before expiry or reactively
(via :func:`force_refresh`) after an upstream 401.

Cross-replica safety comes from the per-connection Redis lock plus a
compare-and-set write-back: the hot path reads the record and serves a
still-fresh token without locking; any read-modify-write of the record happens
under :func:`connection_lock`, so N replicas racing a just-expired token
serialise into one upstream refresh. A refresh can outlive the lock's TTL (the
transient-retry budget), so the write-back is keyed on the ciphertext the refresh
began with via ``store.put(expected_blob=...)``: if a peer rotated the stored
record meanwhile, our compare-and-set loses and we serve the peer's record rather
than clobber it. The store decides the race atomically — there is no
re-read-then-write window.

Failure modes map to typed exceptions the caller (the MCP request handler)
translates into tool errors:

- :class:`ConnectorReconnectRequiredError` — provider returned ``invalid_grant``;
  the user must reconnect.
- :class:`ConnectorRefreshFailingError` — the bounded retry budget on transient
  errors was exhausted; the connection flips to ``REFRESH_FAILING``.
- :class:`ConnectorConnectionError` — the connection is missing (base class).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import SecretStr
from tai_contract.connectors.errors import ConnectorError
from tai_contract.connectors.models import AuthHealthState, ConnectionRecord

from tai_skeleton.connectors.oauth import client as oauth_client
from tai_skeleton.connectors.oauth import crypto
from tai_skeleton.connectors.providers.registry import get_provider
from tai_skeleton.connectors.runtime.launch import resolve_mcp_server
from tai_skeleton.connectors.runtime.locks import (
    clear_refresh_cooldown,
    connection_lock,
    open_refresh_cooldown,
    refresh_cooldown_active,
)
from tai_skeleton.connectors.store import token_store
from tai_skeleton.connectors.store.persistence import (
    ConnectionNotFoundError,
    load_record_with_blob,
    session_expires_at_for,
)

logger = logging.getLogger(__name__)

SAFETY_MARGIN_SECONDS = 60
TRANSIENT_RETRY_BUDGET = 5
TRANSIENT_BACKOFF_BASE_SECONDS = 1.0
TRANSIENT_BACKOFF_FACTOR = 2.0


# -- ManagedAuth + errors -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManagedAuth:
    """Connector-resolved credential for a managed sub-service.

    OAuth: ``access_token`` set — the adapter injects it (HTTP
    ``Authorization: Bearer`` header or the stdio ``_meta`` token field).
    No-auth with config: ``access_token`` is None and ``headers`` (http) or
    ``env`` (stdio) carry the client-supplied static values the adapter merges
    into the request at call time."""

    access_token: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


class ConnectorAuthExpiredError(ConnectorError):
    """Raised by the mcp adapter when ``token_expired`` persists after a
    force-refresh + retry. Carries the connection identity the user needs to
    reconnect so callers can branch without parsing the message."""

    def __init__(
        self,
        *,
        connection_id: str,
        provider_id: str,
        sub_service: str,
    ) -> None:
        super().__init__(
            f"token_expired persisted after force-refresh for "
            f"connection_id={connection_id} "
            f"provider_id={provider_id} sub_service={sub_service}"
        )
        self.connection_id = connection_id
        self.provider_id = provider_id
        self.sub_service = sub_service


class ConnectorConnectionError(ConnectorError):
    def __init__(self, message: str, *, connection_id: str):
        super().__init__(message)
        self.connection_id = connection_id


class ConnectorReconnectRequiredError(ConnectorConnectionError):
    """invalid_grant from the provider — the user must reconnect."""


class ConnectorRefreshFailingError(ConnectorConnectionError):
    """Transient-failure retry budget exhausted."""


# -- Helpers --------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _seconds_until_expiry(record: ConnectionRecord) -> float:
    expires_at = record.access_token_expires_at
    if expires_at is None:
        # An oauth record without an expiry is corrupt — fail loudly, never
        # guess a freshness answer.
        raise ConnectorConnectionError(
            f"oauth record {record.connection_id} has no access_token_expires_at",
            connection_id=record.connection_id,
        )
    return (expires_at - _now_utc()).total_seconds()


def _is_fresh(record: ConnectionRecord) -> bool:
    return record.auth_health_state == AuthHealthState.HEALTHY and _seconds_until_expiry(record) > SAFETY_MARGIN_SECONDS


def _managed_auth(record: ConnectionRecord) -> ManagedAuth:
    access_token = record.access_token
    if access_token is None:
        # An oauth record without a token is corrupt — fail loudly, never serve
        # an unauthenticated credential as if it were authenticated.
        raise ConnectorConnectionError(
            f"oauth record {record.connection_id} has no access_token",
            connection_id=record.connection_id,
        )
    return ManagedAuth(access_token=access_token.get_secret_value())


async def _persist(record: ConnectionRecord, *, expected_blob: bytes) -> bool:
    """Encrypt and compare-and-set the record against the ciphertext the refresh
    began with. Returns ``True`` if the durable write committed, ``False`` on a
    CAS miss — a peer rotated the stored record while our refresh was in flight,
    so our result is stale and must not overwrite theirs."""
    blob = crypto.encrypt(
        record.to_storage_json().encode("utf-8"),
        connection_id=record.connection_id,
    )
    return await token_store().put(
        record.connection_id,
        blob,
        expected_blob=expected_blob,
        session_expires_at=session_expires_at_for(record),
    )


async def _load(connection_id: str) -> tuple[ConnectionRecord, bytes]:
    """Load the record together with the ciphertext it decrypted from (the CAS
    handle for a later write-back)."""
    try:
        return await load_record_with_blob(connection_id)
    except ConnectionNotFoundError as exc:
        raise ConnectorConnectionError(
            f"connection {connection_id} not found",
            connection_id=connection_id,
        ) from exc


# -- Refresh (under the connection lock) ----------------------------------


async def _serve_rotated_peer(connection_id: str) -> ManagedAuth:
    """Resolve the credential to serve after our write-back lost the
    compare-and-set: a peer rotated the stored record while our refresh was in
    flight (the connection lock's TTL can elapse before a slow refresh finishes
    its retry budget). Re-read the peer's current record once and serve it
    instead of clobbering it with our stale result."""
    logger.warning(
        "connectors: refresh write-back lost the compare-and-set for %s — a "
        "concurrent refresh rotated the record; serving the peer's record",
        connection_id,
    )
    peer, _ = await _load(connection_id)
    return _serve_peer(peer)


def _serve_peer(peer: ConnectionRecord) -> ManagedAuth:
    """Resolve the credential to serve when our refresh was fenced out by a peer."""
    connection_id = peer.connection_id
    if peer.auth_health_state == AuthHealthState.RECONNECT_REQUIRED:
        raise ConnectorReconnectRequiredError(
            f"connection {connection_id} requires reconnect",
            connection_id=connection_id,
        )
    if peer.auth_health_state == AuthHealthState.HEALTHY:
        return _managed_auth(peer)
    raise ConnectorRefreshFailingError(
        f"refresh for {connection_id} was superseded by a concurrent refresh that left state {peer.auth_health_state}",
        connection_id=connection_id,
    )


async def _refresh(record: ConnectionRecord, started_blob: bytes) -> ManagedAuth:
    """Drive the upstream OAuth refresh and persist the result. Must run under
    :func:`connection_lock`.

    Always refreshes — both callers have already decided a refresh is needed
    (the freshness gate lives in :func:`resolve_managed_auth` under the lock;
    :func:`force_refresh` is invoked after an upstream 401 told us the token is
    dead despite the local clock). On transient failures it retries the upstream
    call with exponential backoff up to :data:`TRANSIENT_RETRY_BUDGET`.

    ``started_blob`` is the ciphertext this refresh loaded from. Every write-back
    is a compare-and-set against it: if the stored blob has since rotated, a peer
    refresh won the race, the store rejects our write, and we serve the peer's
    record instead of clobbering it (see :func:`_serve_rotated_peer`).
    """
    connection_id = record.connection_id
    if record.refresh_token is None:
        # An oauth record without a refresh token cannot be refreshed — corrupt
        # state fails loudly instead of dispatching a bogus upstream call.
        raise ConnectorConnectionError(
            f"oauth record {connection_id} has no refresh_token",
            connection_id=connection_id,
        )
    refresh_token = record.refresh_token.get_secret_value()
    try:
        descriptor = get_provider(record.provider_id)
    except KeyError as exc:
        # The provider plugin was removed since this connection was created — there
        # is no descriptor to drive the refresh. Surface the typed connection error
        # (matching resolve_managed_auth's no-auth branch) rather than a raw KeyError.
        raise ConnectorConnectionError(
            f"connection {connection_id} references unknown provider {record.provider_id!r}",
            connection_id=connection_id,
        ) from exc
    transient_attempts = 0
    while True:
        try:
            token_resp = await oauth_client.refresh(
                descriptor=descriptor,
                refresh_token=refresh_token,
            )
        except oauth_client.TokenRefreshFailedError as exc:
            if exc.reason == "invalid_grant":
                # The invalid_grant may be spurious: a peer rotated our
                # refresh_token out from under us. Compare-and-set the terminal
                # state so a lost race never clobbers the peer's healthy record.
                record.auth_health_state = AuthHealthState.RECONNECT_REQUIRED
                if not await _persist(record, expected_blob=started_blob):
                    return await _serve_rotated_peer(connection_id)
                raise ConnectorReconnectRequiredError(
                    f"refresh failed with invalid_grant for {connection_id}",
                    connection_id=connection_id,
                ) from exc

            # Transient — apply the retry budget then exponential backoff. The
            # local counter drives the retry loop only; it is not persisted.
            transient_attempts += 1
            if transient_attempts >= TRANSIENT_RETRY_BUDGET:
                record.auth_health_state = AuthHealthState.REFRESH_FAILING
                if not await _persist(record, expected_blob=started_blob):
                    return await _serve_rotated_peer(connection_id)
                # Arm the circuit breaker so the next tool calls fail fast for the
                # cooldown window instead of each re-burning the full retry budget.
                await open_refresh_cooldown(connection_id)
                logger.warning(
                    "connectors: refresh transient budget exhausted for %s",
                    connection_id,
                )
                raise ConnectorRefreshFailingError(
                    f"transient refresh failures exhausted budget ({TRANSIENT_RETRY_BUDGET}) for {connection_id}",
                    connection_id=connection_id,
                ) from exc
            backoff = TRANSIENT_BACKOFF_BASE_SECONDS * (TRANSIENT_BACKOFF_FACTOR ** (transient_attempts - 1))
            logger.info(
                "connectors: refresh transient failure %d/%d for %s (retry in %.1fs)",
                transient_attempts,
                TRANSIENT_RETRY_BUDGET,
                connection_id,
                backoff,
            )
            await asyncio.sleep(backoff)
            continue

        # Success. Write the new access token and any rotated refresh token in
        # the same record (rotation-style providers like Atlassian). The write is
        # a compare-and-set: if a peer already rotated the record while ours was
        # in flight, serve the peer's record rather than overwrite it with our
        # (now superseded) tokens.
        record.access_token = SecretStr(token_resp.access_token)
        record.access_token_expires_at = token_resp.expires_at
        if token_resp.refresh_token:
            record.refresh_token = SecretStr(token_resp.refresh_token)
        record.auth_health_state = AuthHealthState.HEALTHY
        if token_resp.granted_scopes:
            record.granted_scopes = list(token_resp.granted_scopes)
        if not await _persist(record, expected_blob=started_blob):
            return await _serve_rotated_peer(connection_id)
        # Recovery: drop any breaker a prior failing run left so a fresh token
        # already inside the safety margin is not fast-failed by a live cooldown.
        await clear_refresh_cooldown(connection_id)
        return _managed_auth(record)


# -- Public API -----------------------------------------------------------


async def resolve_managed_auth(
    connection_id: str,
    provider_id: str,
    sub_service: str,
) -> ManagedAuth | None:
    """Return the :class:`ManagedAuth` for a managed connection, or None.

    No-auth (kind="none"): short-circuits at the top — no lock, no freshness
    gate, no refresh. Returns None when the connection has no client config
    (inject nothing), else a credential carrying the config_values on the
    sub-service's transport channel (env for stdio, headers for http).

    OAuth: hot path serves a healthy, still-fresh token without the lock;
    otherwise the read-modify-write runs under :func:`connection_lock`, which
    re-loads the record and refreshes only if still needed. ``provider_id`` /
    ``sub_service`` come from the manifest ref; the OAuth body keys on
    ``connection_id``.
    """
    record, _ = await _load(connection_id)

    # The manifest ref names the provider it expects; a record resolving to a
    # different provider means the connection_id was misrouted. Fail loudly
    # rather than inject this connection's token into another provider's config.
    if record.provider_id != provider_id:
        raise ConnectorConnectionError(
            f"connection {connection_id} resolves to provider {record.provider_id!r} "
            f"but the manifest ref names provider {provider_id!r}",
            connection_id=connection_id,
        )

    if record.kind == "none":
        if not record.config_values:
            return None
        try:
            descriptor = get_provider(record.provider_id)
            server = resolve_mcp_server(descriptor, sub_service)
        except KeyError as exc:
            raise ConnectorConnectionError(
                f"connection {connection_id} references unknown provider "
                f"{record.provider_id!r} or sub-service {sub_service!r}",
                connection_id=connection_id,
            ) from exc
        values = {key: value.get_secret_value() for key, value in record.config_values.items()}
        if server.type == "stdio":
            return ManagedAuth(env=values)
        return ManagedAuth(headers=values)

    if record.auth_health_state == AuthHealthState.RECONNECT_REQUIRED:
        raise ConnectorReconnectRequiredError(
            f"connection {connection_id} requires reconnect",
            connection_id=connection_id,
        )
    if _is_fresh(record):
        return _managed_auth(record)

    # Slow path. A refresh that recently exhausted its retry budget leaves a
    # cooldown breaker; fast-fail here (before even queueing on the lock) so a
    # failing connection cannot re-burn the full retry budget on every call and
    # stampede the lock-timeout waiters behind it.
    if await refresh_cooldown_active(connection_id):
        raise ConnectorRefreshFailingError(
            f"connection {connection_id} is in refresh cooldown after a failing refresh",
            connection_id=connection_id,
        )

    async with connection_lock(connection_id):
        record, started_blob = await _load(connection_id)
        if record.auth_health_state == AuthHealthState.RECONNECT_REQUIRED:
            raise ConnectorReconnectRequiredError(
                f"connection {connection_id} requires reconnect",
                connection_id=connection_id,
            )
        if _is_fresh(record):
            return _managed_auth(record)
        # A peer may have opened the breaker while we waited for the lock.
        if await refresh_cooldown_active(connection_id):
            raise ConnectorRefreshFailingError(
                f"connection {connection_id} is in refresh cooldown after a failing refresh",
                connection_id=connection_id,
            )
        return await _refresh(record, started_blob)


async def force_refresh(
    connection_id: str,
    *,
    failed_access_token: str | None = None,
) -> ManagedAuth:
    """Drive an upstream OAuth refresh once and return the fresh credential.

    Called by the MCP tool-call adapter when a managed server returns the
    ``token_expired`` sentinel — the cached token raced a cross-replica rotation
    and the adapter needs a guaranteed-fresh token before its one retry.

    Fences on the failing token: ``failed_access_token`` is the access token the
    401'd call used. Under the lock the record is re-loaded; if a peer already
    rotated the stored token (it no longer equals the failing one), the peer's
    refresh already fixed it — serve the peer's record instead of driving another
    upstream refresh. Refreshing again would, on a single-active-token /
    rotation-invalidating provider, kill the healthy token a peer just minted and
    turn N concurrent 401s into a spurious ``ConnectorAuthExpiredError``.

    A connection already in refresh cooldown fails fast without an upstream call.
    """
    async with connection_lock(connection_id):
        record, started_blob = await _load(connection_id)
        if record.kind == "none":
            raise RuntimeError(
                f"force_refresh called on no-auth connection {connection_id} — a no-auth entry has no token to refresh"
            )
        # A record already flagged RECONNECT_REQUIRED cannot be revived by another
        # exchange; fail fast before the token fence or cooldown so a known-dead
        # token is never burned and the terminal state is not re-persisted.
        if record.auth_health_state == AuthHealthState.RECONNECT_REQUIRED:
            raise ConnectorReconnectRequiredError(
                f"connection {connection_id} requires reconnect",
                connection_id=connection_id,
            )
        if failed_access_token is not None and record.access_token is not None:
            current = record.access_token.get_secret_value()
            if current != failed_access_token:
                # A peer rotated the token while we waited for the lock; serve
                # (or raise on) its record rather than refreshing again.
                return _serve_peer(record)
        if await refresh_cooldown_active(connection_id):
            raise ConnectorRefreshFailingError(
                f"connection {connection_id} is in refresh cooldown after a failing refresh",
                connection_id=connection_id,
            )
        return await _refresh(record, started_blob)
