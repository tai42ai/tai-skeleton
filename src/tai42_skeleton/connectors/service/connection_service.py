"""Connection lifecycle — start_connect, complete_connect, reconnect,
disconnect, patch_sub_services.

Single-namespace: a connection is keyed by its uuid4 ``connection_id`` alone
(globally unique). Every record read-modify-write runs under
``connection_lock`` so concurrent writers serialise on a lock-on-write basis.
The lock is best-effort (a Redis outage lets the body run unlocked), so each
read-modify-write additionally persists via compare-and-set on the ciphertext
it loaded: a writer that lost the race raises
:class:`ConcurrentConnectionUpdateError` instead of clobbering the peer's record
(e.g. a concurrently-rotated refresh token).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr
from tai42_contract.connectors.errors import ConnectorError
from tai42_contract.connectors.models import (
    AuthHealthState,
    ConnectionRecord,
    check_alias,
)
from tai42_contract.connectors.providers import ProviderDescriptor
from tai42_contract.connectors.service import (
    AliasInUseError,
    CompleteConnectResult,
    DisconnectResult,
    NoAuthConnectResult,
    PatchResult,
    StartConnectResult,
)

from tai42_skeleton.config.service import ConfigService
from tai42_skeleton.connectors.oauth import client as oauth_client
from tai42_skeleton.connectors.oauth import crypto, redirect, state
from tai42_skeleton.connectors.providers.registry import get_provider
from tai42_skeleton.connectors.runtime.launch import SUPPORTED_MANAGED_TRANSPORTS, resolve_mcp_server
from tai42_skeleton.connectors.runtime.locks import clear_refresh_cooldown, connection_lock
from tai42_skeleton.connectors.service import manifest_writer
from tai42_skeleton.connectors.store import token_store
from tai42_skeleton.connectors.store.persistence import (
    ConnectionNotFoundError,
    load_record,
    load_record_with_blob,
    session_expires_at_for,
)

# Re-exported so callers import the lifecycle errors from one place.
__all__ = ["AliasInUseError", "ConcurrentConnectionUpdateError", "ConnectionNotFoundError"]

logger = logging.getLogger(__name__)


class ConcurrentConnectionUpdateError(ConnectorError):
    """A read-modify-write lost its compare-and-set: a concurrent writer
    rotated the stored record between this operation's load and its persist.
    Nothing was written — the caller should re-read the connection and retry
    the operation against its current state."""

    def __init__(self, connection_id: str):
        super().__init__(
            f"connection {connection_id} was modified concurrently; "
            f"nothing was written — re-read the connection and retry"
        )
        self.connection_id = connection_id


# -- Validation helpers ---------------------------------------------------


_RETURN_URL_RE = re.compile(r"^/[A-Za-z0-9_\-./?=&%]*\Z")


def _validate_return_url(value: str) -> str:
    # ``//evil.com`` is a protocol-relative URL the browser treats as absolute;
    # the regex permits a leading ``/`` so reject the ``//`` prefix explicitly to
    # close the open-redirect.
    if value.startswith("//") or not _RETURN_URL_RE.match(value):
        raise ValueError(f"return_url must be a same-origin path beginning with '/': {value!r}")
    return value


def _validate_sub_services(descriptor: ProviderDescriptor, sub_services: list[str]) -> None:
    if not sub_services:
        raise ValueError("enabled_sub_services must be non-empty")
    extra = set(sub_services) - set(descriptor.sub_services.keys())
    if extra:
        raise ValueError(f"unknown sub-services for provider {descriptor.id!r}: {sorted(extra)}")
    # Reject a sub-service on a transport managed calls cannot drive (e.g.
    # websocket) at connect time: it would probe healthy but raise on every real
    # tool call, so the user must never be able to complete a Connect for it.
    for sub_id in sub_services:
        server = resolve_mcp_server(descriptor, sub_id)
        if server.type not in SUPPORTED_MANAGED_TRANSPORTS:
            raise ValueError(
                f"sub-service {sub_id!r} of provider {descriptor.id!r} uses transport "
                f"{server.type!r}, which connector-managed calls do not support "
                f"(supported: {sorted(SUPPORTED_MANAGED_TRANSPORTS)})"
            )


def _scopes_for(descriptor: ProviderDescriptor, sub_services: Iterable[str]) -> list[str]:
    return sorted({scope for sub_id in sub_services for scope in descriptor.sub_services[sub_id].scopes})


async def _persist(
    record: ConnectionRecord,
    *,
    create_only: bool = False,
    expected_blob: bytes | None = None,
) -> None:
    """Encrypt the record and write it through the token store.

    ``expected_blob`` is the ciphertext the read-modify-write loaded from; the
    store commits only if the stored blob still equals it (atomic
    compare-and-set). A CAS miss means a concurrent writer rotated the record —
    possibly including a rotated refresh token — so overwriting would corrupt
    it; raise :class:`ConcurrentConnectionUpdateError` instead.
    """
    blob = crypto.encrypt(
        record.to_storage_json().encode("utf-8"),
        connection_id=record.connection_id,
    )
    # provider_id/alias back the durable UNIQUE (provider_id, alias) constraint;
    # a create-only insert that collides raises AliasInUseError from the store (the
    # authority for per-provider alias uniqueness).
    committed = await token_store().put(
        record.connection_id,
        blob,
        create_only=create_only,
        expected_blob=expected_blob,
        session_expires_at=session_expires_at_for(record),
        provider_id=record.provider_id,
        alias=record.alias,
    )
    if expected_blob is not None and not committed:
        raise ConcurrentConnectionUpdateError(record.connection_id)


# -- start_connect / start_reconnect --------------------------------------


def _validate_config_values(descriptor: ProviderDescriptor, config_values: dict[str, str]) -> None:
    """Validate client-supplied config_values against the provider's config_fields.

    Raises on an unknown key or a missing/empty required value — never silently
    drops or defaults.
    """
    allowed = {field.key for field in descriptor.config_fields}
    unknown = set(config_values) - allowed
    if unknown:
        raise ValueError(f"unknown config values for provider {descriptor.id!r}: {sorted(unknown)}")
    for field in descriptor.config_fields:
        if field.required and not config_values.get(field.key):
            raise ValueError(f"missing required config value {field.key!r} for provider {descriptor.id!r}")


async def _start_flow(
    *,
    descriptor: ProviderDescriptor,
    alias: str,
    enabled_sub_services: list[str],
    requested_scopes: list[str],
    return_url: str,
    redirect_uri: str,
    origin: str,
    operation: state.FlowOperation,
    reconnect_connection_id: str | None,
) -> StartConnectResult:
    """Persist an OAuthFlowState and build the provider authorize URL.

    ``state`` is a signed envelope carrying the single-use ``flow_id`` (CSRF
    guard) and ``origin`` — this deployment's own origin, which a callback routed
    through the central OAuth bridge reads to bounce the code back here. The origin
    is validated against the redirect allow-list before it is signed, so a spoofed
    ``Origin`` header cannot mint a deployment-signed state pointing off-list.
    """
    redirect.validate_origin_allowed(origin)
    verifier, challenge = oauth_client.generate_pkce_pair()
    flow_id = str(uuid.uuid4())
    flow_state = state.OAuthFlowState(
        flow_id=flow_id,
        provider_id=descriptor.id,
        alias=alias,
        requested_scopes=requested_scopes,
        enabled_sub_services=list(enabled_sub_services),
        pkce_verifier=verifier,
        return_url=return_url,
        redirect_uri=redirect_uri,
        operation=operation,
        reconnect_connection_id=reconnect_connection_id,
    )
    await state.put(flow_state)

    authorize_url = oauth_client.build_authorize_url(
        descriptor=descriptor,
        scopes=requested_scopes,
        state=state.encode(flow_id=flow_id, origin=origin),
        code_challenge=challenge,
        redirect_uri=redirect_uri,
    )
    return StartConnectResult(flow_id=flow_id, authorize_url=authorize_url)


async def start_connect(
    *,
    provider_id: str,
    alias: str,
    enabled_sub_services: list[str],
    config_values: dict[str, str] | None = None,
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> StartConnectResult | NoAuthConnectResult:
    """Begin a new Connect.

    OAuth provider: validates inputs, persists OAuthFlowState, returns the
    provider's authorize URL. No-auth provider: validates inputs + config_values,
    creates the connection immediately and returns its id (no OAuth flow).

    The Origin is validated only on the OAuth branch, inside ``_start_flow``
    before it signs the redirect state — the no-auth branch has no redirect flow,
    so it is deliberately not gated on the redirect allow-list."""
    config_values = config_values or {}
    try:
        descriptor = get_provider(provider_id)
    except KeyError as exc:
        raise ValueError(f"unknown provider: {provider_id!r}") from exc

    # Alias — shared shape with ConnectionRecord + UI Zod schema. check_alias
    # already raises ValueError with a clear message on a bad shape.
    check_alias(alias)

    _validate_sub_services(descriptor, enabled_sub_services)
    _validate_return_url(return_url)

    # Alias uniqueness is enforced durably by the store's UNIQUE (provider_id,
    # alias) constraint at persist time (create-only insert → AliasInUseError); a
    # pre-check here would still race two concurrent same-alias connects.

    if descriptor.kind == "none":
        return await _connect_no_auth(
            descriptor=descriptor,
            alias=alias,
            enabled_sub_services=enabled_sub_services,
            config_values=config_values,
        )

    if config_values:
        raise ValueError(f"config_values are not accepted for oauth provider {provider_id!r}")
    return await _start_flow(
        descriptor=descriptor,
        alias=alias,
        enabled_sub_services=enabled_sub_services,
        requested_scopes=_scopes_for(descriptor, enabled_sub_services),
        return_url=return_url,
        redirect_uri=redirect_uri,
        origin=origin,
        operation=state.FlowOperation.CONNECT,
        reconnect_connection_id=None,
    )


async def _connect_no_auth(
    *,
    descriptor: ProviderDescriptor,
    alias: str,
    enabled_sub_services: list[str],
    config_values: dict[str, str],
) -> NoAuthConnectResult:
    """Create a no-auth connection: no OAuth flow, no token. Validate the client
    config, persist a minimal record, write the managed manifest entries."""
    _validate_config_values(descriptor, config_values)

    connection_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    record = ConnectionRecord(
        connection_id=connection_id,
        provider_id=descriptor.id,
        kind="none",
        alias=alias,
        enabled_sub_services=list(enabled_sub_services),
        config_values={key: SecretStr(value) for key, value in config_values.items()},
        auth_health_state=AuthHealthState.HEALTHY,
        created_at=now,
    )

    # Persist + manifest-add under the connection lock (the id is freshly
    # generated, so no contention) so a concurrent disconnect cannot delete the
    # record between the persist and the add and strand the added entries.
    added: list[str] = []

    def add_entries(document: dict[str, Any]) -> None:
        added[:] = manifest_writer.add_managed_entries(
            document,
            descriptor=descriptor,
            enabled_sub_services=enabled_sub_services,
            alias=alias,
            connection_id=connection_id,
        )

    async with connection_lock(connection_id):
        await _persist(record, create_only=True)
        # Append the managed entries through the single pipeline: it validates,
        # persists, locally reloads, and broadcasts the reload to the whole fleet.
        applied = await ConfigService.from_app().apply_change(add_entries)

    return NoAuthConnectResult(
        connection_id=connection_id,
        added_manifest_entries=added,
        fanout=applied.fanout,
    )


async def start_reconnect(
    *,
    connection_id: str,
    enabled_sub_services: list[str],
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> StartConnectResult:
    """Re-run the OAuth flow for an existing connection (add scopes / recover
    from RECONNECT_REQUIRED). On callback the new tokens replace the old in
    :func:`complete_connect`.

    Reconnect is always an OAuth redirect, so the Origin is validated inside
    ``_start_flow`` before it signs the redirect state; nothing is persisted or
    mutated before that call."""
    record = await load_record(connection_id)
    if record.kind == "none":
        raise ValueError(f"no-auth connection {connection_id} cannot be reconnected")
    try:
        descriptor = get_provider(record.provider_id)
    except KeyError as exc:
        # The provider plugin was removed since this connection was created; surface
        # a typed ValueError the router maps to a 4xx instead of a raw KeyError 500.
        raise ValueError(f"unknown provider: {record.provider_id!r}") from exc

    _validate_sub_services(descriptor, enabled_sub_services)
    _validate_return_url(return_url)

    return await _start_flow(
        descriptor=descriptor,
        alias=record.alias,
        enabled_sub_services=enabled_sub_services,
        requested_scopes=_scopes_for(descriptor, enabled_sub_services),
        return_url=return_url,
        redirect_uri=redirect_uri,
        origin=origin,
        operation=state.FlowOperation.RECONNECT,
        reconnect_connection_id=connection_id,
    )


# -- complete_connect -----------------------------------------------------


async def complete_connect(
    *,
    flow_id: str,
    code: str,
) -> CompleteConnectResult:
    """Exchange code for tokens, persist the encrypted ConnectionRecord, and
    write managed manifest entries. The token-exchange ``redirect_uri`` is the
    one stored in the flow state at authorize-start (byte-identical per RFC
    6749), never recomputed from the completion request."""
    flow_state = await state.get_and_delete(flow_id)
    if flow_state is None:
        raise oauth_client.OAuthError("state mismatch: no flow record found for the given flow_id")

    try:
        descriptor = get_provider(flow_state.provider_id)
    except KeyError as exc:
        # The provider plugin was removed between authorize-start and completion;
        # surface it as a typed OAuthError so the router maps it to a 4xx failed
        # body instead of a raw 500.
        raise oauth_client.OAuthError(f"provider {flow_state.provider_id!r} is no longer registered") from exc

    # Defence in depth: re-validate the stored redirect_uri against the allow-list
    # at the token exchange, not only at authorize-start, so a redirect_uri that
    # fell off the allow-list since authorize-start cannot drive an exchange.
    oauth_client.validate_redirect_uri(flow_state.redirect_uri)

    # On RECONNECT / TOGGLE the provider may not re-issue a refresh_token; the
    # existing one is inherited in _complete_reconnect_or_toggle.
    require_refresh_token = flow_state.operation == state.FlowOperation.CONNECT
    token_resp = await oauth_client.exchange_code(
        descriptor=descriptor,
        code=code,
        code_verifier=flow_state.pkce_verifier,
        redirect_uri=flow_state.redirect_uri,
        require_refresh_token=require_refresh_token,
    )

    if flow_state.operation in (
        state.FlowOperation.RECONNECT,
        state.FlowOperation.TOGGLE_SUBSERVICE_ON,
    ):
        return await _complete_reconnect_or_toggle(
            flow_state=flow_state,
            descriptor=descriptor,
            token_resp=token_resp,
        )

    account_identity = _extract_account_identity(token_resp.raw) or "unknown"

    connection_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    record = ConnectionRecord(
        connection_id=connection_id,
        provider_id=flow_state.provider_id,
        kind="oauth",
        alias=flow_state.alias,
        account_identity=account_identity,
        enabled_sub_services=list(flow_state.enabled_sub_services),
        granted_scopes=token_resp.granted_scopes or list(flow_state.requested_scopes),
        access_token=SecretStr(token_resp.access_token),
        refresh_token=SecretStr(token_resp.refresh_token) if token_resp.refresh_token is not None else None,
        access_token_expires_at=token_resp.expires_at,
        auth_health_state=AuthHealthState.HEALTHY,
        created_at=now,
    )

    # Persist + manifest-add under the connection lock (the id is freshly
    # generated, so no contention) so a concurrent disconnect cannot delete the
    # record between the persist and the add and strand the added entries.
    added: list[str] = []

    def add_entries(document: dict[str, Any]) -> None:
        added[:] = manifest_writer.add_managed_entries(
            document,
            descriptor=descriptor,
            enabled_sub_services=flow_state.enabled_sub_services,
            alias=flow_state.alias,
            connection_id=connection_id,
        )

    async with connection_lock(connection_id):
        # create_only guards against a duplicate connection_id (fresh UUID, so a
        # collision is a hard error rather than a silent overwrite). A collision on
        # the durable (provider_id, alias) uniqueness raises AliasInUseError here.
        try:
            await _persist(record, create_only=True)
        except Exception:
            # The just-issued grant is brand-new and unshared; the completion
            # failed before it was persisted, so revoke it upstream rather than
            # orphan a live consent. Best-effort, never masks the original error.
            await _revoke_fresh_grant(descriptor, token_resp)
            raise

        # Append the managed entries through the single pipeline: it validates,
        # persists, locally reloads, and broadcasts the reload to the whole fleet.
        applied = await ConfigService.from_app().apply_change(add_entries)

    return CompleteConnectResult(
        connection_id=connection_id,
        return_url=flow_state.return_url,
        operation=flow_state.operation,
        added_manifest_entries=added,
        removed_manifest_entries=[],
        fanout=applied.fanout,
    )


async def _revoke_fresh_grant(descriptor: ProviderDescriptor, token_resp: oauth_client.TokenResponse) -> None:
    """Best-effort upstream revoke of a brand-new, unshared grant whose fresh
    Connect failed after the code exchange.

    Only ever called on the fresh-CONNECT path: that ``refresh_token`` is unshared
    (a reconnect/toggle would inherit an existing record's token, so revoking it
    could kill a live connection — those paths deliberately do NOT revoke). Logged,
    never raises, never masks the caller's original error."""
    if token_resp.refresh_token is None:
        return
    try:
        outcome = await oauth_client.revoke(descriptor=descriptor, token=token_resp.refresh_token)
        logger.warning(
            "connectors: revoked orphaned fresh grant for provider %s (outcome=%s)",
            descriptor.id,
            outcome.outcome,
        )
    except Exception:
        logger.warning(
            "connectors: best-effort revoke of orphaned fresh grant for provider %s raised",
            descriptor.id,
            exc_info=True,
        )


async def _complete_reconnect_or_toggle(
    *,
    flow_state: state.OAuthFlowState,
    descriptor: ProviderDescriptor,
    token_resp: oauth_client.TokenResponse,
) -> CompleteConnectResult:
    """Replace tokens + scopes + enabled_sub_services on an existing record and
    reconcile its managed manifest entries, under the connection lock."""
    cid = flow_state.reconnect_connection_id
    if not cid:
        raise oauth_client.OAuthError(f"flow operation {flow_state.operation} requires reconnect_connection_id")

    removed: list[str] = []
    added: list[str] = []
    async with connection_lock(cid):
        record, started_blob = await load_record_with_blob(cid)

        record.access_token = SecretStr(token_resp.access_token)
        record.access_token_expires_at = token_resp.expires_at
        if token_resp.refresh_token:
            record.refresh_token = SecretStr(token_resp.refresh_token)
        # A provider that omits ``scope`` on the exchange is treated as granting
        # exactly what this flow requested (mirrors the CONNECT path), never as
        # revoking every scope.
        record.granted_scopes = list(token_resp.granted_scopes or flow_state.requested_scopes)
        record.auth_health_state = AuthHealthState.HEALTHY

        # Reconcile enabled_sub_services + manifest in lock-step. A RECONNECT may
        # carry a smaller set than before (validated non-empty + known, but not a
        # superset), so de-selected sub-services lose their entries too. For
        # TOGGLE_SUBSERVICE_ON to_remove is empty — patch_sub_services already
        # applied removals before forking to the consent flow.
        new_enabled = sorted(set(flow_state.enabled_sub_services))
        prior_enabled = set(record.enabled_sub_services)
        to_add = set(new_enabled) - prior_enabled
        to_remove = prior_enabled - set(new_enabled)
        record.enabled_sub_services = new_enabled

        # Compare-and-set against the ciphertext this operation loaded: the
        # lock is best-effort, so a peer (e.g. a token refresh that rotated the
        # refresh token) may have written meanwhile — losing the CAS raises
        # rather than clobbering the peer's record.
        await _persist(record, expected_blob=started_blob)

        # Recovery: an explicit reconnect restores fresh tokens + HEALTHY, so drop
        # any refresh-cooldown breaker a prior failing run armed — otherwise a
        # fresh token already inside the safety margin would be fast-failed by the
        # still-live cooldown until it expires.
        await clear_refresh_cooldown(cid)

        # Reconcile the managed manifest in ONE pipeline transaction (remove +
        # append in a single mutator ⇒ one persist, one reload, one broadcast)
        # INSIDE the lock so a concurrent disconnect (which also takes the lock)
        # cannot delete the connection between this persist and the reconcile and
        # strand the added entries.
        def reconcile(document: dict[str, Any]) -> None:
            removed[:] = (
                manifest_writer.remove_managed_entries(document, connection_id=cid, sub_services=to_remove)
                if to_remove
                else []
            )
            added[:] = (
                manifest_writer.add_managed_entries(
                    document,
                    descriptor=descriptor,
                    enabled_sub_services=to_add,
                    alias=record.alias,
                    connection_id=cid,
                )
                if to_add
                else []
            )

        # No delta ⇒ no apply_change ran, so there is no broadcast to report; the
        # fanout is honestly ``None`` (this path mutated no manifest).
        fanout: dict[str, Any] | None = None
        if to_remove or to_add:
            fanout = (await ConfigService.from_app().apply_change(reconcile)).fanout

    return CompleteConnectResult(
        connection_id=cid,
        return_url=flow_state.return_url,
        operation=flow_state.operation,
        added_manifest_entries=added,
        removed_manifest_entries=removed,
        fanout=fanout,
    )


# -- disconnect -----------------------------------------------------------


async def disconnect(
    *,
    connection_id: str,
) -> DisconnectResult:
    """Disconnect a connection: best-effort upstream revoke, then purge the
    encrypted blob and remove managed manifest entries.

    Runs under the connection lock so an in-flight ``patch`` / reconnect (which
    reconciles the manifest under the same lock) cannot add managed entries after
    this disconnect removed them — otherwise those entries would be stranded
    against a deleted connection.
    """
    async with connection_lock(connection_id):
        # Cleanup must reach an EXPIRED connection too: a lapsed session still has
        # an encrypted blob + managed manifest entries that only disconnect can
        # purge, so load with include_expired (serving reads keep filtering).
        record = await load_record(connection_id, include_expired=True)

        removed: list[str] = []

        def remove_entries(document: dict[str, Any]) -> None:
            removed[:] = manifest_writer.remove_managed_entries(document, connection_id=connection_id)

        # No-auth has no token to revoke and no provider creds to read — skip the
        # upstream revoke (and the catalog get_provider lookup) entirely, before
        # the token access below would crash on a None refresh_token.
        if record.kind == "none":
            # Remove the managed entries through the single pipeline (validate +
            # persist + reload + broadcast) before purging the blob.
            applied = await ConfigService.from_app().apply_change(remove_entries)
            await token_store().delete(connection_id)
            return DisconnectResult(
                connection_id=connection_id,
                upstream_revoke_outcome="skipped",
                upstream_revoke_status=None,
                removed_manifest_entries=removed,
                fanout=applied.fanout,
            )

        if record.refresh_token is None:
            # An oauth record without a refresh token is corrupt — fail loudly
            # rather than send an empty revocation upstream.
            raise ConnectorError(f"oauth record {connection_id} has no refresh_token")
        try:
            descriptor = get_provider(record.provider_id)
        except KeyError:
            # The provider plugin was removed, so there is nothing to revoke
            # against — but the local blob + manifest entries must still be
            # purgeable. Skip the upstream revoke and proceed with local removal.
            logger.warning(
                "connectors: provider %r for connection %s is no longer registered; "
                "skipping upstream revoke and purging locally",
                record.provider_id,
                connection_id,
            )
            revoke_outcome = oauth_client.RevokeOutcome(outcome="skipped")
        else:
            revoke_outcome = await oauth_client.revoke(
                descriptor=descriptor,
                token=record.refresh_token.get_secret_value(),
            )

        # Remove manifest entries first so the tool stops being callable
        # immediately, then purge the blob. revoke() already ran above, so a
        # crash between the two leaves only an already-revoked (dead) orphan
        # token blob — harmless, reclaimed by the session-TTL cache expiry or a
        # re-disconnect. The removal crosses the single pipeline (validate +
        # persist + reload + broadcast).
        applied = await ConfigService.from_app().apply_change(remove_entries)

        # "user clicked Disconnect" revokes access even if a peer just wrote new
        # tokens. delete is idempotent.
        await token_store().delete(connection_id)

    return DisconnectResult(
        connection_id=connection_id,
        upstream_revoke_outcome=revoke_outcome.outcome,
        upstream_revoke_status=revoke_outcome.http_status,
        removed_manifest_entries=removed,
        fanout=applied.fanout,
    )


# -- patch_sub_services ---------------------------------------------------


async def patch_sub_services(
    *,
    connection_id: str,
    desired: list[str],
    return_url: str,
    redirect_uri: str,
    origin: str,
) -> PatchResult:
    """Toggle which sub-services are enabled.

    Sub-services toggled OFF lose their manifest entries. Sub-services toggled
    ON whose scopes are already granted are recreated inline; otherwise a
    ``TOGGLE_SUBSERVICE_ON`` flow is started and the authorize URL returned for a
    consent popup that :func:`complete_connect` finalises.
    """
    _validate_return_url(return_url)

    # Mutate the record under the lock and BEFORE touching the manifest, so a
    # losing writer never tears manifest entries out against a record it failed
    # to persist.
    async with connection_lock(connection_id):
        record, started_blob = await load_record_with_blob(connection_id)
        try:
            descriptor = get_provider(record.provider_id)
        except KeyError as exc:
            # The provider plugin was removed since this connection was created;
            # surface a typed ValueError the router maps to a 4xx, not a raw 500.
            raise ValueError(f"unknown provider: {record.provider_id!r}") from exc
        _validate_sub_services(descriptor, desired)

        current = set(record.enabled_sub_services)
        desired_set = set(desired)
        if current == desired_set:
            raise ValueError("enabled_sub_services unchanged")

        to_remove = current - desired_set
        to_add = desired_set - current

        granted = set(record.granted_scopes)
        needs_consent_subs: list[str] = []
        can_inline_subs: list[str] = []
        for sub_id in to_add:
            # A no-auth connection has no OAuth endpoints, so there is no
            # consent flow to fork — every toggle is inline regardless of any
            # scopes the descriptor declares.
            if record.kind == "none" or set(descriptor.sub_services[sub_id].scopes).issubset(granted):
                can_inline_subs.append(sub_id)
            else:
                needs_consent_subs.append(sub_id)

        # A consent-requiring toggle forks an OAuth flow whose signed state
        # carries this origin, so fail closed on an off-list Origin here — after
        # the pure consent/inline split, before the inline persist below — so a
        # spoofed Origin never commits a partial sub-service change. An
        # inline-only toggle has no redirect flow and is deliberately not gated.
        if needs_consent_subs:
            redirect.validate_origin_allowed(origin)

        new_enabled = sorted((current - to_remove) | set(can_inline_subs))
        record.enabled_sub_services = new_enabled
        # Compare-and-set against the ciphertext this operation loaded: the
        # lock is best-effort, so a peer (e.g. a token refresh that rotated the
        # refresh token) may have written meanwhile — losing the CAS raises
        # rather than clobbering the peer's record.
        await _persist(record, expected_blob=started_blob)

        # Reconcile the inline manifest changes in ONE pipeline transaction
        # (remove + append in a single mutator ⇒ one persist, one reload, one
        # broadcast) INSIDE the lock so a concurrent disconnect (which also takes
        # the lock) cannot delete the connection between this persist and the
        # reconcile and strand the added entries against a deleted connection.
        removed: list[str] = []
        added: list[str] = []

        def reconcile(document: dict[str, Any]) -> None:
            removed[:] = (
                manifest_writer.remove_managed_entries(document, connection_id=connection_id, sub_services=to_remove)
                if to_remove
                else []
            )
            added[:] = (
                manifest_writer.add_managed_entries(
                    document,
                    descriptor=descriptor,
                    enabled_sub_services=can_inline_subs,
                    alias=record.alias,
                    connection_id=connection_id,
                )
                if can_inline_subs
                else []
            )

        # A consent-only toggle (nothing to remove, nothing inline-addable) makes no
        # manifest change here, so no apply_change runs and the fanout is honestly
        # ``None``; the forked consent flow's own completion reports its broadcast.
        fanout: dict[str, Any] | None = None
        if to_remove or can_inline_subs:
            fanout = (await ConfigService.from_app().apply_change(reconcile)).fanout

    if not needs_consent_subs:
        return PatchResult(
            connection_id=connection_id,
            enabled_sub_services=new_enabled,
            consent_required=False,
            flow_id=None,
            authorize_url=None,
            added_manifest_entries=added,
            removed_manifest_entries=removed,
            fanout=fanout,
        )

    # Consent fork: carry both already-enabled and newly-requested sub-services
    # so the callback produces a complete record.
    consent_subs = sorted(set(new_enabled) | set(needs_consent_subs))
    result = await _start_flow(
        descriptor=descriptor,
        alias=record.alias,
        enabled_sub_services=consent_subs,
        requested_scopes=_scopes_for(descriptor, consent_subs),
        return_url=return_url,
        redirect_uri=redirect_uri,
        origin=origin,
        operation=state.FlowOperation.TOGGLE_SUBSERVICE_ON,
        reconnect_connection_id=connection_id,
    )
    return PatchResult(
        connection_id=connection_id,
        enabled_sub_services=new_enabled,
        consent_required=True,
        flow_id=result.flow_id,
        authorize_url=result.authorize_url,
        added_manifest_entries=added,
        removed_manifest_entries=removed,
        fanout=fanout,
    )


# -- helpers --------------------------------------------------------------


def _extract_account_identity(raw_token_payload: dict) -> str | None:
    """Recover the connected account's email from an id_token (best-effort).

    The id_token is decoded WITHOUT signature verification — it was just returned
    over the TLS channel we initiated, so its origin is trusted enough for
    display only. Auth decisions rely on the access token.
    """
    id_token = raw_token_payload.get("id_token")
    if not id_token:
        return None
    try:
        parts = id_token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("email")
    except Exception:
        logger.warning(
            "connectors: id_token parse for account identity failed",
            exc_info=True,
        )
        return None


# -- Protocol conformance -------------------------------------------------

if TYPE_CHECKING:
    # This module IS the ConnectionService implementation (free functions, no
    # implementing class). Bind those functions as staticmethods so pyright checks
    # them structurally against the contract Protocol: a signature drift (e.g. a
    # missing kw-only ``origin``) fails the ``_CONFORMS`` assignment below at
    # type-check time.
    from tai42_contract.connectors.service import ConnectionService as _ConnectionServiceProtocol

    class _ModuleConnectionService:
        start_connect = staticmethod(start_connect)
        start_reconnect = staticmethod(start_reconnect)
        complete_connect = staticmethod(complete_connect)
        disconnect = staticmethod(disconnect)
        patch_sub_services = staticmethod(patch_sub_services)

    _CONFORMS: _ConnectionServiceProtocol = _ModuleConnectionService()
