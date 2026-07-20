"""Pydantic-settings for the Connectors engine.

The engine config is de-mixed into three co-located settings, so a secret never
shares a class with plain feature config:

  * :class:`ConnectorCryptoSecrets` (``CONNECTORS_*``) — the two secrets the
    engine signs/encrypts with: the AES-GCM token KEK and the OAuth state-HMAC
    key.
  * :class:`ConnectorEngineConfig` (``CONNECTORS_*``) — non-secret engine knobs:
    the session/token-cache TTL cap and the OAuth redirect-URI allow-list.
  * :class:`ConnectorStoreSettings` (``CONNECTOR_STORE_*``) — the token-store
    backend, composing the kit Redis + Postgres connection settings so the store
    reaches both through the pooled clients, plus the connector key prefix.

Alongside the engine, :class:`ConnectorAdapterSettings` (``CONNECTORS_*``) carries
the wire-format contract between the outbound MCP adapter and the connector-
launched servers — the ``_meta`` token key and the structured-error prefix.

Provider-specific credentials/endpoints (google/atlassian client ids, MCP-server
distribution) are NOT here — they ship with the provider plugins.
"""

from __future__ import annotations

import base64
from datetime import timedelta

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict
from tai_kit.clients import PostgresConnectionSettings, RedisConnectionSettings
from tai_kit.settings import TaiBaseSettings, settings_cache

_KEK_BYTE_LENGTH = 32
_STATE_HMAC_MIN_BYTE_LENGTH = 32


def _validate_b64_key(value: str | None, *, env_var: str, min_bytes: int, exact: bool = False) -> str | None:
    """Validate a base64 secret's decoded length, or raise.

    ``exact`` requires exactly ``min_bytes``; otherwise at least ``min_bytes``.
    Empty/unset returns None so the catalog endpoint can render before the
    secret is provisioned; encrypt/sign call sites must require it then.
    """
    if value is None or value == "":
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise ValueError(f"{env_var} is not valid base64") from exc
    if exact and len(decoded) != min_bytes:
        raise ValueError(f"{env_var} must decode to exactly {min_bytes} bytes (got {len(decoded)})")
    if not exact and len(decoded) < min_bytes:
        raise ValueError(f"{env_var} must decode to at least {min_bytes} bytes (got {len(decoded)})")
    return value


def _require_key_bytes(value: str | None, *, env_var: str, what: str) -> bytes:
    """Decode a required base64 secret to bytes, raising loudly if unset."""
    if not value:
        raise RuntimeError(
            f"Connectors {what} is not configured: set the {env_var} env var "
            f"(base64-encoded random bytes). Generate one with: "
            f"python -c 'import secrets, base64; "
            f"print(base64.b64encode(secrets.token_bytes(32)).decode())'"
        )
    return base64.b64decode(value, validate=True)


# -- Secrets -----------------------------------------------------------------


class ConnectorCryptoSecrets(TaiBaseSettings):
    """The engine's two secrets: the token-blob KEK and the OAuth state-HMAC key.

    Both are ``None`` until provisioned; the ``require_*`` accessors raise at the
    encrypt/sign call site when a secret is actually needed.
    """

    model_config = SettingsConfigDict(env_prefix="CONNECTORS_")

    # Base64 32-byte key for AES-GCM token-blob encryption. SecretStr keeps it
    # out of repr/logs/tracebacks; the require_* accessors validate + reveal it
    # only when the engine actually encrypts/signs.
    kek: SecretStr | None = None

    # Optional base64 32-byte PREVIOUS KEK, retained across a rotation so blobs
    # written under the old key still decrypt. Encryption always uses ``kek``;
    # this key is decrypt-only, dropped once every blob has been re-encrypted.
    kek_previous: SecretStr | None = None

    # Base64 key (>=32 bytes) signing the OAuth ``state`` param.
    state_hmac_key: SecretStr | None = None

    # NB: base64/length validation is deferred to require_*_bytes (below), NOT a
    # pydantic field/model validator. A validator failure captures the raw key as
    # the error's ``input`` — exposed via ``ValidationError.errors()`` and the
    # traceback even with ``hide_input_in_errors`` — which would defeat SecretStr.
    # Validating at the use site keeps the plaintext out of pydantic entirely; the
    # raises there carry only the env-var name + expected length.

    def require_kek_bytes(self) -> bytes:
        """Return the decoded KEK, or raise (value-free) if unset/malformed."""
        kek = self.kek.get_secret_value() if self.kek is not None else None
        _validate_b64_key(kek, env_var="CONNECTORS_KEK", min_bytes=_KEK_BYTE_LENGTH, exact=True)
        return _require_key_bytes(kek, env_var="CONNECTORS_KEK", what="encryption KEK")

    def kek_ring_bytes(self) -> list[bytes]:
        """Return the decryption key-ring: the current KEK first, then the
        optional previous KEK. Encryption always uses the first entry; the
        previous KEK lets blobs written before a rotation still decrypt.

        Raises (value-free) if the current KEK is unset/malformed, or if a
        supplied previous KEK is malformed.
        """
        ring = [self.require_kek_bytes()]
        previous = self.kek_previous.get_secret_value() if self.kek_previous is not None else None
        if previous:
            _validate_b64_key(previous, env_var="CONNECTORS_KEK_PREVIOUS", min_bytes=_KEK_BYTE_LENGTH, exact=True)
            ring.append(_require_key_bytes(previous, env_var="CONNECTORS_KEK_PREVIOUS", what="previous encryption KEK"))
        return ring

    def require_state_hmac_key_bytes(self) -> bytes:
        """Return the decoded state-HMAC key, or raise (value-free) if unset/malformed."""
        hmac_key = self.state_hmac_key.get_secret_value() if self.state_hmac_key is not None else None
        _validate_b64_key(hmac_key, env_var="CONNECTORS_STATE_HMAC_KEY", min_bytes=_STATE_HMAC_MIN_BYTE_LENGTH)
        return _require_key_bytes(hmac_key, env_var="CONNECTORS_STATE_HMAC_KEY", what="state HMAC key")


@settings_cache
def connector_crypto_secrets() -> ConnectorCryptoSecrets:
    return ConnectorCryptoSecrets()


# -- Engine config -----------------------------------------------------------


class ConnectorEngineConfig(TaiBaseSettings):
    """Non-secret engine knobs: session-TTL cap, redirect-URI allow-list, and the
    optional central OAuth-bridge origin."""

    model_config = SettingsConfigDict(env_prefix="CONNECTORS_")

    # Record lifetime. Used by the redis-pg store as both the redis cache TTL
    # and the postgres dead-session bound. It is a CAP, reset on every
    # successful use/refresh — effectively an inactivity window.
    # A bare integer env value is seconds (15552000 -> 180d); ISO-8601
    # duration strings also parse.
    max_session_ttl: timedelta = timedelta(days=180)

    # Comma-separated origins (scheme://host[:port]) allowed as OAuth redirect
    # origins. Empty rejects every Connect flow — no implicit fallback.
    redirect_uri_allowlist: str = ""

    # Optional central OAuth-bridge origin (scheme://host[:port]). When set, the
    # provider redirect URI points at this shared bridge instead of the request
    # origin; the bridge bounces the code back to the originating deployment using
    # the origin signed into the OAuth ``state``. The bridge origin must ALSO
    # appear in ``redirect_uri_allowlist`` or every Connect flow fails closed at
    # ``validate_redirect_uri``. Unset = the provider redirects to this deployment
    # directly (the single-deployment default).
    oauth_bridge_url: str | None = None

    @field_validator("max_session_ttl", mode="before")
    @classmethod
    def _coerce_max_session_ttl(cls, value: object) -> object:
        # pydantic's timedelta parser rejects a bare digit string; coerce it
        # to seconds. ISO-8601 / numeric / timedelta inputs pass through.
        if isinstance(value, str) and value.strip().lstrip("+").isdigit():
            return timedelta(seconds=int(value.strip()))
        return value

    @field_validator("max_session_ttl")
    @classmethod
    def _validate_max_session_ttl(cls, value: timedelta) -> timedelta:
        if value.total_seconds() <= 0:
            raise ValueError(
                "CONNECTORS_MAX_SESSION_TTL must be a positive duration "
                "(seconds as an integer, e.g. 15552000, or an ISO-8601 string)"
            )
        return value

    @property
    def redirect_uri_allowlist_origins(self) -> list[str]:
        return [item.strip().rstrip("/") for item in self.redirect_uri_allowlist.split(",") if item.strip()]


@settings_cache
def connector_engine_config() -> ConnectorEngineConfig:
    return ConnectorEngineConfig()


# -- Token-store backend -----------------------------------------------------


class ConnectorStoreRedisSettings(RedisConnectionSettings):
    """``CONNECTOR_STORE_*`` redis connection for the token cache + lock.

    A dedicated namespace, kept separate from the access-control / app redis.
    Token blobs / lock tokens are raw bytes, so ``decode_responses`` is False.
    """

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_STORE_")

    redis_url: str | None = "redis://localhost:6379/0"
    redis_max_connections: int | None = 10
    decode_responses: bool = False

    # A black-holed Redis fails the token-store op loudly within 5s instead of
    # hanging the request/loop task: the connect phase and each command read are
    # both bounded. Must be positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = Field(default=5, gt=0)


class ConnectorStorePgSettings(PostgresConnectionSettings):
    """``CONNECTOR_STORE_*`` postgres connection for the ``connector_connections``
    durable source of truth. No baked-in credential — supply the password via
    ``CONNECTOR_STORE_PG_PASSWORD``."""

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_STORE_")

    pg_db: str = "tai"


class ConnectorStoreSettings(TaiBaseSettings):
    """Token-store backend: the composed kit Redis + Postgres connection
    settings plus the connector key prefix."""

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_STORE_")

    # Infra: the connections are composed from the kit (fields, not bases), so
    # the store config declares no connection fields of its own.
    redis: ConnectorStoreRedisSettings = Field(default_factory=ConnectorStoreRedisSettings)
    pg: ConnectorStorePgSettings = Field(default_factory=ConnectorStorePgSettings)

    # Namespace prefix for every connector key (record cache + lock).
    key_prefix: str = "connectors:"


@settings_cache
def connector_store_settings() -> ConnectorStoreSettings:
    return ConnectorStoreSettings()


# -- Adapter wire-format contract --------------------------------------------


class ConnectorAdapterSettings(TaiBaseSettings):
    """Wire-format contract between the outbound MCP adapter and the
    connector-launched MCP servers: the JSON-RPC ``_meta`` key the adapter writes
    the per-call access token into, and the prefix those servers stamp on
    structured ``ToolError`` payloads.

    Both are read lazily by ``token_injection``, so an env override
    (``CONNECTORS_META_TOKEN_KEY`` / ``CONNECTORS_ERROR_PREFIX``) takes effect as
    long as it is set before first use. The adapter and the launched server must
    agree on both values, or a managed call fails to inject the token / parse the
    error payload.
    """

    model_config = SettingsConfigDict(env_prefix="CONNECTORS_")

    #: JSON-RPC ``_meta`` key the adapter writes the per-call access token into.
    meta_token_key: str = "tai_hub.access_token"
    #: Prefix the connector servers stamp on ``ToolError`` strings to carry a
    #: structured payload; the adapter strips it before JSON-decoding.
    error_prefix: str = "tai-hub-err:"


@settings_cache
def connector_adapter_settings() -> ConnectorAdapterSettings:
    return ConnectorAdapterSettings()
