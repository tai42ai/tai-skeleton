"""The three de-mixed connector settings classes + their validators."""

from __future__ import annotations

import base64
from datetime import timedelta

import pytest
from pydantic import SecretStr, ValidationError
from tai_kit.settings import reset_all_settings

from tai_skeleton.connectors.settings import (
    ConnectorAdapterSettings,
    ConnectorCryptoSecrets,
    ConnectorEngineConfig,
    ConnectorStoreSettings,
    _require_key_bytes,
    _validate_b64_key,
    connector_adapter_settings,
    connector_crypto_secrets,
    connector_engine_config,
    connector_store_settings,
)

_KEK = base64.b64encode(bytes(32)).decode()
_HMAC = base64.b64encode(bytes(40)).decode()


# -- _validate_b64_key -------------------------------------------------------


def test_validate_b64_key_none_and_empty():
    assert _validate_b64_key(None, env_var="X", min_bytes=32) is None
    assert _validate_b64_key("", env_var="X", min_bytes=32) is None


def test_validate_b64_key_invalid_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        _validate_b64_key("!!!", env_var="X", min_bytes=32)


def test_validate_b64_key_exact_length_mismatch():
    short = base64.b64encode(bytes(16)).decode()
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        _validate_b64_key(short, env_var="X", min_bytes=32, exact=True)


def test_validate_b64_key_min_length_mismatch():
    short = base64.b64encode(bytes(16)).decode()
    with pytest.raises(ValueError, match="at least 32 bytes"):
        _validate_b64_key(short, env_var="X", min_bytes=32)


def test_validate_b64_key_ok():
    assert _validate_b64_key(_KEK, env_var="X", min_bytes=32, exact=True) == _KEK


# -- _require_key_bytes ------------------------------------------------------


def test_require_key_bytes_missing_raises():
    with pytest.raises(RuntimeError, match="not configured"):
        _require_key_bytes(None, env_var="CONNECTORS_KEK", what="encryption KEK")


def test_require_key_bytes_decodes():
    assert _require_key_bytes(_KEK, env_var="CONNECTORS_KEK", what="KEK") == bytes(32)


# -- ConnectorCryptoSecrets --------------------------------------------------


def test_crypto_secrets_validate_kek_wrong_length(monkeypatch):
    # Validation is deferred to the require accessor (not a pydantic validator, to
    # keep the key out of pydantic's error machinery); construction succeeds, the
    # use site raises value-free.
    monkeypatch.setenv("CONNECTORS_KEK", base64.b64encode(bytes(16)).decode())
    reset_all_settings()
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        ConnectorCryptoSecrets().require_kek_bytes()


def test_crypto_secrets_validate_hmac_too_short(monkeypatch):
    monkeypatch.setenv("CONNECTORS_STATE_HMAC_KEY", base64.b64encode(bytes(16)).decode())
    reset_all_settings()
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ConnectorCryptoSecrets().require_state_hmac_key_bytes()


def test_crypto_secrets_require_accessors():
    s = ConnectorCryptoSecrets(kek=SecretStr(_KEK), state_hmac_key=SecretStr(_HMAC))
    assert s.require_kek_bytes() == bytes(32)
    assert s.require_state_hmac_key_bytes() == bytes(40)


def test_crypto_secrets_require_raises_when_unset():
    s = ConnectorCryptoSecrets(kek=None, state_hmac_key=None)
    with pytest.raises(RuntimeError):
        s.require_kek_bytes()
    with pytest.raises(RuntimeError):
        s.require_state_hmac_key_bytes()


def test_crypto_secrets_masked_in_repr_and_dump():
    # SecretStr keeps the keys out of repr/logs/serialization; guards against a
    # silent downgrade back to plain ``str``.
    s = ConnectorCryptoSecrets(kek=SecretStr(_KEK), state_hmac_key=SecretStr(_HMAC))
    assert _KEK not in repr(s)
    assert _HMAC not in repr(s)
    assert _KEK not in s.model_dump_json()
    assert _HMAC not in s.model_dump_json()


def test_malformed_kek_error_does_not_leak_plaintext(monkeypatch):
    # A misconfigured key must not echo its plaintext anywhere in the raised
    # error. Validation lives in require_kek_bytes (a plain ValueError, not a
    # pydantic ValidationError), so the raw key never enters pydantic's error
    # ``input`` — which ``.errors()`` / the traceback would expose even with
    # hide_input_in_errors. The message carries only the env-var name + length.
    bad = base64.b64encode(b"sixteen-byteskey").decode()  # valid base64, wrong length (16B)
    monkeypatch.setenv("CONNECTORS_KEK", bad)
    reset_all_settings()
    with pytest.raises(ValueError, match="must decode to exactly") as ei:
        ConnectorCryptoSecrets().require_kek_bytes()
    assert bad not in str(ei.value)


def test_kek_ring_bytes_current_only():
    s = ConnectorCryptoSecrets(kek=SecretStr(_KEK))
    assert s.kek_ring_bytes() == [bytes(32)]


def test_kek_ring_bytes_includes_previous():
    prev = base64.b64encode(bytes(range(32))).decode()
    s = ConnectorCryptoSecrets(kek=SecretStr(_KEK), kek_previous=SecretStr(prev))
    assert s.kek_ring_bytes() == [bytes(32), bytes(range(32))]


def test_kek_ring_bytes_rejects_malformed_previous():
    bad_prev = base64.b64encode(bytes(16)).decode()  # valid base64, wrong length
    s = ConnectorCryptoSecrets(kek=SecretStr(_KEK), kek_previous=SecretStr(bad_prev))
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        s.kek_ring_bytes()


def test_connector_crypto_secrets_is_cached():
    assert connector_crypto_secrets() is connector_crypto_secrets()


# -- ConnectorEngineConfig ---------------------------------------------------


def test_max_session_ttl_default():
    assert ConnectorEngineConfig().max_session_ttl == timedelta(days=180)


def test_max_session_ttl_bare_integer_seconds(monkeypatch):
    monkeypatch.setenv("CONNECTORS_MAX_SESSION_TTL", "3600")
    reset_all_settings()
    assert ConnectorEngineConfig().max_session_ttl == timedelta(seconds=3600)


def test_max_session_ttl_iso8601(monkeypatch):
    monkeypatch.setenv("CONNECTORS_MAX_SESSION_TTL", "PT2H")
    reset_all_settings()
    assert ConnectorEngineConfig().max_session_ttl == timedelta(hours=2)


def test_max_session_ttl_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("CONNECTORS_MAX_SESSION_TTL", "0")
    reset_all_settings()
    with pytest.raises(ValidationError):
        ConnectorEngineConfig()


def test_redirect_uri_allowlist_origins_parsing(monkeypatch):
    monkeypatch.setenv(
        "CONNECTORS_REDIRECT_URI_ALLOWLIST",
        " https://a.test/ , https://b.test , ,https://c.test ",
    )
    reset_all_settings()
    assert ConnectorEngineConfig().redirect_uri_allowlist_origins == [
        "https://a.test",
        "https://b.test",
        "https://c.test",
    ]


def test_redirect_uri_allowlist_origins_empty():
    assert ConnectorEngineConfig(redirect_uri_allowlist="").redirect_uri_allowlist_origins == []


def test_oauth_bridge_url_defaults_none():
    assert ConnectorEngineConfig().oauth_bridge_url is None


def test_oauth_bridge_url_from_env(monkeypatch):
    monkeypatch.setenv("CONNECTORS_OAUTH_BRIDGE_URL", "https://bridge.tai42.ai")
    reset_all_settings()
    assert ConnectorEngineConfig().oauth_bridge_url == "https://bridge.tai42.ai"


def test_connector_engine_config_is_cached():
    assert connector_engine_config() is connector_engine_config()


# -- ConnectorStoreSettings --------------------------------------------------


def test_store_settings_defaults():
    s = ConnectorStoreSettings()
    assert s.key_prefix == "connectors:"
    assert s.redis.decode_responses is False
    assert s.redis.redis_url == "redis://localhost:6379/0"
    assert s.pg.pg_db == "tai"


def test_connector_store_settings_is_cached():
    assert connector_store_settings() is connector_store_settings()


# -- ConnectorAdapterSettings ------------------------------------------------


def test_adapter_settings_defaults():
    s = ConnectorAdapterSettings()
    assert s.meta_token_key == "tai_hub.access_token"
    assert s.error_prefix == "tai-hub-err:"


def test_adapter_settings_env_override(monkeypatch):
    monkeypatch.setenv("CONNECTORS_META_TOKEN_KEY", "custom.token")
    monkeypatch.setenv("CONNECTORS_ERROR_PREFIX", "custom-err:")
    reset_all_settings()
    s = ConnectorAdapterSettings()
    assert s.meta_token_key == "custom.token"
    assert s.error_prefix == "custom-err:"


def test_connector_adapter_settings_is_cached():
    assert connector_adapter_settings() is connector_adapter_settings()
