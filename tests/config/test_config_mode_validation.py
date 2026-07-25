"""ConfigModeSettings validation: a non-string or an unknown mode is rejected
loudly, never coerced to a silent default. Also the cached ``config_mode``
accessor for the k8s value."""

from __future__ import annotations

import pytest

from tai42_skeleton.config.config_mode import ConfigMode, ConfigModeSettings, config_mode


def test_non_string_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid TAI_CONFIG_MODE"):
        ConfigModeSettings(config_mode=123)  # type: ignore[arg-type]


def test_unknown_string_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid TAI_CONFIG_MODE"):
        ConfigModeSettings(config_mode="vault")  # pyright: ignore[reportArgumentType]


def test_known_mode_is_normalized() -> None:
    # Stripped + lowercased before matching.
    settings = ConfigModeSettings(config_mode="  K8S ")  # pyright: ignore[reportArgumentType]
    assert settings.config_mode == ConfigMode.k8s


def test_config_mode_accessor_returns_k8s_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_CONFIG_MODE", "k8s")
    config_mode.cache_clear()
    try:
        assert config_mode() == "k8s"
    finally:
        config_mode.cache_clear()
