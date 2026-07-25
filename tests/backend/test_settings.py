"""BackendSettings defaults + the cached accessor."""

from __future__ import annotations

from tai42_skeleton.backend.settings import BackendSettings, base_backend_settings


def test_backend_settings_defaults() -> None:
    settings = BackendSettings()
    assert settings.manifest_key == "MANIFEST_KEY"
    assert settings.task_timeout == 300
    assert settings.tool_name_arg == "backend_tool_name"


def test_base_backend_settings_is_cached() -> None:
    base_backend_settings.cache_clear()
    try:
        first = base_backend_settings()
        assert isinstance(first, BackendSettings)
        # The accessor is memoized: same instance on the next call.
        assert base_backend_settings() is first
    finally:
        base_backend_settings.cache_clear()
