"""App-root settings cache accessors.

Each accessor reads its source setting once and is memoized by
``settings_cache``. The tests clear the per-accessor cache around each
assertion so an env override is read fresh and never leaks into other tests,
and they confirm the trim/lowercase normalization the provider accessors apply.
"""

from __future__ import annotations

import pytest

from tai42_skeleton.settings import cache
from tai42_skeleton.settings.settings import AppArgsSettings


def test_manifest_path_reads_core_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_MANIFEST_PATH", "/etc/tai/manifest.yaml")
    cache.manifest_path.cache_clear()
    try:
        assert cache.manifest_path() == "/etc/tai/manifest.yaml"
    finally:
        cache.manifest_path.cache_clear()


def test_manifest_path_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    cache.manifest_path.cache_clear()
    try:
        assert cache.manifest_path() is None
    finally:
        cache.manifest_path.cache_clear()


def test_backend_provider_is_trimmed_and_lowercased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_MCP_BACKEND", "  Redis  ")
    cache.backend_provider.cache_clear()
    try:
        assert cache.backend_provider() == "redis"
    finally:
        cache.backend_provider.cache_clear()


def test_backend_provider_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MCP_BACKEND", raising=False)
    cache.backend_provider.cache_clear()
    try:
        assert cache.backend_provider() == ""
    finally:
        cache.backend_provider.cache_clear()


def test_template_provider_is_trimmed_and_lowercased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_MCP_TEMPLATE", "  Jinja  ")
    cache.template_provider.cache_clear()
    try:
        assert cache.template_provider() == "jinja"
    finally:
        cache.template_provider.cache_clear()


def test_mcp_probe_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MCP_MCP_PROBE_TIMEOUT", raising=False)
    cache.mcp_probe_timeout.cache_clear()
    try:
        assert cache.mcp_probe_timeout() == 15.0
    finally:
        cache.mcp_probe_timeout.cache_clear()


def test_app_args_settings_returns_model() -> None:
    cache.app_args_settings.cache_clear()
    try:
        a = cache.app_args_settings()
        b = cache.app_args_settings()
        assert isinstance(a, AppArgsSettings)
        # Memoized: the second call returns the very same cached instance.
        assert a is b
    finally:
        cache.app_args_settings.cache_clear()
