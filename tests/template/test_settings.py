"""TemplateCacheSettings: the empty/none coercion validator and the cached
accessor + defaults."""

from __future__ import annotations

import pytest

from tai42_skeleton.template.settings import TemplateCacheSettings, template_cache_settings


def test_defaults() -> None:
    settings = TemplateCacheSettings()
    assert settings.ttl == 60 * 5
    assert settings.max_size == 256


@pytest.mark.parametrize("raw", ["", "none", "null", "undefined", "NONE", "Null"])
def test_empty_or_none_strings_coerce_to_none(raw: str) -> None:
    settings = TemplateCacheSettings(ttl=raw, max_size=raw)  # type: ignore[arg-type]
    assert settings.ttl is None
    assert settings.max_size is None


def test_numeric_strings_pass_through() -> None:
    settings = TemplateCacheSettings(ttl="30", max_size="4")  # type: ignore[arg-type]
    assert settings.ttl == 30
    assert settings.max_size == 4


def test_accessor_is_cached() -> None:
    template_cache_settings.cache_clear()
    try:
        first = template_cache_settings()
        assert isinstance(first, TemplateCacheSettings)
        assert template_cache_settings() is first
    finally:
        template_cache_settings.cache_clear()
