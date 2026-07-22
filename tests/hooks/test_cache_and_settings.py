"""The manager-selection singleton (in-memory vs redis by config) and the
``HooksSettings`` derived keys / ``in_memory`` flag.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError
from tai42_contract.hooks.models import HookParams
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai42_skeleton.hooks.settings import HooksRedisSettings, HooksSettings


def _clear_singleton() -> None:
    get_hooks_manager.cache_clear()


def test_selects_in_memory_when_no_redis_url(monkeypatch):
    monkeypatch.delenv("HOOKS_REDIS_URL", raising=False)
    _clear_singleton()
    try:
        assert isinstance(get_hooks_manager(), InMemoryHooksManager)
    finally:
        _clear_singleton()


def test_selects_redis_when_redis_url_set(monkeypatch):
    monkeypatch.setenv("HOOKS_REDIS_URL", "redis://localhost:6379/0")
    _clear_singleton()
    try:
        assert isinstance(get_hooks_manager(), RedisHooksManager)
    finally:
        _clear_singleton()


def test_in_memory_selection_warns_loudly(monkeypatch, caplog):
    # In-memory mode silently no-ops sibling deliveries under multi-worker serving,
    # so its selection emits a loud once-per-process warning.
    monkeypatch.delenv("HOOKS_REDIS_URL", raising=False)
    _clear_singleton()
    try:
        with caplog.at_level(logging.WARNING, logger="tai42_skeleton.hooks.cache"):
            manager = get_hooks_manager()
        assert isinstance(manager, InMemoryHooksManager)
        assert "IN-MEMORY" in caplog.text
        assert "HOOKS_REDIS_URL" in caplog.text
    finally:
        _clear_singleton()


def test_redis_selection_does_not_warn(monkeypatch, caplog):
    monkeypatch.setenv("HOOKS_REDIS_URL", "redis://localhost:6379/0")
    _clear_singleton()
    try:
        with caplog.at_level(logging.WARNING, logger="tai42_skeleton.hooks.cache"):
            manager = get_hooks_manager()
        assert isinstance(manager, RedisHooksManager)
        assert "IN-MEMORY" not in caplog.text
    finally:
        _clear_singleton()


def test_reload_drops_the_hooks_manager_singleton(monkeypatch):
    # The singleton is registered with the settings-reset registry, so a live
    # reload rebuilds it against the new config (backend + connection) rather than
    # keeping the stale one bound to the old HooksSettings.
    monkeypatch.delenv("HOOKS_REDIS_URL", raising=False)
    _clear_singleton()
    try:
        first = get_hooks_manager()
        assert isinstance(first, InMemoryHooksManager)

        # Flip the backend config and reload every registered settings cache.
        monkeypatch.setenv("HOOKS_REDIS_URL", "redis://localhost:6379/0")
        reset_all_settings()

        rebuilt = get_hooks_manager()
        assert rebuilt is not first
        assert isinstance(rebuilt, RedisHooksManager)
    finally:
        _clear_singleton()


async def test_hook_count_reflects_live_in_memory_hooks():
    manager = InMemoryHooksManager(HooksSettings())
    assert manager.hook_count == 0
    await manager.register(HookParams(name="a", topic="orders", tool="ship"))
    await manager.register(HookParams(name="b", topic="orders", tool="notify"))
    await manager.register(HookParams(name="c", topic="refunds", tool="refund"))
    # Counts across every topic bucket.
    assert manager.hook_count == 3
    await manager.unregister("b")
    assert manager.hook_count == 2


async def test_reset_warns_naming_count_and_drops_in_memory_hooks(monkeypatch, caplog):
    # The in-memory manager honors HOOKS_* config, so a reload drops it — and its
    # in-memory hooks — with a loud warning naming the count (not a silent loss).
    monkeypatch.delenv("HOOKS_REDIS_URL", raising=False)
    _clear_singleton()
    try:
        manager = get_hooks_manager()
        assert isinstance(manager, InMemoryHooksManager)
        await manager.register(HookParams(name="h1", topic="orders", tool="ship"))
        assert manager.hook_count == 1

        with caplog.at_level(logging.WARNING, logger="tai42_skeleton.hooks.cache"):
            reset_all_settings()

        assert "discarding the in-memory hooks manager with 1 registered hook(s)" in caplog.text

        rebuilt = get_hooks_manager()
        assert rebuilt is not manager
        assert isinstance(rebuilt, InMemoryHooksManager)
        assert rebuilt.hook_count == 0
    finally:
        _clear_singleton()


async def test_reset_empty_in_memory_manager_emits_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("HOOKS_REDIS_URL", raising=False)
    _clear_singleton()
    try:
        assert isinstance(get_hooks_manager(), InMemoryHooksManager)  # no hooks registered
        with caplog.at_level(logging.WARNING, logger="tai42_skeleton.hooks.cache"):
            reset_all_settings()
        assert "discarding the in-memory hooks manager" not in caplog.text
    finally:
        _clear_singleton()


def test_reset_redis_mode_emits_no_warning(monkeypatch, caplog):
    # A redis-backed manager keeps its state in Redis, so a reset loses nothing and
    # must not warn.
    monkeypatch.setenv("HOOKS_REDIS_URL", "redis://localhost:6379/0")
    _clear_singleton()
    try:
        assert isinstance(get_hooks_manager(), RedisHooksManager)
        with caplog.at_level(logging.WARNING, logger="tai42_skeleton.hooks.cache"):
            reset_all_settings()
        assert "discarding the in-memory hooks manager" not in caplog.text
    finally:
        _clear_singleton()


def test_in_memory_flag_reflects_redis_url():
    assert HooksSettings().in_memory is True
    with_url = HooksSettings(redis=HooksRedisSettings(redis_url="redis://localhost:6379/0"))
    assert with_url.in_memory is False


def test_derived_keys():
    settings = HooksSettings()
    assert settings.get_hook_key("orders") == "hooks:topic:orders"
    assert settings.name_trigger_map_key == "hooks:name_trigger_map"


def test_trigger_key_helpers_round_trip():
    settings = HooksSettings()
    token_hash = "a" * 64

    # Each helper builds its documented ``hooks:trigger:rec:/name:/tomb:`` form.
    assert settings.trigger_record_key(token_hash) == f"hooks:trigger:rec:{token_hash}"
    assert settings.trigger_name_key("orders") == "hooks:trigger:name:orders"
    assert settings.trigger_tomb_key(token_hash) == f"hooks:trigger:tomb:{token_hash}"

    # The prefix properties match exactly what the key helpers prepend.
    assert settings.trigger_record_key_prefix == "hooks:trigger:rec:"
    assert settings.trigger_name_key_prefix == "hooks:trigger:name:"
    assert settings.trigger_tomb_key_prefix == "hooks:trigger:tomb:"
    assert settings.trigger_record_key(token_hash) == f"{settings.trigger_record_key_prefix}{token_hash}"
    assert settings.trigger_name_key("orders") == f"{settings.trigger_name_key_prefix}orders"
    assert settings.trigger_tomb_key(token_hash) == f"{settings.trigger_tomb_key_prefix}{token_hash}"

    # The scan patterns glob their own prefixes.
    assert settings.trigger_name_scan_pattern() == "hooks:trigger:name:*"
    assert settings.trigger_tomb_scan_pattern() == "hooks:trigger:tomb:*"


def test_max_workers_defaults_and_rejects_non_positive():
    # The global in-flight bound is always on: it defaults to 10 and a
    # non-positive value is a config error, never an unbounded mode.
    assert HooksSettings().max_workers == 10
    with pytest.raises(ValidationError):
        HooksSettings(max_workers=0)
    with pytest.raises(ValidationError):
        HooksSettings(max_workers=-1)
