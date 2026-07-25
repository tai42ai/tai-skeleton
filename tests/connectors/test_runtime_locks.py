"""Per-connection Redis lock: acquire / release / best-effort degradation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pytest

import tai42_skeleton.connectors.runtime.locks as locks
from tai42_skeleton.connectors.runtime.locks import (
    _acquire,
    _lock_key,
    _refresh_cooldown_key,
    _release,
    clear_refresh_cooldown,
    connection_lock,
    open_refresh_cooldown,
    refresh_cooldown_active,
)

from .conftest import CID


class FakeRedis:
    """Single-slot SET NX PX lock + eval-based release."""

    def __init__(self, *, set_error=False, eval_error=False, held_by=None) -> None:
        self.value = held_by  # pre-existing holder token, or None
        self.set_error = set_error
        self.eval_error = eval_error
        self.eval_calls: list = []

    async def set(self, key, token, nx=False, px=None):
        if self.set_error:
            raise RuntimeError("redis down")
        if nx and self.value is not None:
            return False
        self.value = token
        return True

    async def eval(self, script, numkeys, key, token):
        self.eval_calls.append((key, token))
        if self.eval_error:
            raise RuntimeError("redis down")
        if self.value == token:
            self.value = None
            return 1
        return 0


@pytest.fixture
def install_redis(monkeypatch):
    def _install(redis):
        @asynccontextmanager
        async def fake_client_ctx(client_cls, settings=None, **kwargs):
            yield redis

        monkeypatch.setattr(locks, "client_ctx", fake_client_ctx)
        return redis

    return _install


def test_lock_key():
    assert _lock_key(CID) == f"connectors:lock:{CID}"


async def test_acquire_success_returns_token(install_redis):
    install_redis(FakeRedis())
    token = await _acquire(CID)
    assert token is not None


async def test_acquire_redis_error_returns_none(install_redis):
    install_redis(FakeRedis(set_error=True))
    assert await _acquire(CID) is None


async def test_acquire_times_out_proceeds_without_lock(install_redis, monkeypatch):
    """A perpetually-held lock makes the waiter give up and proceed unlocked."""
    install_redis(FakeRedis(held_by="someone-else"))

    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(locks.asyncio, "sleep", fake_sleep)
    # Collapse the deadline so the loop bails after one poll.
    monkeypatch.setattr(locks, "ACQUIRE_TIMEOUT_SECONDS", 0.0)
    assert await _acquire(CID) is None


async def test_release_with_token_deletes(install_redis):
    redis = install_redis(FakeRedis(held_by="tok"))
    await _release(CID, "tok")
    assert redis.value is None
    assert redis.eval_calls


async def test_release_none_token_noops(install_redis):
    redis = install_redis(FakeRedis())
    await _release(CID, None)
    assert redis.eval_calls == []


async def test_release_redis_error_is_swallowed_with_warning(install_redis, caplog):
    install_redis(FakeRedis(eval_error=True, held_by="tok"))
    with caplog.at_level(logging.WARNING):
        await _release(CID, "tok")  # no raise — lock will expire via PX
    # The swallowed release failure must surface as a WARNING, never silently.
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
    assert "redis error releasing lock" in caplog.text


async def test_connection_lock_acquires_and_releases(install_redis):
    redis = install_redis(FakeRedis())
    async with connection_lock(CID):
        assert redis.value is not None
    assert redis.value is None  # released on exit


async def test_connection_lock_releases_on_exception(install_redis):
    """A raise inside the ``async with`` body must still free the lock — the
    release runs on the exception path via the context manager's finally."""
    redis = install_redis(FakeRedis())

    async def hold_and_boom():
        async with connection_lock(CID):
            assert redis.value is not None  # held while inside the body
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await hold_and_boom()
    assert redis.value is None  # released despite the exception
    assert redis.eval_calls  # the release actually ran


async def test_connection_lock_proceeds_when_acquire_fails(install_redis):
    install_redis(FakeRedis(set_error=True))
    entered = False
    async with connection_lock(CID):
        entered = True
    assert entered  # body still ran despite no lock


# -- refresh cooldown / circuit breaker --------------------------------------


class FakeCooldownRedis:
    """Key-value redis modelling the refresh cooldown breaker."""

    def __init__(self, *, fail: bool = False) -> None:
        self.keys: dict[str, bytes] = {}
        self.set_calls: list[tuple[str, int | None]] = []
        self._fail = fail

    async def exists(self, key):
        if self._fail:
            raise RuntimeError("redis down")
        return 1 if key in self.keys else 0

    async def set(self, key, value, ex=None):
        if self._fail:
            raise RuntimeError("redis down")
        self.keys[key] = value
        self.set_calls.append((key, ex))

    async def delete(self, key):
        if self._fail:
            raise RuntimeError("redis down")
        self.keys.pop(key, None)


def test_refresh_cooldown_key():
    assert _refresh_cooldown_key(CID) == f"connectors:refresh_cooldown:{CID}"


async def test_open_then_active_then_clear(install_redis):
    redis = install_redis(FakeCooldownRedis())
    assert await refresh_cooldown_active(CID) is False
    await open_refresh_cooldown(CID)
    assert await refresh_cooldown_active(CID) is True
    # armed with the bounded TTL so it self-expires
    key, ex = redis.set_calls[-1]
    assert key == _refresh_cooldown_key(CID)
    assert ex == int(locks.REFRESH_COOLDOWN_SECONDS)
    await clear_refresh_cooldown(CID)
    assert await refresh_cooldown_active(CID) is False


async def test_cooldown_active_redis_error_is_fail_open(install_redis, caplog):
    install_redis(FakeCooldownRedis(fail=True))
    with caplog.at_level(logging.WARNING):
        # A Redis error is treated as "no cooldown" (fail-open like the lock),
        # surfaced as a WARNING — never silently.
        assert await refresh_cooldown_active(CID) is False
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
