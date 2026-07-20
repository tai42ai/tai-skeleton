"""Per-connection Redis lock used by the resolver and the service layer.

One short ``SET NX PX`` lock per connection serialises every read-modify-write
of a connection record across replicas. Use it as::

    async with connection_lock(connection_id):
        <read record, mutate, store.put>

The lock is best-effort: if Redis is unreachable or the wait times out, the
context manager logs a WARNING and proceeds WITHOUT the lock so a Redis outage
never wedges token refresh. That degradation is visible (WARNING), never silent.

Redis is reached through the app-pooled ``RedisClient`` (closed centrally at
shutdown).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.connectors.settings import connector_store_settings
from tai_skeleton.utils.redis_typing import awaited

logger = logging.getLogger(__name__)

# Lock lifetime. A normal refresh finishes in a few seconds, but a run that
# exhausts the transient-retry budget (5 attempts at the provider HTTP timeout +
# backoff) can exceed this TTL; if the key expires mid-refresh another replica
# may acquire the lock and refresh too. That race is made safe by the resolver's
# write-back fence (it discards a refresh whose refresh_token rotated under it),
# so even a rotating-refresh-token provider cannot have a stale loser clobber the
# winner's healthy record.
LOCK_TTL_SECONDS = 60.0
# How long a waiter polls for the lock before giving up and proceeding WITHOUT
# it (holder stuck/dead). Slightly above the TTL so a waiter outlasts a dead
# holder's key expiry rather than bailing one tick early.
ACQUIRE_TIMEOUT_SECONDS = 61.0
_POLL_INTERVAL_SECONDS = 0.1

# Compare-and-delete: only release if we still hold the lock (the stored value
# is still our token), so we never delete a lock a later holder acquired after
# our PX expired.
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


# Circuit-breaker cooldown after a refresh exhausts its retry budget (flips a
# connection to REFRESH_FAILING). While the breaker key lives, resolution
# fast-fails instead of re-burning the full 1+2+4+8s retry budget under the lock
# on every tool call (and stampeding lock-timeout waiters behind it). The key
# self-expires after the cooldown so a recovered provider is retried in bounded
# time.
REFRESH_COOLDOWN_SECONDS = 60.0


def _lock_key(connection_id: str) -> str:
    return f"{connector_store_settings().key_prefix}lock:{connection_id}"


def _refresh_cooldown_key(connection_id: str) -> str:
    return f"{connector_store_settings().key_prefix}refresh_cooldown:{connection_id}"


async def refresh_cooldown_active(connection_id: str) -> bool:
    """Whether a refresh cooldown breaker is currently open for the connection.

    Best-effort like the lock: a Redis error is logged at WARNING and treated as
    "no cooldown" (proceed to attempt the refresh) — a Redis outage already
    disables the connection lock too, so the fail-open posture is consistent and
    visible, never silent."""
    try:
        async with client_ctx(RedisClient, connector_store_settings().redis) as client:
            return bool(await awaited(client.exists(_refresh_cooldown_key(connection_id))))
    except Exception:
        logger.warning(
            "connectors: redis error checking refresh cooldown for %s — proceeding as if inactive",
            connection_id,
            exc_info=True,
        )
        return False


async def open_refresh_cooldown(connection_id: str) -> None:
    """Open the refresh cooldown breaker for :data:`REFRESH_COOLDOWN_SECONDS`.

    Best-effort: a Redis error is logged at WARNING (the storm-suppression is
    lost until the next failing refresh re-arms it), never silently swallowed."""
    try:
        async with client_ctx(RedisClient, connector_store_settings().redis) as client:
            await client.set(
                _refresh_cooldown_key(connection_id),
                b"1",
                ex=int(REFRESH_COOLDOWN_SECONDS),
            )
    except Exception:
        logger.warning(
            "connectors: redis error opening refresh cooldown for %s — fast-fail suppression not armed",
            connection_id,
            exc_info=True,
        )


async def clear_refresh_cooldown(connection_id: str) -> None:
    """Clear the refresh cooldown breaker after a successful refresh, so a
    connection whose fresh token is already inside the safety margin is not
    fast-failed by a still-live breaker. Best-effort (WARNING on error)."""
    try:
        async with client_ctx(RedisClient, connector_store_settings().redis) as client:
            await client.delete(_refresh_cooldown_key(connection_id))
    except Exception:
        logger.warning(
            "connectors: redis error clearing refresh cooldown for %s (it will expire via TTL)",
            connection_id,
            exc_info=True,
        )


async def _acquire(connection_id: str) -> str | None:
    """Poll ``SET NX PX`` until won or the wait deadline elapses.

    Returns the lock token on success, or ``None`` on timeout / Redis error —
    in which case the caller proceeds WITHOUT the lock.
    """
    token = secrets.token_hex(16)
    key = _lock_key(connection_id)
    px = int(LOCK_TTL_SECONDS * 1000)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + ACQUIRE_TIMEOUT_SECONDS
    while True:
        try:
            async with client_ctx(RedisClient, connector_store_settings().redis) as client:
                won = await client.set(key, token, nx=True, px=px)
        except Exception:  # lock is best-effort; never block writes
            logger.warning(
                "connectors: redis error acquiring lock for %s — proceeding without the connection lock",
                connection_id,
                exc_info=True,
            )
            return None
        if won:
            return token
        if loop.time() >= deadline:
            logger.warning(
                "connectors: timed out waiting for lock on %s — proceeding without the connection lock",
                connection_id,
            )
            return None
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def _release(connection_id: str, token: str | None) -> None:
    if token is None:
        return
    try:
        async with client_ctx(RedisClient, connector_store_settings().redis) as client:
            await awaited(client.eval(_RELEASE_LUA, 1, _lock_key(connection_id), token))
    except Exception:
        logger.warning(
            "connectors: redis error releasing lock for %s (it will expire via PX)",
            connection_id,
            exc_info=True,
        )


@asynccontextmanager
async def connection_lock(connection_id: str) -> AsyncIterator[None]:
    """Hold the per-connection lock for the duration of the ``async with`` body.

    Best-effort: a ``None`` token (Redis error / wait timeout) still enters the
    body — the WARNING from ``_acquire`` records that the body ran unlocked.
    """
    token = await _acquire(connection_id)
    try:
        yield
    finally:
        await _release(connection_id, token)
