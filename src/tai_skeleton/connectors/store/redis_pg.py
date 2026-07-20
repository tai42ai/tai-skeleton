"""Redis-cached, Postgres-durable concrete :class:`ConnectorTokenStore`.

Postgres (table ``connector_connections``, defined in the skeleton init SQL) is
the durable source of truth; Redis is a hot cache in front of it so a large
fleet never opens a Postgres connection per request. ``get`` reads Redis first
and falls back to Postgres on a miss (repopulating the cache); ``put`` commits
to Postgres and then refreshes Redis.

Single-namespace: each deployment owns its store, so ``connection_id`` (a uuid4)
is globally unique on its own and keys every record — Redis key ``rec:{cid}``,
Postgres primary key ``(connection_id)``.

The store treats the blob as opaque ciphertext — the AES-GCM boundary lives at
``tai_skeleton.connectors.oauth.crypto``. Both Redis and Postgres hold
ciphertext only.

Cache coherence is fenced by a per-row monotonic ``cache_version``: every
durable write bumps it and returns the new value, and the Redis record carries
it alongside the blob. Both the ``put`` cache refresh and the cache-miss
read-populate apply their write ONLY when the incoming version is newer than the
cached one (an atomic set-if-newer in Lua). A reader that loaded an old snapshot
from Postgres therefore can never overwrite a newer entry a concurrent writer
already installed — the cache can never regress to a value older than the latest
durable write, so it never serves an expired token past a completed refresh. A
delete is the latest durable write too: it leaves a version-fenced tombstone (a
version-only marker, no blob) at the deleted version, so a read-populate that
loaded the row just before the delete cannot resurrect it into the cache.

If a cache write itself fails after the durable commit, a stale prior entry may
still sit in Redis; the store then deletes the key so the next read repopulates
from Postgres, and if that delete also fails it logs LOUDLY (the record could be
served stale until its ``session_expires_at`` TTL elapses).

``provider_id`` and ``alias`` are persisted as plaintext columns purely to back
the ``UNIQUE (provider_id, alias)`` constraint. Every insert path — a create-only
insert or a plain upsert — writes them, so an upsert overwrites the stored
identity to match the incoming one; either path that collides on the constraint
(a second connection claiming an ``(provider_id, alias)`` a different connection
already holds) raises :class:`AliasInUseError` (the durable authority for
per-provider alias uniqueness).

``session_expires_at`` (supplied by the caller, who can read the decrypted
record) is persisted as a column and is the durable dead-session bound: ``get``
and ``list`` read only rows whose ``session_expires_at`` is null or still in the
future, so a connection idle past its session/refresh-token death stops serving
even from Postgres. The same column drives the Redis ``EXPIREAT`` so the cache
evicts around the same time; a cache-miss repopulates with the remaining lifetime
read back from Postgres.

Both backends are reached through the app-pooled ``PostgresClient`` /
``RedisClient`` so the connection pools are shared (same-DSN callers — token /
catalog / sources stores — share one Postgres pool) and closed centrally at app
shutdown.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import cast

from psycopg.errors import UniqueViolation
from tai_contract.connectors.errors import ConnectorError
from tai_contract.connectors.service import AliasInUseError
from tai_contract.connectors.store import ConnectorTokenStore
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.connectors.settings import connector_store_settings
from tai_skeleton.utils.redis_typing import eval_script

logger = logging.getLogger(__name__)

_BLOB_FIELD = b"blob"
_VER_FIELD = b"ver"

# Name of the durable ``UNIQUE (provider_id, alias)`` constraint (see the
# skeleton init SQL). A create-only insert that trips it is an alias collision,
# distinct from a ``connection_id`` primary-key conflict.
_ALIAS_UNIQUE_CONSTRAINT = "connector_connections_provider_alias_unique"

# Version-fenced cache write: install the blob + version ONLY when the record is
# absent or the incoming version is newer than the cached one, then (re)apply the
# key's expiry — all in one atomic server-side step. A late read-populate holding
# an older version is a no-op, so the cache never regresses to a value older than
# the latest durable write.
#   KEYS[1] = record key
#   ARGV[1] = version field, ARGV[2] = blob field,
#   ARGV[3] = new blob, ARGV[4] = new version,
#   ARGV[5] = expireat unix-seconds, or "" for no expiry (persist)
_CACHE_SET_IF_NEWER_LUA = """
local curver = redis.call('HGET', KEYS[1], ARGV[1])
if curver == false or tonumber(curver) < tonumber(ARGV[4]) then
    redis.call('HSET', KEYS[1], ARGV[2], ARGV[3], ARGV[1], ARGV[4])
    if ARGV[5] == '' then
        redis.call('PERSIST', KEYS[1])
    else
        redis.call('EXPIREAT', KEYS[1], ARGV[5])
    end
    return 1
end
return 0
"""

# Seconds a delete tombstone lives in Redis. A tombstone only needs to outlast an
# in-flight ``get`` that loaded the row from Postgres just before the delete (a
# sub-second-to-seconds window); this margin covers a slow/retrying read while
# staying short enough that the tiny marker auto-cleans well before it matters.
_TOMBSTONE_TTL_SECONDS = 300

# Version-fenced delete tombstone: replace the record with a version-only marker
# (no blob) so a concurrent read-populate holding the just-deleted version cannot
# resurrect the entry. Unlike the set-if-newer write, the fence is ``<=``: the
# tombstone must overwrite an equal-version blob a racing ``get`` may have already
# installed, and connection ids are never reused so no legitimate write at this
# version can follow. A subsequent ``get`` finds no blob field → cache miss →
# Postgres (now empty) → ``None``.
#   KEYS[1] = record key
#   ARGV[1] = version field, ARGV[2] = deleted version,
#   ARGV[3] = expireat unix-seconds
_CACHE_TOMBSTONE_LUA = """
local curver = redis.call('HGET', KEYS[1], ARGV[1])
if curver == false or tonumber(curver) <= tonumber(ARGV[2]) then
    redis.call('DEL', KEYS[1])
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
    redis.call('EXPIREAT', KEYS[1], ARGV[3])
    return 1
end
return 0
"""


def _expireat_arg(session_expires_at: datetime | None) -> int | None:
    """Absolute unix-seconds for Redis ``EXPIREAT``, or ``None`` for no expiry."""
    if session_expires_at is None:
        return None
    if session_expires_at.tzinfo is None:
        session_expires_at = session_expires_at.replace(tzinfo=UTC)
    return int(session_expires_at.timestamp())


class RedisPgConnectorTokenStore(ConnectorTokenStore):
    """Redis-cached, Postgres-durable token store."""

    def __init__(self) -> None:
        settings = connector_store_settings()
        self._settings = settings
        self._key_prefix = settings.key_prefix

    # -- key helpers ---------------------------------------------------------

    def _rec_key(self, connection_id: str) -> str:
        return f"{self._key_prefix}rec:{connection_id}"

    @staticmethod
    def _as_uuid(connection_id: str) -> uuid.UUID:
        try:
            return uuid.UUID(connection_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(f"connection_id is not a valid UUID: {connection_id!r}") from exc

    async def _cache_set(
        self,
        connection_id: str,
        blob: bytes,
        session_expires_at: datetime | None,
        version: int,
    ) -> None:
        """Version-fenced cache refresh after a durable commit.

        The write is a set-if-newer (atomic Lua): it installs ``blob`` + version
        only when the cached record is absent or older, so a stale late
        write-back cannot overwrite a cache entry a peer already moved forward.

        A Redis failure here would otherwise leave a stale prior entry in the
        cache; that must never mask an already-durable Postgres write, so we log
        a WARNING, then delete the key so the next ``get`` repopulates from
        Postgres. If that delete also fails, we log LOUDLY (ERROR) — the record
        may be served stale until its ``session_expires_at`` TTL elapses."""
        key = self._rec_key(connection_id)
        expire_at = _expireat_arg(session_expires_at)
        try:
            async with client_ctx(RedisClient, self._settings.redis) as client:
                await eval_script(
                    client,
                    _CACHE_SET_IF_NEWER_LUA,
                    1,
                    key,
                    _VER_FIELD,
                    _BLOB_FIELD,
                    blob,
                    version,
                    "" if expire_at is None else expire_at,
                )
        except Exception:
            logger.warning(
                "RedisPgConnectorTokenStore: cache write failed for %s "
                "(durable write already committed) — invalidating the cache entry "
                "so the next read repopulates from Postgres",
                connection_id,
                exc_info=True,
            )
            await self._invalidate_after_failed_write(connection_id)

    async def _invalidate_after_failed_write(self, connection_id: str) -> None:
        """Drop a possibly-stale cache entry after a failed cache write. If the
        delete itself fails the entry may linger until its TTL, so log LOUDLY."""
        try:
            async with client_ctx(RedisClient, self._settings.redis) as client:
                await client.delete(self._rec_key(connection_id))
        except Exception:
            logger.error(
                "RedisPgConnectorTokenStore: cache write AND its invalidation both "
                "failed for %s — the cache may serve a stale record until its "
                "session TTL elapses",
                connection_id,
                exc_info=True,
            )

    # -- ConnectorTokenStore API ---------------------------------------------

    async def get(self, connection_id: str, *, include_expired: bool = False) -> bytes | None:
        # 1) Redis hot path. A cached entry is only ever an unexpired record (its
        # key carries the session EXPIREAT, so an expired record has already been
        # evicted), so this hot hit is valid for both callers — include_expired
        # only needs to widen the durable fallback below.
        try:
            async with client_ctx(RedisClient, self._settings.redis) as client:
                # redis-py 7.x stubs type hget's field as ``str`` and its value as
                # ``str``; a bytes field is accepted and the undecoded value comes
                # back as bytes at runtime.
                blob = await cast(
                    "Awaitable[bytes | None]",
                    client.hget(self._rec_key(connection_id), _BLOB_FIELD),  # pyright: ignore[reportArgumentType]
                )
        except Exception:  # degrade to Postgres on cache error
            logger.warning(
                "RedisPgConnectorTokenStore: cache read failed for %s — falling back to Postgres",
                connection_id,
                exc_info=True,
            )
            blob = None
        if blob is not None:
            return bytes(blob)

        # 2) Postgres source of truth; repopulate cache (version-fenced) on hit.
        # The default read hides an expired record (serving reads / list must
        # never surface a lapsed session); the cleanup path (include_expired)
        # drops that filter so disconnect can still load and purge it.
        expiry_filter = "" if include_expired else "AND (session_expires_at IS NULL OR session_expires_at > now())"
        async with (
            client_ctx(PostgresClient, self._settings.pg) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT encrypted_blob, session_expires_at, cache_version "
                "FROM connector_connections WHERE connection_id = %s " + expiry_filter,
                (self._as_uuid(connection_id),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        blob, session_expires_at, version = bytes(row[0]), row[1], int(row[2])
        # Never repopulate the hot cache with an already-expired record loaded
        # solely for cleanup — its EXPIREAT is in the past (Redis would evict it
        # immediately) and no serving read may see it. An unexpired record (the
        # only kind the default filter yields) always repopulates.
        expire_epoch = _expireat_arg(session_expires_at)
        if expire_epoch is None or expire_epoch > int(datetime.now(UTC).timestamp()):
            await self._cache_set(connection_id, blob, session_expires_at, version)
        return blob

    async def put(
        self,
        connection_id: str,
        blob: bytes,
        *,
        create_only: bool = False,
        expected_blob: bytes | None = None,
        session_expires_at: datetime | None = None,
        provider_id: str | None = None,
        alias: str | None = None,
    ) -> bool:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("blob must be bytes")
        blob = bytes(blob)
        if create_only and expected_blob is not None:
            raise ValueError("put: create_only and expected_blob are mutually exclusive")
        if expected_blob is not None:
            expected_blob = bytes(expected_blob)
        conn_uuid = self._as_uuid(connection_id)

        # An INSERT path (create-only or plain upsert) writes the plaintext
        # provider_id/alias columns that back the uniqueness constraint, so both
        # are required there. A compare-and-set is a pure UPDATE and never
        # touches them.
        inserts = create_only or expected_blob is None
        if inserts and (provider_id is None or alias is None):
            raise ValueError("put: provider_id and alias are required for an insert (create-only or upsert)")

        async with (
            client_ctx(PostgresClient, self._settings.pg) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            if create_only:
                try:
                    await cur.execute(
                        "INSERT INTO connector_connections "
                        "(connection_id, provider_id, alias, encrypted_blob, session_expires_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (connection_id) DO NOTHING "
                        "RETURNING cache_version",
                        (conn_uuid, provider_id, alias, blob, session_expires_at),
                    )
                except UniqueViolation as exc:
                    # A connection_id conflict is caught by ON CONFLICT (returns no
                    # row); a UniqueViolation that escapes is the alias constraint.
                    if getattr(exc.diag, "constraint_name", None) == _ALIAS_UNIQUE_CONSTRAINT:
                        raise AliasInUseError(
                            f"alias {alias!r} is already in use for provider {provider_id!r}"
                        ) from exc
                    raise
                row = await cur.fetchone()
                if row is None:
                    raise ConnectorError(f"create-only put: record already exists for {connection_id}")
                version = int(row[0])
            elif expected_blob is not None:
                # Atomic compare-and-set on the durable source of truth: commit
                # only if the stored ciphertext still equals the blob the caller
                # refreshed from, bumping the version. 0 rows ⇒ a peer rotated it
                # first (or it's gone) ⇒ CAS miss, caller lost.
                await cur.execute(
                    "UPDATE connector_connections "
                    "SET encrypted_blob = %s, "
                    "    session_expires_at = %s, "
                    "    cache_version = cache_version + 1, "
                    "    updated_at = now() "
                    "WHERE connection_id = %s "
                    "  AND encrypted_blob = %s "
                    "RETURNING cache_version",
                    (blob, session_expires_at, conn_uuid, expected_blob),
                )
                row = await cur.fetchone()
                if row is None:
                    return False
                version = int(row[0])
            else:
                # A plain upsert writes the plaintext provider_id/alias on both
                # the insert and the update branch (EXCLUDED.*), so an update
                # overwrites the stored identity to match the incoming one; the
                # ``UNIQUE (provider_id, alias)`` constraint can therefore trip
                # here too when the new alias is already held by another
                # connection, surfaced as AliasInUseError like the create-only path.
                try:
                    await cur.execute(
                        "INSERT INTO connector_connections "
                        "(connection_id, provider_id, alias, encrypted_blob, session_expires_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (connection_id) DO UPDATE "
                        "SET provider_id = EXCLUDED.provider_id, "
                        "    alias = EXCLUDED.alias, "
                        "    encrypted_blob = EXCLUDED.encrypted_blob, "
                        "    session_expires_at = EXCLUDED.session_expires_at, "
                        "    cache_version = connector_connections.cache_version + 1, "
                        "    updated_at = now() "
                        "RETURNING cache_version",
                        (conn_uuid, provider_id, alias, blob, session_expires_at),
                    )
                except UniqueViolation as exc:
                    if getattr(exc.diag, "constraint_name", None) == _ALIAS_UNIQUE_CONSTRAINT:
                        raise AliasInUseError(
                            f"alias {alias!r} is already in use for provider {provider_id!r}"
                        ) from exc
                    raise
                row = await cur.fetchone()
                if row is None:
                    # An upsert with RETURNING always yields a row; a None here is
                    # a broken driver/contract, not a normal outcome — fail loudly.
                    raise ConnectorError(f"upsert put returned no cache_version for {connection_id}")
                version = int(row[0])

        await self._cache_set(connection_id, blob, session_expires_at, version)
        return True

    async def delete(self, connection_id: str) -> None:
        conn_uuid = self._as_uuid(connection_id)
        async with (
            client_ctx(PostgresClient, self._settings.pg) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "DELETE FROM connector_connections WHERE connection_id = %s RETURNING cache_version",
                (conn_uuid,),
            )
            row = await cur.fetchone()
        deleted_version = int(row[0]) if row is not None else None

        # Clear the cache entry (best-effort — the PG delete is authoritative). A
        # plain ``DEL`` alone would race a concurrent read-populate: a ``get`` that
        # loaded the row from Postgres just before this delete could re-install it
        # afterwards and serve it until its TTL. When a row was actually deleted we
        # instead write a version-fenced tombstone at the deleted version, which
        # blocks that stale re-install; if nothing was deleted here there is no
        # version to fence and a plain key drop suffices.
        key = self._rec_key(connection_id)
        try:
            async with client_ctx(RedisClient, self._settings.redis) as client:
                if deleted_version is None:
                    await client.delete(key)
                else:
                    expire_at = int((datetime.now(UTC) + timedelta(seconds=_TOMBSTONE_TTL_SECONDS)).timestamp())
                    await eval_script(
                        client,
                        _CACHE_TOMBSTONE_LUA,
                        1,
                        key,
                        _VER_FIELD,
                        deleted_version,
                        expire_at,
                    )
        except Exception:  # cache delete must never mask a commit
            # The durable delete stands; a lingering warm entry here is a
            # now-revoked token, so escalate exactly like a failed write-back —
            # try a plain key drop, and if that also fails log LOUDLY (ERROR).
            # A plain drop trades away the tombstone's resurrection fence, but
            # clearing a revoked token outweighs that.
            logger.warning(
                "RedisPgConnectorTokenStore: cache tombstone write failed for %s "
                "(durable delete already committed) — dropping the cache key so a "
                "revoked token is not served",
                connection_id,
                exc_info=True,
            )
            await self._invalidate_after_failed_write(connection_id)

    async def list(self) -> list[str]:
        # Postgres-authoritative: the primary key covers the listing read as an
        # index-only scan, it's a low-frequency path (catalog / connections
        # list), and it sidesteps the cold-vs-empty ambiguity a Redis index set
        # would introduce.
        async with (
            client_ctx(PostgresClient, self._settings.pg) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT connection_id FROM connector_connections "
                "WHERE session_expires_at IS NULL OR session_expires_at > now() "
                "ORDER BY connection_id"
            )
            rows = await cur.fetchall()
        return [str(r[0]) for r in rows]
