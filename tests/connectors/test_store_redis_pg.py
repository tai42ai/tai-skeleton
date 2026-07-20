"""RedisPgConnectorTokenStore: round-trip, cache fallback, error degradation,
version-fenced cache coherence, and durable alias uniqueness."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg.errors import UniqueViolation
from tai42_contract.connectors.errors import ConnectorError
from tai42_contract.connectors.service import AliasInUseError
from tai42_kit.clients.impl.postgres import PostgresClient
from tai42_kit.clients.impl.redis import RedisClient

import tai42_skeleton.connectors.store.redis_pg as redis_pg
from tai42_skeleton.connectors.store.redis_pg import (
    _ALIAS_UNIQUE_CONSTRAINT,
    RedisPgConnectorTokenStore,
    _expireat_arg,
)

from .conftest import CID, CID2

_BLOB_FIELD = b"blob"
_VER_FIELD = b"ver"


def _expired(exp) -> bool:
    """A ``session_expires_at`` in the past, matching the durable read's now() filter."""
    return exp is not None and exp <= datetime.now(UTC)


class _AliasUniqueViolation(UniqueViolation):
    """A UniqueViolation carrying the alias constraint name, as psycopg raises it
    when the durable ``UNIQUE (provider_id, alias)`` is tripped."""

    diag: Any = SimpleNamespace(constraint_name=_ALIAS_UNIQUE_CONSTRAINT)


# -- Stateful fakes ----------------------------------------------------------


class FakeRedis:
    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.store: dict[str, dict[bytes, bytes]] = {}
        self.persisted: list[str] = []
        self.expired: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self._fail = fail or set()

    async def hget(self, key, field):
        if "hget" in self._fail:
            raise RuntimeError("redis down")
        return self.store.get(key, {}).get(field)

    async def delete(self, key):
        if "delete" in self._fail:
            raise RuntimeError("redis down")
        self.deleted.append(key)
        self.store.pop(key, None)

    async def eval(self, script, numkeys, *args):
        """Model the two version-fenced Lua scripts: the set-if-newer cache write
        (installs the blob + version only when absent or strictly newer) and the
        delete tombstone (replaces the record with a version-only marker, no blob,
        when absent or the deleted version is >= the cached one)."""
        if "eval" in self._fail:
            raise RuntimeError("redis down")
        if "DEL" in script:  # version-fenced delete tombstone
            key, ver_field, version, expireat = args
            curver = self.store.get(key, {}).get(ver_field)
            if curver is not None and int(curver) > int(version):
                return 0
            self.store[key] = {ver_field: str(int(version)).encode()}
            self.expired.append((key, int(expireat)))
            return 1
        key, ver_field, blob_field, blob, version, expireat = args
        curver = self.store.get(key, {}).get(ver_field)
        if curver is not None and int(curver) >= int(version):
            return 0
        rec = self.store.setdefault(key, {})
        rec[blob_field] = bytes(blob)
        rec[ver_field] = str(int(version)).encode()
        if expireat == "":
            self.persisted.append(key)
        else:
            self.expired.append((key, int(expireat)))
        return 1


class FakeCursor:
    def __init__(self, pg: FakePg) -> None:
        self._pg = pg
        self.rowcount = 0
        self._result_one = None
        self._result_all: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._pg.executed.append((norm, params))
        if norm.startswith("SELECT encrypted_blob"):
            cid = str(params[0])
            row = self._pg.rows.get(cid)
            # The default read excludes a session-expired row (WHERE ... AND
            # (session_expires_at IS NULL OR session_expires_at > now())); the
            # cleanup read (include_expired=True) drops that clause, so the SQL
            # carries the filter only on the serving path.
            filters_expired = "session_expires_at > now()" in norm
            if row is not None and filters_expired and _expired(row["exp"]):
                row = None
            self._result_one = None if row is None else (row["blob"], row["exp"], row["ver"])
        elif norm.startswith("SELECT connection_id"):
            # list() applies the same session-expiry filter as get().
            self._result_all = [(uuid.UUID(c),) for c in sorted(self._pg.rows) if not _expired(self._pg.rows[c]["exp"])]
        elif norm.startswith("INSERT") and "DO NOTHING" in norm:
            # create_only: params = (cid, provider_id, alias, blob, exp)
            cid, provider_id, alias, blob, exp = str(params[0]), params[1], params[2], params[3], params[4]
            if cid in self._pg.rows:
                self.rowcount = 0
                self._result_one = None  # connection_id conflict
            elif self._pg.alias_taken(provider_id, alias):
                raise _AliasUniqueViolation()
            else:
                self._pg.rows[cid] = {"blob": blob, "exp": exp, "ver": 1, "provider_id": provider_id, "alias": alias}
                self.rowcount = 1
                self._result_one = (1,)
        elif norm.startswith("INSERT") and "DO UPDATE" in norm:
            # upsert: params = (cid, provider_id, alias, blob, exp). The DO UPDATE
            # now sets provider_id/alias too, so a changed alias can trip the
            # durable UNIQUE (provider_id, alias) against a DIFFERENT connection.
            cid, provider_id, alias, blob, exp = str(params[0]), params[1], params[2], params[3], params[4]
            for other_cid, r in self._pg.rows.items():
                if other_cid != cid and r["provider_id"] == provider_id and r["alias"] == alias:
                    raise _AliasUniqueViolation()
            existing = self._pg.rows.get(cid)
            if existing is None:
                self._pg.rows[cid] = {"blob": blob, "exp": exp, "ver": 1, "provider_id": provider_id, "alias": alias}
                ver = 1
            else:
                ver = existing["ver"] + 1
                existing.update(blob=blob, exp=exp, ver=ver, provider_id=provider_id, alias=alias)
            self.rowcount = 1
            self._result_one = (ver,)
        elif norm.startswith("UPDATE"):
            # CAS: params = (new_blob, session_expires_at, conn_uuid, expected_blob)
            cid = str(params[2])
            row = self._pg.rows.get(cid)
            if row is not None and row["blob"] == params[3]:
                row.update(blob=params[0], exp=params[1], ver=row["ver"] + 1)
                self.rowcount = 1
                self._result_one = (row["ver"],)
            else:
                self.rowcount = 0
                self._result_one = None
        elif norm.startswith("DELETE"):
            cid = str(params[0])
            row = self._pg.rows.pop(cid, None)
            if row is None:
                self.rowcount = 0
                self._result_one = None
            else:
                self.rowcount = 1
                self._result_one = (row["ver"],)  # RETURNING cache_version

    async def fetchone(self):
        return self._result_one

    async def fetchall(self):
        return self._result_all


class FakeConn:
    def __init__(self, pg: FakePg) -> None:
        self._pg = pg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self._pg)


class FakePg:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.executed: list[tuple[str, tuple]] = []

    def alias_taken(self, provider_id, alias) -> bool:
        return any(r["provider_id"] == provider_id and r["alias"] == alias for r in self.rows.values())

    def connection(self):
        return FakeConn(self)


def _row(blob: bytes, exp=None, *, ver: int = 1, provider_id: str = "acme", alias: str = "work") -> dict:
    return {"blob": blob, "exp": exp, "ver": ver, "provider_id": provider_id, "alias": alias}


@pytest.fixture
def store_fakes(monkeypatch):
    redis = FakeRedis()
    pg = FakePg()
    classes: list[type] = []

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        classes.append(client_cls)
        if client_cls is RedisClient:
            yield redis
        elif client_cls is PostgresClient:
            yield pg
        else:
            raise AssertionError(f"unexpected client {client_cls!r}")

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    return redis, pg, classes


# -- _expireat_arg -----------------------------------------------------------


def test_expireat_arg_none():
    assert _expireat_arg(None) is None


def test_expireat_arg_aware():
    dt = datetime(2030, 1, 1, tzinfo=UTC)
    assert _expireat_arg(dt) == int(dt.timestamp())


def test_expireat_arg_naive_treated_as_utc():
    naive = datetime(2030, 1, 1)
    assert _expireat_arg(naive) == int(naive.replace(tzinfo=UTC).timestamp())


# -- key helpers -------------------------------------------------------------


def test_rec_key_no_tenant_segment():
    store = RedisPgConnectorTokenStore()
    assert store._rec_key(CID) == f"connectors:rec:{CID}"


def test_as_uuid_invalid_raises():
    with pytest.raises(ValueError, match="not a valid UUID"):
        RedisPgConnectorTokenStore._as_uuid("not-a-uuid")


# -- put / get / delete / list ----------------------------------------------


async def test_put_then_get_cache_hit(store_fakes):
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"cipher", provider_id="acme", alias="work")
    # served from the redis cache (no SELECT executed)
    assert await store.get(CID) == b"cipher"
    assert not any(s.startswith("SELECT encrypted_blob") for s, _ in pg.executed)


async def test_put_requires_provider_id_and_alias_for_insert(store_fakes):
    store = RedisPgConnectorTokenStore()
    with pytest.raises(ValueError, match="provider_id and alias are required"):
        await store.put(CID, b"x", create_only=True)


async def test_get_cache_miss_falls_back_to_pg_and_repopulates(store_fakes):
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    # Durable row exists, cache is cold.
    pg.rows[CID] = _row(b"durable", datetime(2030, 1, 1, tzinfo=UTC), ver=7)
    blob = await store.get(CID)
    assert blob == b"durable"
    # cache repopulated with the durable version
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"durable"
    assert redis.store[f"connectors:rec:{CID}"][_VER_FIELD] == b"7"


async def test_get_pg_miss_returns_none(store_fakes):
    store = RedisPgConnectorTokenStore()
    assert await store.get(CID) is None


async def test_get_cache_error_degrades_to_pg(monkeypatch):
    redis = FakeRedis(fail={"hget"})
    pg = FakePg()
    pg.rows[CID] = _row(b"durable", None, ver=1)

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    assert await store.get(CID) == b"durable"


async def test_put_create_only_conflict_raises(store_fakes):
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"first", create_only=True, provider_id="acme", alias="work")
    with pytest.raises(ConnectorError, match="already exists"):
        await store.put(CID, b"second", create_only=True, provider_id="acme", alias="work2")


async def test_put_create_only_duplicate_alias_raises_alias_in_use(store_fakes):
    """A second connection with the same (provider_id, alias) trips the durable
    UNIQUE constraint, surfaced as AliasInUseError — the alias-uniqueness authority."""
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"first", create_only=True, provider_id="acme", alias="work")
    with pytest.raises(AliasInUseError, match="already in use"):
        await store.put(CID2, b"second", create_only=True, provider_id="acme", alias="work")


async def test_put_upsert_overwrites(store_fakes):
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"first", create_only=True, provider_id="acme", alias="work")
    await store.put(CID, b"second", provider_id="acme", alias="work")
    assert pg.rows[CID]["blob"] == b"second"


async def test_put_upsert_overwrites_provider_id_and_alias(store_fakes):
    """A plain upsert with a changed identity writes the new provider_id/alias to
    the durable columns (they back the uniqueness constraint), not just the blob."""
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"first", create_only=True, provider_id="acme", alias="work")
    await store.put(CID, b"second", provider_id="beta", alias="renamed")
    assert pg.rows[CID]["provider_id"] == "beta"
    assert pg.rows[CID]["alias"] == "renamed"


async def test_put_upsert_colliding_alias_raises_alias_in_use(store_fakes):
    """An upsert that changes a connection's alias to one a DIFFERENT connection
    already holds trips the durable UNIQUE constraint, surfaced as AliasInUseError
    (the same translation the create-only path applies)."""
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"a", create_only=True, provider_id="acme", alias="work")
    await store.put(CID2, b"b", create_only=True, provider_id="acme", alias="other")
    with pytest.raises(AliasInUseError, match="already in use"):
        await store.put(CID2, b"b2", provider_id="acme", alias="work")


async def test_put_cas_match_commits_and_returns_true(store_fakes):
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"old", None, ver=3)
    committed = await store.put(CID, b"new", expected_blob=b"old")
    assert committed is True
    assert pg.rows[CID]["blob"] == b"new"
    assert pg.rows[CID]["ver"] == 4  # version bumped


async def test_put_cas_executes_versioned_conditional_update(store_fakes):
    """The durable CAS is a single UPDATE gated on the exact blob, bumping the
    version, with params in (blob, expiry, connection_id, expected_blob) order."""
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"old", None, ver=1)
    await store.put(CID, b"new", expected_blob=b"old")
    update_sql, params = next((s, p) for s, p in pg.executed if s.startswith("UPDATE"))
    assert "WHERE connection_id = %s AND encrypted_blob = %s" in update_sql
    assert "cache_version = cache_version + 1" in update_sql
    assert "RETURNING cache_version" in update_sql
    assert params == (b"new", None, RedisPgConnectorTokenStore._as_uuid(CID), b"old")


async def test_put_cas_mismatch_returns_false_and_leaves_row(store_fakes):
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"current", None, ver=2)  # a peer already rotated it
    committed = await store.put(CID, b"loser", expected_blob=b"stale")
    assert committed is False
    assert pg.rows[CID]["blob"] == b"current"  # durable record NOT clobbered
    assert pg.rows[CID]["ver"] == 2  # version untouched


async def test_put_cas_missing_row_returns_false(store_fakes):
    store = RedisPgConnectorTokenStore()
    assert await store.put(CID, b"x", expected_blob=b"whatever") is False


async def test_put_cas_and_create_only_mutually_exclusive(store_fakes):
    store = RedisPgConnectorTokenStore()
    with pytest.raises(ValueError, match="mutually exclusive"):
        await store.put(CID, b"x", create_only=True, expected_blob=b"y")


async def test_put_cas_refreshes_cache_when_version_newer(store_fakes):
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"old", None, ver=1)
    redis.store[f"connectors:rec:{CID}"] = {_BLOB_FIELD: b"old", _VER_FIELD: b"1"}
    await store.put(CID, b"new", expected_blob=b"old")  # durable bumps to ver 2
    # cache set-if-newer advanced the field to the committed value
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"new"
    assert redis.store[f"connectors:rec:{CID}"][_VER_FIELD] == b"2"


async def test_put_rejects_non_bytes(store_fakes):
    store = RedisPgConnectorTokenStore()
    with pytest.raises(TypeError, match="blob must be bytes"):
        await store.put(CID, "not-bytes", provider_id="acme", alias="work")  # type: ignore[arg-type]


async def test_put_with_session_expiry_sets_expireat(store_fakes):
    redis, _pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    exp = datetime(2030, 6, 1, tzinfo=UTC)
    await store.put(CID, b"x", session_expires_at=exp, provider_id="acme", alias="work")
    assert redis.expired[-1] == (f"connectors:rec:{CID}", int(exp.timestamp()))


async def test_put_without_expiry_persists_key(store_fakes):
    redis, _pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"x", provider_id="acme", alias="work")
    assert f"connectors:rec:{CID}" in redis.persisted


# -- cache coherence (H6 version fence) --------------------------------------


async def test_stale_read_populate_cannot_poison_a_newer_cache(store_fakes):
    """The poisoning interleaving: a reader loads durable V1, a concurrent CAS
    writer commits V2 and refreshes the cache to V2, THEN the reader's late
    read-populate runs holding its stale V1 snapshot. The version fence makes the
    late populate a no-op, so the cache never regresses below the latest durable
    write (no expired token served past a completed refresh)."""
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()

    # Durable V1 the reader snapshots on its cache-miss SELECT.
    pg.rows[CID] = _row(b"V1", None, ver=1)
    v1_blob, v1_exp, v1_ver = b"V1", None, 1

    # A concurrent CAS writer commits V2 and refreshes the cache to V2.
    committed = await store.put(CID, b"V2", expected_blob=b"V1")
    assert committed is True
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"V2"

    # The reader's LATE read-populate now runs with its stale V1 snapshot.
    await store._cache_set(CID, v1_blob, v1_exp, v1_ver)

    # The version fence rejected the stale write — cache still holds V2.
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"V2"
    assert redis.store[f"connectors:rec:{CID}"][_VER_FIELD] == b"2"


async def test_cache_write_failure_invalidates_stale_entry(monkeypatch):
    """A cache-write failure after a durable commit must not leave a stale entry:
    the store deletes the key so the next read repopulates from Postgres."""
    redis = FakeRedis(fail={"eval"})
    redis.store[f"connectors:rec:{CID}"] = {_BLOB_FIELD: b"stale", _VER_FIELD: b"1"}
    pg = FakePg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"durable", provider_id="acme", alias="work")  # eval fails
    assert pg.rows[CID]["blob"] == b"durable"  # durable commit stands
    # stale cache entry invalidated so the next read repopulates from Postgres
    assert f"connectors:rec:{CID}" not in redis.store
    assert f"connectors:rec:{CID}" in redis.deleted


async def test_cache_write_and_invalidate_both_fail_is_loud(monkeypatch, caplog):
    """If the post-failure invalidation also fails, it is logged LOUDLY (ERROR),
    never silently — the record may serve stale until its TTL."""
    import logging

    redis = FakeRedis(fail={"eval", "delete"})
    pg = FakePg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    with caplog.at_level(logging.ERROR):
        await store.put(CID, b"durable", provider_id="acme", alias="work")  # no raise
    assert pg.rows[CID]["blob"] == b"durable"
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


async def test_cache_write_failure_does_not_mask_commit(monkeypatch):
    redis = FakeRedis(fail={"eval"})
    pg = FakePg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"durable", provider_id="acme", alias="work")  # no raise
    assert pg.rows[CID]["blob"] == b"durable"


async def test_delete_drops_durable_and_cache(store_fakes):
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    await store.put(CID, b"x", provider_id="acme", alias="work")
    await store.delete(CID)
    assert CID not in pg.rows
    # A tombstone (version-only marker, no blob) replaces the entry so a racing
    # read cannot resurrect it — the served blob is gone.
    assert _BLOB_FIELD not in redis.store.get(f"connectors:rec:{CID}", {})


async def test_delete_tombstone_blocks_stale_read_populate(store_fakes):
    """The resurrection interleaving: a reader loads durable V5 on its cache-miss
    SELECT, then a concurrent delete commits and writes the tombstone, THEN the
    reader's late read-populate runs holding its stale V5 snapshot. The tombstone
    fences the populate, so the deleted token never re-enters the cache."""
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"live", None, ver=5)
    snap_blob, snap_exp, snap_ver = b"live", None, 5

    await store.delete(CID)
    assert CID not in pg.rows

    # The racing reader's LATE read-populate runs with its stale pre-delete snapshot.
    await store._cache_set(CID, snap_blob, snap_exp, snap_ver)

    key = f"connectors:rec:{CID}"
    assert _BLOB_FIELD not in redis.store.get(key, {})  # blocked — no blob resurrected
    assert await store.get(CID) is None  # miss → PG empty → None


async def test_delete_tombstone_overwrites_already_resurrected_blob(store_fakes):
    """The other ordering: the reader's read-populate lands FIRST (blob back in the
    cache at V5), then the delete's tombstone runs and must clear it."""
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"live", None, ver=5)

    await store._cache_set(CID, b"live", None, 5)
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"live"

    await store.delete(CID)

    key = f"connectors:rec:{CID}"
    assert _BLOB_FIELD not in redis.store.get(key, {})  # tombstone cleared the blob
    assert await store.get(CID) is None


async def test_delete_missing_row_drops_cache_key(store_fakes):
    """A delete that finds no durable row (already gone) has no version to fence,
    so it just drops any cache key best-effort."""
    redis, _pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    redis.store[f"connectors:rec:{CID}"] = {_BLOB_FIELD: b"stray", _VER_FIELD: b"1"}
    await store.delete(CID)
    assert f"connectors:rec:{CID}" in redis.deleted
    assert f"connectors:rec:{CID}" not in redis.store


async def test_delete_cache_error_does_not_mask_commit(monkeypatch):
    redis = FakeRedis(fail={"eval"})
    pg = FakePg()
    pg.rows[CID] = _row(b"x", None, ver=1)

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    await store.delete(CID)  # no raise
    assert CID not in pg.rows


async def test_delete_tombstone_and_invalidate_both_fail_is_loud(monkeypatch, caplog):
    """If the delete tombstone write AND its fallback key drop both fail, it is
    logged LOUDLY (ERROR) — a revoked token may linger in cache until its TTL."""
    import logging

    redis = FakeRedis(fail={"eval", "delete"})
    pg = FakePg()
    pg.rows[CID] = _row(b"x", None, ver=1)

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis if client_cls is RedisClient else pg

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    store = RedisPgConnectorTokenStore()
    with caplog.at_level(logging.ERROR):
        await store.delete(CID)  # no raise
    assert CID not in pg.rows  # durable delete stands
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


# -- session-expiry read enforcement (I) -------------------------------------


async def test_get_excludes_session_expired_row(store_fakes):
    """A cache-cold record whose session_expires_at is in the past is no longer
    served from Postgres — the durable read filters it out."""
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"dead", datetime(2000, 1, 1, tzinfo=UTC), ver=1)  # long expired
    assert await store.get(CID) is None


async def test_get_serves_unexpired_row(store_fakes):
    """A record whose session_expires_at is still in the future is served."""
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"live", datetime(2999, 1, 1, tzinfo=UTC), ver=1)
    assert await store.get(CID) == b"live"


async def test_get_include_expired_returns_expired_row_without_repopulating_cache(store_fakes):
    """The cleanup read (include_expired=True) returns a session-expired row so
    disconnect can still purge it — but it must NOT repopulate the hot cache, whose
    EXPIREAT would be in the past (an expired record may never be served)."""
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"dead", datetime(2000, 1, 1, tzinfo=UTC), ver=1)  # long expired

    # Default serving read still hides it; cleanup read returns it.
    assert await store.get(CID) is None
    assert await store.get(CID, include_expired=True) == b"dead"
    # No cache entry was written for the expired record.
    assert f"connectors:rec:{CID}" not in redis.store


async def test_get_include_expired_serves_and_repopulates_a_live_row(store_fakes):
    """include_expired is a superset: a still-live row is returned AND repopulated
    into the cache exactly as the default read would."""
    redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"live", datetime(2999, 1, 1, tzinfo=UTC), ver=4)
    assert await store.get(CID, include_expired=True) == b"live"
    assert redis.store[f"connectors:rec:{CID}"][_BLOB_FIELD] == b"live"
    assert redis.store[f"connectors:rec:{CID}"][_VER_FIELD] == b"4"


async def test_list_excludes_session_expired_rows(store_fakes):
    """list() applies the same dead-session bound as get(): a session-expired row
    is dropped from the listing while live and no-expiry rows remain."""
    _redis, pg, _ = store_fakes
    store = RedisPgConnectorTokenStore()
    pg.rows[CID] = _row(b"dead", datetime(2000, 1, 1, tzinfo=UTC), ver=1, alias="dead")
    pg.rows[CID2] = _row(b"live", None, ver=1, alias="live")
    assert await store.list() == [CID2]


async def test_list_returns_sorted_ids(store_fakes):
    store = RedisPgConnectorTokenStore()
    await store.put(CID2, b"b", create_only=True, provider_id="acme", alias="second")
    await store.put(CID, b"a", create_only=True, provider_id="acme", alias="first")
    assert await store.list() == sorted([CID, CID2])


async def test_list_empty(store_fakes):
    store = RedisPgConnectorTokenStore()
    assert await store.list() == []


def test_token_store_builds_concrete_store():
    """The engine's store accessor returns the concrete redis-pg store."""
    from tai42_skeleton.connectors.store import token_store

    assert isinstance(token_store(), RedisPgConnectorTokenStore)
