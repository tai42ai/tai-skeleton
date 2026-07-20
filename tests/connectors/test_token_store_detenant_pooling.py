"""Connector token store: de-tenant + pooled-client conformance.

Covers the two architectural standards applied on copy-in:

* De-tenant — the store keys every record by ``connection_id`` alone: the Redis
  key is ``rec:{cid}`` (no tenant segment) and every SQL statement references
  ``connection_id`` and never ``client_name``; the contract ABC exposes ``list``
  (not ``list_for_tenant``) and tenant-free method signatures.
* Pooling — Redis and Postgres are reached only through the kit pooled clients
  (``RedisClient`` / ``PostgresClient`` via ``client_ctx``); the connector source
  tree opens no raw ``psycopg_pool`` / ``httpx`` / ``fastmcp`` client.

The round-trip runs the real store logic against in-memory fakes injected in
place of the pooled clients.
"""

from __future__ import annotations

import inspect
import pathlib
from contextlib import asynccontextmanager

import pytest
from tai_contract.connectors.store import ConnectorTokenStore
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient

import tai_skeleton.connectors.store.redis_pg as redis_pg
from tai_skeleton.connectors.store.redis_pg import RedisPgConnectorTokenStore

_CID = "11111111-1111-4111-8111-111111111111"
_CONNECTORS_DIR = pathlib.Path(redis_pg.__file__).resolve().parents[1]


# -- Fakes -------------------------------------------------------------------


class _FakeRedis:
    """Minimal hash-backed redis for the cache hot path."""

    def __init__(self) -> None:
        self.store: dict[str, dict[bytes, bytes]] = {}
        self.expireat_calls: list[tuple[str, int]] = []

    async def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    async def eval(self, script, numkeys, *args):
        # Model both version-fenced Lua scripts: the set-if-newer cache write and
        # the delete tombstone (a version-only marker, no blob).
        if "DEL" in script:
            key, ver_field, version, expireat = args
            curver = self.store.get(key, {}).get(ver_field)
            if curver is not None and int(curver) > int(version):
                return 0
            self.store[key] = {ver_field: str(int(version)).encode()}
            self.expireat_calls.append((key, int(expireat)))
            return 1
        key, ver_field, blob_field, blob, version, expireat = args
        rec = self.store.get(key, {})
        curver = rec.get(ver_field)
        if curver is None or int(curver) < int(version):
            rec = self.store.setdefault(key, {})
            rec[blob_field] = bytes(blob)
            rec[ver_field] = str(int(version)).encode()
            if expireat != "":
                self.expireat_calls.append((key, int(expireat)))
        return 1

    async def delete(self, key):
        self.store.pop(key, None)


class _FakeCursor:
    def __init__(self, recorder: _FakePg) -> None:
        self._rec = recorder
        self.rowcount = 1
        self._last_select_row = None
        self._last_select_rows: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        self._rec.executed.append((" ".join(sql.split()), params))
        if sql.lstrip().upper().startswith("SELECT"):
            # The fakes serve the cache hot path; a Postgres SELECT is only hit
            # on a cache miss, which the round-trip drives only after a delete —
            # so the durable source is empty.
            self._last_select_row = None
            self._last_select_rows = []
        elif "RETURNING cache_version" in sql:
            # A write returns the new monotonic cache_version the store fences on.
            self._last_select_row = (1,)

    async def fetchone(self):
        return self._last_select_row

    async def fetchall(self):
        return self._last_select_rows


class _FakeConn:
    def __init__(self, recorder: _FakePg) -> None:
        self._rec = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self._rec)


class _FakePg:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def connection(self):
        return _FakeConn(self)


@pytest.fixture
def fakes(monkeypatch):
    """Inject in-memory fakes for the pooled Redis + Postgres clients and record
    which pooled-client class each ``client_ctx`` call asked for."""
    redis = _FakeRedis()
    pg = _FakePg()
    client_classes: list[type] = []

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        client_classes.append(client_cls)
        if client_cls is RedisClient:
            yield redis
        elif client_cls is PostgresClient:
            yield pg
        else:  # pragma: no cover - guards an unexpected client class
            raise AssertionError(f"unexpected pooled client: {client_cls!r}")

    monkeypatch.setattr(redis_pg, "client_ctx", fake_client_ctx)
    return redis, pg, client_classes


# -- De-tenant ---------------------------------------------------------------


def test_contract_abc_is_detenanted():
    assert issubclass(RedisPgConnectorTokenStore, ConnectorTokenStore)
    # tenant-scoped listing is gone; the de-tenanted ABC exposes ``list``.
    assert hasattr(RedisPgConnectorTokenStore, "list")
    assert not hasattr(RedisPgConnectorTokenStore, "list_for_tenant")
    # ``get`` keys on connection_id alone (self + one positional); its only other
    # parameter is the keyword-only ``include_expired`` cleanup flag — never a
    # tenant discriminator.
    sig = inspect.signature(RedisPgConnectorTokenStore.get)
    positional = [n for n, p in sig.parameters.items() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert positional == ["self", "connection_id"]
    assert sig.parameters["include_expired"].kind is inspect.Parameter.KEYWORD_ONLY


def test_redis_key_has_no_tenant_segment():
    store = RedisPgConnectorTokenStore()
    assert store._rec_key(_CID) == f"connectors:rec:{_CID}"
    # exactly one ':cid' tail after the 'rec' marker — no tenant in between.
    assert store._rec_key(_CID).count(":") == 2


async def test_put_sql_keys_by_connection_id_only(fakes):
    _redis, pg, _ = fakes
    store = RedisPgConnectorTokenStore()

    await store.put(_CID, b"cipher", create_only=True, provider_id="acme", alias="work")

    insert_sql, params = pg.executed[-1]
    assert "connection_id" in insert_sql
    assert "client_name" not in insert_sql
    assert "ON CONFLICT (connection_id)" in insert_sql
    # (connection_id, provider_id, alias, blob, session_expires_at) — no tenant.
    assert len(params) == 5


# -- Round-trip --------------------------------------------------------------


async def test_round_trip_by_connection_id(fakes):
    redis, pg, _ = fakes
    store = RedisPgConnectorTokenStore()
    blob = b"\x00\x01encrypted-blob"

    # save
    await store.put(_CID, blob, create_only=True, provider_id="acme", alias="work")
    assert f"connectors:rec:{_CID}" in redis.store

    # load (Redis hot path)
    loaded = await store.get(_CID)
    assert loaded == blob

    # delete — drops the durable row and replaces the cache entry with a
    # version-only tombstone (no blob) so a racing read cannot resurrect it
    await store.delete(_CID)
    assert b"blob" not in redis.store.get(f"connectors:rec:{_CID}", {})
    delete_sql, delete_params = pg.executed[-1]
    assert delete_sql.startswith("DELETE FROM connector_connections")
    assert "client_name" not in delete_sql
    assert len(delete_params) == 1

    # load after delete — cache miss falls through to the (empty) durable store
    assert await store.get(_CID) is None


# -- Pooling -----------------------------------------------------------------


async def test_store_uses_kit_pooled_clients(fakes):
    _redis, _pg, client_classes = fakes
    store = RedisPgConnectorTokenStore()

    await store.put(_CID, b"cipher", create_only=True, provider_id="acme", alias="work")
    await store.get(_CID)
    await store.list()

    # Every backend touch went through a kit pooled client class.
    assert PostgresClient in client_classes
    assert RedisClient in client_classes
    assert set(client_classes) <= {PostgresClient, RedisClient}


def test_connector_tree_opens_no_raw_clients():
    """No module in the connectors engine constructs a raw psycopg pool, httpx
    client, or fastmcp client — every connection goes through a pooled client."""
    forbidden = (
        "psycopg_pool",
        "AsyncConnectionPool",
        "httpx.AsyncClient(",
        "from fastmcp import Client",
    )
    offenders: list[str] = []
    for path in _CONNECTORS_DIR.rglob("*.py"):
        text = path.read_text()
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, offenders
