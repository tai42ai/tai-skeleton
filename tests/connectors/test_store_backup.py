"""Store-level connector backup: catalog + connections SQL round-trips.

Postgres is faked at the kit ``client_ctx`` seam (a stateful in-memory model of
the four connector tables); no real database is touched. The catalog cache
reload (``refresh_catalog``) is stubbed so these tests pin the SQL round-trip,
not the separately-tested cache machinery.
"""

from __future__ import annotations

import base64
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.exceptions import InvalidTag
from psycopg.errors import UniqueViolation
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient
from tai_kit.settings import reset_all_settings

import tai_skeleton.connectors.store.backup as store_backup
from tai_skeleton.connectors.oauth import crypto
from tai_skeleton.connectors.store.backup import (
    export_connector_catalog,
    export_connector_connections,
    import_connector_catalog,
    import_connector_connections,
)
from tai_skeleton.connectors.store.redis_pg import _ALIAS_UNIQUE_CONSTRAINT, RedisPgConnectorTokenStore

from .conftest import CID, CID2, make_noauth_http_descriptor, make_oauth_descriptor

# Default creation time stamped on a fake row that was seeded without an explicit
# one, so the created_at-bearing export SELECTs always have a value to serialize.
_DEFAULT_CREATED_AT = datetime(2020, 1, 1, tzinfo=UTC)


class _AliasUniqueViolation(UniqueViolation):
    """A UniqueViolation carrying the alias constraint name, as psycopg raises it
    when the durable ``UNIQUE (provider_id, alias)`` is tripped."""

    diag: Any = SimpleNamespace(constraint_name=_ALIAS_UNIQUE_CONSTRAINT)


class _OtherUniqueViolation(UniqueViolation):
    """A UniqueViolation for a constraint OTHER than the alias uniqueness — the
    importer must re-raise this loudly rather than swallow it per-row."""

    diag: Any = SimpleNamespace(constraint_name="connector_connections_pkey")


# -- Stateful fake Postgres modelling the four connector tables --------------


class _FakeTxn:
    """A savepoint stand-in: propagates any exception (like psycopg's transaction
    context, which rolls back to the savepoint and re-raises)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeCursor:
    def __init__(self, pg: _FakePg) -> None:
        self._pg = pg
        self._result_all: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._pg.executed.append(norm)
        self._result_all = []
        pg = self._pg
        if norm.startswith("SELECT id, display_name, sort_order"):
            self._result_all = [
                (c["id"], c["display_name"], c["sort_order"], c.get("created_at", _DEFAULT_CREATED_AT))
                for c in sorted(pg.categories.values(), key=lambda c: (c["sort_order"], c["id"]))
            ]
        elif norm.startswith("SELECT provider_id, descriptor"):
            self._result_all = [
                (
                    p["provider_id"],
                    p["descriptor"],
                    p["origin"],
                    p["category"],
                    p["source_url"],
                    p["added_by"],
                    p["enabled"],
                    p.get("created_at", _DEFAULT_CREATED_AT),
                )
                for p in sorted(pg.catalog.values(), key=lambda p: p["provider_id"])
            ]
        elif norm.startswith("SELECT id, url, enabled"):
            self._result_all = [
                (s["id"], s["url"], s["enabled"], s.get("created_at", _DEFAULT_CREATED_AT))
                for s in sorted(pg.sources.values(), key=lambda s: s["id"])
            ]
        elif norm == "SELECT id FROM connector_category":
            self._result_all = [(k,) for k in pg.categories]
        elif norm == "SELECT provider_id FROM connector_catalog":
            self._result_all = [(k,) for k in pg.catalog]
        elif norm == "SELECT id FROM connector_allowed_source":
            self._result_all = [(k,) for k in pg.sources]
        elif norm.startswith("INSERT INTO connector_category"):
            cid, display_name, sort_order, created_at = params
            pg.categories[cid] = {
                "id": cid,
                "display_name": display_name,
                "sort_order": sort_order,
                "created_at": created_at,
            }
        elif norm.startswith("INSERT INTO connector_catalog"):
            pid, descriptor, origin, category, source_url, added_by, enabled, created_at = params
            pg.catalog[pid] = {
                "provider_id": pid,
                "descriptor": descriptor.obj,
                "origin": origin,
                "category": category,
                "source_url": source_url,
                "added_by": added_by,
                "enabled": enabled,
                "created_at": created_at,
            }
        elif norm.startswith("INSERT INTO connector_allowed_source"):
            sid, url, enabled, created_at = params
            pg.sources[sid] = {"id": sid, "url": url, "enabled": enabled, "created_at": created_at}
        elif norm.startswith("SELECT connection_id, provider_id, alias, encrypted_blob"):
            self._result_all = [
                (uuid.UUID(k), r["provider_id"], r["alias"], r["blob"], r["exp"])
                for k, r in sorted(pg.connections.items())
            ]
        elif norm == "SELECT connection_id FROM connector_connections":
            self._result_all = [(uuid.UUID(k),) for k in pg.connections]
        elif norm.startswith("INSERT INTO connector_connections"):
            if pg.raise_on_conn_insert is not None:
                raise pg.raise_on_conn_insert
            conn_uuid, provider_id, alias, blob, exp = params
            cid = str(conn_uuid)
            for other_cid, r in pg.connections.items():
                if other_cid != cid and r["provider_id"] == provider_id and r["alias"] == alias:
                    raise _AliasUniqueViolation()
            pg.connections[cid] = {"provider_id": provider_id, "alias": alias, "blob": bytes(blob), "exp": exp}
        else:
            raise AssertionError(f"unhandled SQL in fake: {norm!r}")

    async def fetchall(self):
        return self._result_all


class _FakeConn:
    def __init__(self, pg: _FakePg) -> None:
        self._pg = pg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self._pg)

    def transaction(self):
        return _FakeTxn()


class _FakeRedis:
    """Minimal stand-in for the pooled Redis client the cache-invalidation path
    reaches through ``client_ctx(RedisClient, ...)``.

    ``warm`` models keys currently present in the cache; ``delete`` records every
    key it was asked to drop (so a test can assert the invalidation happened) and
    discards it from ``warm``, mirroring redis ``DEL`` returning the drop count."""

    def __init__(self) -> None:
        self.warm: set[str] = set()
        self.deleted: list[str] = []

    async def delete(self, *keys: str) -> int:
        dropped = 0
        for key in keys:
            self.deleted.append(key)
            if key in self.warm:
                self.warm.discard(key)
                dropped += 1
        return dropped


class _FakePg:
    def __init__(self) -> None:
        self.categories: dict[str, dict] = {}
        self.catalog: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self.connections: dict[str, dict] = {}
        self.executed: list[str] = []
        self.redis = _FakeRedis()
        # When set to an exception, the next connector_connections INSERT raises
        # it — used to drive the non-alias UniqueViolation re-raise path.
        self.raise_on_conn_insert: Any = None

    def connection(self):
        return _FakeConn(self)


@pytest.fixture
def pg(monkeypatch):
    fake = _FakePg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is PostgresClient:
            yield fake
        elif client_cls is RedisClient:
            yield fake.redis
        else:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")

    monkeypatch.setattr(store_backup, "client_ctx", fake_client_ctx)

    calls: list[int] = []

    async def fake_refresh_catalog():
        calls.append(1)

    monkeypatch.setattr(store_backup, "refresh_catalog", fake_refresh_catalog)
    fake.refresh_calls = calls  # type: ignore[attr-defined]
    return fake


def _wipe(pg: _FakePg) -> None:
    """Empty every table in place (simulate a fresh target) while keeping the
    fixture's ``client_ctx`` binding pointed at the same fake."""
    pg.categories.clear()
    pg.catalog.clear()
    pg.sources.clear()
    pg.connections.clear()


def _descriptor_json(provider_id: str = "httpsvc") -> dict:
    return make_noauth_http_descriptor(provider_id=provider_id).model_dump(mode="json", exclude={"origin", "category"})


# -- connector_catalog -------------------------------------------------------


async def test_catalog_round_trip_includes_disabled_row(pg):
    # Seed categories, an ENABLED and a DISABLED catalog row, and two sources.
    pg.categories["data"] = {"id": "data", "display_name": "Data", "sort_order": 4}
    pg.catalog["live"] = {
        "provider_id": "live",
        "descriptor": _descriptor_json("live"),
        "origin": "system",
        "category": "data",
        "source_url": None,
        "added_by": None,
        "enabled": True,
    }
    pg.catalog["hidden"] = {
        "provider_id": "hidden",
        "descriptor": _descriptor_json("hidden"),
        "origin": "community",
        "category": "data",
        "source_url": "https://example.test/hidden",
        "added_by": "someone",
        "enabled": False,  # disabled — must still round-trip
    }
    pg.sources["github"] = {"id": "github", "url": "https://github.com", "enabled": True}
    pg.sources["off"] = {"id": "off", "url": "https://off.test", "enabled": False}

    payload = await export_connector_catalog()

    # The disabled row is present in the export (fetch_catalog would have dropped it).
    provider_ids = {p["provider_id"] for p in payload["providers"]}
    assert provider_ids == {"live", "hidden"}
    hidden = next(p for p in payload["providers"] if p["provider_id"] == "hidden")
    assert hidden["enabled"] is False
    assert hidden["descriptor"] == _descriptor_json("hidden")  # JSONB carried verbatim

    # Wipe every table and restore from the payload.
    _wipe(pg)
    report = await import_connector_catalog(payload)
    assert report == {"created": 2 + 2 + 1, "updated": 0, "skipped": 0, "errors": []}  # 1 cat + 2 prov + 2 src
    assert pg.refresh_calls == [1]  # cache reloaded so rows go live in-process

    # The restored tables equal the source, disabled row and all.
    assert pg.catalog["hidden"]["enabled"] is False
    assert pg.catalog["hidden"]["descriptor"] == _descriptor_json("hidden")
    assert pg.categories["data"]["display_name"] == "Data"
    assert pg.sources["off"]["enabled"] is False


async def test_catalog_reimport_is_idempotent_updates(pg):
    pg.categories["data"] = {"id": "data", "display_name": "Data", "sort_order": 4}
    pg.catalog["live"] = {
        "provider_id": "live",
        "descriptor": _descriptor_json("live"),
        "origin": "system",
        "category": "data",
        "source_url": None,
        "added_by": None,
        "enabled": True,
    }
    payload = await export_connector_catalog()

    _wipe(pg)
    first = await import_connector_catalog(payload)
    assert first == {"created": 2, "updated": 0, "skipped": 0, "errors": []}

    # Re-import over the now-populated tables: every row is an update, none created.
    second = await import_connector_catalog(payload)
    assert second == {"created": 0, "updated": 2, "skipped": 0, "errors": []}


# -- connector_catalog restore validation ------------------------------------


def _cat_entry(*, id: str = "data", display_name: str = "Data", sort_order: int = 1) -> dict:
    return {
        "id": id,
        "display_name": display_name,
        "sort_order": sort_order,
        "created_at": _DEFAULT_CREATED_AT.isoformat(),
    }


def _prov_entry(
    *,
    provider_id: str = "httpsvc",
    descriptor: dict | None = None,
    origin: str = "system",
    category: str = "data",
    source_url: str | None = None,
    added_by: str | None = None,
    enabled: bool = True,
) -> dict:
    return {
        "provider_id": provider_id,
        "descriptor": _descriptor_json(provider_id) if descriptor is None else descriptor,
        "origin": origin,
        "category": category,
        "source_url": source_url,
        "added_by": added_by,
        "enabled": enabled,
        "created_at": _DEFAULT_CREATED_AT.isoformat(),
    }


def _catalog_payload(providers: list[dict], categories: list[dict] | None = None) -> dict:
    return {
        "categories": categories if categories is not None else [_cat_entry()],
        "providers": providers,
        "sources": [],
    }


async def test_catalog_restore_accepts_valid_enabled_set(pg):
    """A backup whose every enabled row parses cleanly restores and publishes."""
    payload = _catalog_payload([_prov_entry(provider_id="httpsvc")])
    report = await import_connector_catalog(payload)
    assert report == {"created": 2, "updated": 0, "skipped": 0, "errors": []}  # 1 cat + 1 prov
    assert pg.catalog["httpsvc"]["enabled"] is True
    assert pg.refresh_calls == [1]  # publish step ran


@pytest.mark.parametrize(
    ("provider", "match"),
    [
        (
            _prov_entry(descriptor={**_descriptor_json("httpsvc"), "origin": "system"}),
            "must not embed",
        ),
        (_prov_entry(category="ghost"), "unknown category"),
        (_prov_entry(origin="community", added_by=None), "community-origin but has no added_by"),
        (_prov_entry(descriptor={"garbage": True}), "invalid descriptor"),
        (
            _prov_entry(descriptor=make_oauth_descriptor().model_dump(mode="json", exclude={"origin", "category"})),
            "must have kind='none'",
        ),
        (_prov_entry(provider_id="mismatch", descriptor=_descriptor_json("httpsvc")), "does not match descriptor id"),
    ],
)
async def test_catalog_restore_rejects_invalid_enabled_row_before_any_write(pg, provider, match):
    """A malformed ENABLED row aborts the restore with NOTHING committed — the
    validation runs inside the transaction before the first INSERT, so a poison
    backup can never land a row that then fails every worker at startup."""
    # The oauth-kind case declares its own valid category so validation reaches
    # the kind check rather than tripping the category gate first.
    categories = [_cat_entry(), _cat_entry(id="productivity", display_name="Productivity", sort_order=2)]
    payload = _catalog_payload([provider], categories=categories)
    with pytest.raises(ValueError, match=match):
        await import_connector_catalog(payload)
    # Nothing was written and the cache publish never ran.
    assert not any(s.startswith("INSERT") for s in pg.executed)
    assert pg.catalog == {}
    assert pg.categories == {}
    assert pg.refresh_calls == []


async def test_catalog_restore_skips_validation_for_disabled_row(pg):
    """A malformed DISABLED row is carried verbatim, mirroring fetch_catalog's
    WHERE enabled — only enabled rows are gated."""
    bad_disabled = _prov_entry(
        provider_id="hidden",
        descriptor={"garbage": True},  # would fail validation if enabled
        enabled=False,
    )
    payload = _catalog_payload([bad_disabled])
    report = await import_connector_catalog(payload)
    assert report == {"created": 2, "updated": 0, "skipped": 0, "errors": []}
    assert pg.catalog["hidden"]["enabled"] is False


# -- connector_connections ---------------------------------------------------


def _seed_encrypted(pg: _FakePg, *, connection_id: str, provider_id: str, alias: str, plaintext: bytes) -> bytes:
    blob = crypto.encrypt(plaintext, connection_id=connection_id)
    pg.connections[connection_id] = {"provider_id": provider_id, "alias": alias, "blob": blob, "exp": None}
    return blob


async def test_connections_round_trip_blob_identical(pg):
    exp = datetime(2030, 6, 1, tzinfo=UTC)
    blob = crypto.encrypt(b"secret-token", connection_id=CID)
    pg.connections[CID] = {"provider_id": "acme", "alias": "work", "blob": blob, "exp": exp}

    payload = await export_connector_connections()
    assert len(payload) == 1
    entry = payload[0]
    assert entry["connection_id"] == CID
    assert entry["provider_id"] == "acme"
    assert entry["session_expires_at"] == exp.isoformat()
    # Ciphertext is base64 of the exact stored bytes — never decrypted.
    assert base64.b64decode(entry["encrypted_blob_b64"]) == blob

    # Wipe and restore.
    _wipe(pg)
    report = await import_connector_connections(payload)
    assert report == {"created": 1, "updated": 0, "skipped": 0, "errors": []}

    restored = pg.connections[CID]
    assert restored["blob"] == blob  # byte-for-byte identical
    assert restored["provider_id"] == "acme"
    assert restored["exp"] == exp


async def test_restored_blob_undecryptable_under_wrong_kek_raises(pg, monkeypatch):
    """A restore under a DIFFERENT CONNECTORS_KEK yields intact-but-dead
    ciphertext: decrypt fails LOUDLY (InvalidTag), never a silent skip."""
    blob = crypto.encrypt(b"secret-token", connection_id=CID)
    pg.connections[CID] = {"provider_id": "acme", "alias": "work", "blob": blob, "exp": None}

    payload = await export_connector_connections()
    _wipe(pg)
    await import_connector_connections(payload)
    restored = pg.connections[CID]["blob"]

    # Same KEK still decrypts the restored ciphertext.
    assert crypto.decrypt(restored, connection_id=CID) == b"secret-token"

    # Swap in a different 32-byte KEK; the restored ciphertext no longer decrypts.
    other_kek = base64.b64encode(bytes(range(100, 132))).decode("ascii")
    monkeypatch.setenv("CONNECTORS_KEK", other_kek)
    reset_all_settings()
    with pytest.raises(InvalidTag):
        crypto.decrypt(restored, connection_id=CID)


async def test_connections_alias_collision_on_different_id_is_reported(pg):
    """Importing a connection whose (provider_id, alias) is already held by a
    DIFFERENT connection_id trips the durable UNIQUE constraint, reported as a
    per-row error — never silent — while other rows still restore."""
    # Target already holds the alias under CID.
    _seed_encrypted(pg, connection_id=CID, provider_id="acme", alias="work", plaintext=b"a")

    # The backup carries a colliding row under CID2 plus a clean row (CID3).
    cid3 = "33333333-3333-4333-8333-333333333333"
    payload = [
        {
            "connection_id": CID2,
            "provider_id": "acme",
            "alias": "work",  # collides with CID's alias
            "session_expires_at": None,
            "encrypted_blob_b64": base64.b64encode(b"b").decode("ascii"),
        },
        {
            "connection_id": cid3,
            "provider_id": "acme",
            "alias": "home",  # clean
            "session_expires_at": None,
            "encrypted_blob_b64": base64.b64encode(b"c").decode("ascii"),
        },
    ]

    report = await import_connector_connections(payload)
    assert report["created"] == 1  # the clean row restored
    assert report["skipped"] == 1  # the colliding row rejected
    assert len(report["errors"]) == 1
    assert "already in use" in report["errors"][0]
    assert CID2 in report["errors"][0]
    # The clean row landed; the colliding one did not.
    assert cid3 in pg.connections
    assert CID2 not in pg.connections


async def test_connections_reimport_counts_updates(pg):
    blob = crypto.encrypt(b"tok", connection_id=CID)
    pg.connections[CID] = {"provider_id": "acme", "alias": "work", "blob": blob, "exp": None}
    payload = await export_connector_connections()

    # Re-import over the existing row: it's an update, not a create.
    report = await import_connector_connections(payload)
    assert report == {"created": 0, "updated": 1, "skipped": 0, "errors": []}


async def test_import_invalidates_warm_cache(pg):
    """A restore into a running deployment must DROP each restored connection's
    Redis cache key so the next ``get`` repopulates the restored token from
    Postgres — ``get`` serves a cached blob on a HIT with no read-side version
    check, so a warm entry would otherwise keep serving the stale pre-import token."""
    blob = crypto.encrypt(b"fresh-token", connection_id=CID)
    pg.connections[CID] = {"provider_id": "acme", "alias": "work", "blob": blob, "exp": None}
    payload = await export_connector_connections()

    rec_key = RedisPgConnectorTokenStore()._rec_key(CID)
    pg.redis.warm.add(rec_key)  # the connection is warm in the cache before the restore

    await import_connector_connections(payload)

    assert rec_key in pg.redis.deleted  # the cache key was dropped
    assert rec_key not in pg.redis.warm  # so the next get repopulates from Postgres


async def test_import_invalidates_cache_on_canonical_id(pg):
    """A backup carrying a non-canonical connection_id (uppercase) still drops the
    CANONICAL cache key get() reads — invalidation keys on str(UUID), not the raw
    backup string."""
    payload = [
        {
            "connection_id": CID.upper(),
            "provider_id": "acme",
            "alias": "work",
            "session_expires_at": None,
            "encrypted_blob_b64": base64.b64encode(b"z").decode("ascii"),
        }
    ]
    canonical_key = RedisPgConnectorTokenStore()._rec_key(CID)  # CID is already canonical
    pg.redis.warm.add(canonical_key)

    await import_connector_connections(payload)

    assert canonical_key in pg.redis.deleted


async def test_connections_non_alias_unique_violation_raises(pg):
    """A UniqueViolation whose constraint is NOT the alias uniqueness must
    RE-RAISE loudly, never be swallowed into per-row errors (which would hide an
    unrelated durable-constraint failure as a benign skip)."""
    payload = [
        {
            "connection_id": CID,
            "provider_id": "acme",
            "alias": "work",
            "session_expires_at": None,
            "encrypted_blob_b64": base64.b64encode(b"x").decode("ascii"),
        }
    ]
    pg.raise_on_conn_insert = _OtherUniqueViolation()
    with pytest.raises(UniqueViolation):
        await import_connector_connections(payload)


async def test_catalog_created_at_preserved_on_restore(pg):
    """The original ``created_at`` survives a restore into a fresh table for every
    catalog table — it is not reset to the restore time."""
    seeded = datetime(2021, 3, 4, 5, 6, 7, tzinfo=UTC)
    pg.categories["data"] = {"id": "data", "display_name": "Data", "sort_order": 1, "created_at": seeded}
    pg.catalog["p"] = {
        "provider_id": "p",
        "descriptor": _descriptor_json("p"),
        "origin": "system",
        "category": "data",
        "source_url": None,
        "added_by": None,
        "enabled": True,
        "created_at": seeded,
    }
    pg.sources["gh"] = {"id": "gh", "url": "https://github.com", "enabled": True, "created_at": seeded}
    payload = await export_connector_catalog()

    _wipe(pg)
    await import_connector_catalog(payload)

    assert pg.categories["data"]["created_at"] == seeded
    assert pg.catalog["p"]["created_at"] == seeded
    assert pg.sources["gh"]["created_at"] == seeded
