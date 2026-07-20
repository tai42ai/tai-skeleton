"""The attribution store over an in-memory fake Postgres.

A ``FakeMarketplacePg`` models the single ``marketplace_installs`` table and
interprets the store's exact SQL by normalized prefix, monkeypatched in over the
pooled ``client_ctx``. It covers the upsert-on-conflict, get hit/miss, delete
True/False, list ordering, and the ``spec`` / ``repository_url`` / ``tag`` /
``artifact_ref`` / ``sha256`` round-trips (github carries the pin columns, pypi
keeps them NULL).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tai_kit.clients.impl.postgres import Json, PostgresClient

from tai_skeleton.marketplace import store as store_module
from tai_skeleton.marketplace.store import MarketplaceInstallStore


def _unwrap(value: Any) -> Any:
    return value.obj if isinstance(value, Json) else value


class _FakeCursor:
    def __init__(self, pg: FakeMarketplacePg) -> None:
        self._pg = pg
        self.rowcount = 0
        self._one: Any = None
        self._all: list = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, sql: str, params: tuple = ()) -> None:
        norm = " ".join(sql.split())
        pg = self._pg
        self._one = None
        self._all = []
        self.rowcount = 0
        if norm.startswith("INSERT INTO marketplace_installs"):
            ref, version, source, repository_url, tag, artifact_ref, sha256, spec = params
            pg.rows[ref] = {
                "ref": ref,
                "version": version,
                "source": source,
                "repository_url": repository_url,
                "tag": tag,
                "artifact_ref": artifact_ref,
                "sha256": sha256,
                "spec": _unwrap(spec),
                "installed_at": pg.next_time(),
            }
        elif norm.startswith("SELECT"):
            if "WHERE ref" in norm:
                (ref,) = params
                self._one = pg.row_tuple(ref)
            else:  # ORDER BY ref — the full-list read
                self._all = [pg.row_tuple(ref) for ref in sorted(pg.rows)]
        elif norm.startswith("DELETE FROM marketplace_installs WHERE ref"):
            (ref,) = params
            if ref in pg.rows:
                del pg.rows[ref]
                self.rowcount = 1
        else:
            raise AssertionError(f"unhandled SQL in fake: {norm!r}")

    async def fetchone(self) -> Any:
        return self._one

    async def fetchall(self) -> list:
        return self._all


class _FakeConn:
    def __init__(self, pg: FakeMarketplacePg) -> None:
        self._pg = pg

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._pg)


class _FakePool:
    def __init__(self, pg: FakeMarketplacePg) -> None:
        self._pg = pg

    @asynccontextmanager
    async def connection(self):
        yield _FakeConn(self._pg)


class FakeMarketplacePg:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self._time_seq = 0

    def next_time(self) -> datetime:
        self._time_seq += 1
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self._time_seq)

    def row_tuple(self, ref: str) -> tuple | None:
        row = self.rows.get(ref)
        if row is None:
            return None
        return (
            row["ref"],
            row["version"],
            row["source"],
            row["repository_url"],
            row["tag"],
            row["artifact_ref"],
            row["sha256"],
            row["spec"],
            row["installed_at"],
        )


def make_pg_ctx(pg: FakeMarketplacePg):
    @asynccontextmanager
    async def _ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield _FakePool(pg)

    return _ctx


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch) -> FakeMarketplacePg:
    fake = FakeMarketplacePg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(fake))
    return fake


@pytest.fixture
def store() -> MarketplaceInstallStore:
    return MarketplaceInstallStore()


async def test_record_and_get_round_trip_pypi(pg, store) -> None:
    spec = {"namespace": "tai42", "name": "toolbox", "nested": {"a": [1, 2]}}
    await store.record("tai42/toolbox", "1.0.0", "pypi", None, None, None, None, spec)
    rec = await store.get("tai42/toolbox")
    assert rec is not None
    assert (rec.ref, rec.version, rec.source) == ("tai42/toolbox", "1.0.0", "pypi")
    assert rec.repository_url is None
    assert rec.tag is None
    # A pypi row keeps the github pin columns NULL — no fetch/verify on install.
    assert rec.artifact_ref is None
    assert rec.sha256 is None
    assert rec.spec == spec  # JSON round-trips untouched


async def test_record_github_round_trips_pin_and_verified_columns(pg, store) -> None:
    await store.record(
        "tai42/gh",
        "2.0.0",
        "github",
        "https://github.com/tai42ai/gh",
        "v2.0.0",
        "https://codeload.github.com/tai42ai/gh/tar.gz/refs/tags/v2.0.0",
        "a" * 64,
        {"k": 1},
    )
    rec = await store.get("tai42/gh")
    assert rec is not None
    assert rec.repository_url == "https://github.com/tai42ai/gh"
    assert rec.tag == "v2.0.0"
    # The verified-install columns round-trip so update-unwind can re-fetch them.
    assert rec.artifact_ref == "https://codeload.github.com/tai42ai/gh/tar.gz/refs/tags/v2.0.0"
    assert rec.sha256 == "a" * 64


async def test_record_upserts_on_conflict(pg, store) -> None:
    await store.record("tai42/toolbox", "1.0.0", "pypi", None, None, None, None, {"v": 1})
    await store.record("tai42/toolbox", "2.0.0", "pypi", None, None, None, None, {"v": 2})
    rec = await store.get("tai42/toolbox")
    assert rec is not None
    assert rec.version == "2.0.0"
    assert rec.spec == {"v": 2}
    assert len(pg.rows) == 1  # replaced, not duplicated


async def test_get_miss_returns_none(pg, store) -> None:
    assert await store.get("nope/gone") is None


async def test_delete_returns_true_then_false(pg, store) -> None:
    await store.record("tai42/toolbox", "1.0.0", "pypi", None, None, None, None, {})
    assert await store.delete("tai42/toolbox") is True
    assert await store.delete("tai42/toolbox") is False


async def test_list_installed_is_ordered_by_ref(pg, store) -> None:
    await store.record("tai42/zeta", "1.0.0", "pypi", None, None, None, None, {})
    await store.record("tai42/alpha", "1.0.0", "pypi", None, None, None, None, {})
    rows = await store.list_installed()
    assert [r.ref for r in rows] == ["tai42/alpha", "tai42/zeta"]
