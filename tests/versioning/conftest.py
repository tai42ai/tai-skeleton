"""A stateful fake Postgres for the generic versioned-document store tests.

Mirrors the repo's pg-backed store test pattern (``connectors/store`` fakes): a
stand-in that models the two tables in memory and interprets the store's SQL by
normalized prefix, monkeypatched in over the pooled ``client_ctx``. It is faithful
to the real Postgres semantics the store leans on:

* ``transaction()`` snapshots the tables on enter and RESTORES them on an
  exception (a real rollback), so a partial-failure write leaves no orphan;
* the partial-unique index ``versioned_documents_active_name_unique`` is enforced
  — inserting a second ACTIVE row for a ``(kind, name)`` raises a
  ``UniqueViolation`` carrying the index name (as psycopg reports it);
* the FK ``ON DELETE CASCADE`` removes a document's version rows with its row.

An injectable ``fault`` forces a chosen statement to raise, driving the
partial-failure rollback paths.
"""

from __future__ import annotations

import copy
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg.errors import UniqueViolation
from tai_kit.clients.impl.postgres import Json, PostgresClient

import tai_skeleton.versioning.store as store_module

_ACTIVE_NAME_UNIQUE_INDEX = "versioned_documents_active_name_unique"
_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


_DOC_VERSION_UNIQUE_CONSTRAINT = "versioned_document_versions_doc_version_unique"


class _ActiveNameViolation(UniqueViolation):
    """A UniqueViolation carrying the partial-unique index name, as psycopg reports
    a duplicate active ``(kind, name)``."""

    diag: Any = SimpleNamespace(constraint_name=_ACTIVE_NAME_UNIQUE_INDEX)


class _DocVersionViolation(UniqueViolation):
    """A UniqueViolation carrying the ``(document_id, version)`` unique-constraint
    name, as psycopg reports a duplicate append. The store never maps this one — a
    numbering regression surfaces as a raw violation here, exactly as real Postgres
    would raise it, so the fake cannot silently accept a colliding version."""

    diag: Any = SimpleNamespace(constraint_name=_DOC_VERSION_UNIQUE_CONSTRAINT)


def _unwrap(value: Any) -> Any:
    return value.obj if isinstance(value, Json) else value


class _FakeTxn:
    """Snapshot-and-restore savepoint: rolls the tables back on any exception."""

    def __init__(self, pg: FakeVersioningPg) -> None:
        self._pg = pg
        self._snapshot: tuple[list[dict], list[dict]] | None = None

    async def __aenter__(self) -> _FakeTxn:
        self._snapshot = (copy.deepcopy(self._pg.documents), copy.deepcopy(self._pg.versions))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and self._snapshot is not None:
            self._pg.documents, self._pg.versions = self._snapshot
        return False


class _FakeCursor:
    def __init__(self, pg: FakeVersioningPg) -> None:
        self._pg = pg
        self.rowcount = 0
        self._one: Any = None
        self._all: list = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, sql: str, params: tuple = ()) -> None:
        norm = " ".join(sql.split())
        pg = self._pg
        pg.executed.append(norm)
        if pg.fault is not None and norm.startswith(pg.fault[0]):
            raise pg.fault[1]
        self._one = None
        self._all = []
        self.rowcount = 0

        if norm.startswith("INSERT INTO versioned_documents"):
            kind, name = params
            if any(d["kind"] == kind and d["name"] == name and d["is_active"] for d in pg.documents):
                raise _ActiveNameViolation()
            doc = {
                "id": pg.next_doc_id(),
                "kind": kind,
                "name": name,
                "active_version": 1,
                "is_active": True,
                "created_at": pg.next_time(),
            }
            pg.documents.append(doc)
            self._one = (doc["id"], doc["created_at"])
        elif norm.startswith("INSERT INTO versioned_document_versions"):
            doc_id, version, body, tags = params
            if any(v["document_id"] == doc_id and v["version"] == version for v in pg.versions):
                raise _DocVersionViolation()
            pg.versions.append(
                {
                    "id": pg.next_ver_id(),
                    "document_id": doc_id,
                    "version": version,
                    "body": _unwrap(body),
                    "tags": list(tags),
                    "created_at": pg.next_time(),
                }
            )
            if "RETURNING created_at" in norm:
                self._one = (pg.versions[-1]["created_at"],)
        elif norm.startswith("SELECT MAX(version)"):
            (doc_id,) = params
            versions = [v["version"] for v in pg.versions if v["document_id"] == doc_id]
            self._one = (max(versions) if versions else None,)
        elif norm.startswith("UPDATE versioned_documents SET active_version"):
            active_version, doc_id = params
            for d in pg.documents:
                if d["id"] == doc_id:
                    d["active_version"] = active_version
        elif norm.startswith("UPDATE versioned_documents SET name"):
            new_name, kind, name = params
            # Mirror real Postgres: a missing source row updates zero rows (no
            # index evaluation), and the partial-unique active index on
            # (kind, new_name) never conflicts with the row being updated.
            doc = pg.active_doc(kind, name)
            if doc is not None:
                if any(
                    d["kind"] == kind and d["name"] == new_name and d["is_active"] and d is not doc
                    for d in pg.documents
                ):
                    raise _ActiveNameViolation()
                doc["name"] = new_name
                self.rowcount = 1
                if "RETURNING active_version, created_at" in norm:
                    self._one = (doc["active_version"], doc["created_at"])
        elif norm.startswith("UPDATE versioned_document_versions SET tags"):
            tags, doc_id, version = params
            ver = pg.version_row(doc_id, version)
            if ver is not None:
                ver["tags"] = list(tags)
                self.rowcount = 1
        elif norm.startswith("UPDATE versioned_documents SET is_active = FALSE"):
            kind, name = params
            for d in pg.documents:
                if d["kind"] == kind and d["name"] == name and d["is_active"]:
                    d["is_active"] = False
                    self.rowcount = 1
        elif norm.startswith("DELETE FROM versioned_documents"):
            kind, name = params
            doc = next((d for d in pg.documents if d["kind"] == kind and d["name"] == name and d["is_active"]), None)
            if doc is not None:
                pg.documents = [d for d in pg.documents if d["id"] != doc["id"]]
                pg.versions = [v for v in pg.versions if v["document_id"] != doc["id"]]  # FK cascade
                self.rowcount = 1
        elif norm.startswith("SELECT id, active_version FROM versioned_documents"):
            self._one = self._active_doc_tuple(params, ("id", "active_version"))
        elif norm.startswith("SELECT id, created_at FROM versioned_documents"):
            self._one = self._active_doc_tuple(params, ("id", "created_at"))
        elif norm.startswith("SELECT id FROM versioned_documents"):
            self._one = self._active_doc_tuple(params, ("id",))
        elif norm.startswith("SELECT active_version, created_at FROM versioned_documents"):
            self._one = self._active_doc_tuple(params, ("active_version", "created_at"))
        elif norm.startswith("SELECT kind, name, active_version, is_active, created_at FROM versioned_documents"):
            (kind,) = params
            self._all = [
                (d["kind"], d["name"], d["active_version"], d["is_active"], d["created_at"])
                for d in sorted(pg.documents, key=lambda d: d["name"])
                if d["kind"] == kind and d["is_active"]
            ]
        elif norm.startswith("SELECT d.name, v.body FROM versioned_documents d"):
            (kind,) = params
            rows: list[tuple[str, Any]] = []
            for doc in sorted(pg.documents, key=lambda d: d["name"]):
                if doc["kind"] != kind or not doc["is_active"]:
                    continue
                ver = pg.version_row(doc["id"], doc["active_version"])
                if ver is not None:
                    rows.append((doc["name"], ver["body"]))
            self._all = rows
        elif norm.startswith("SELECT v.body FROM versioned_documents d"):
            kind, name = params
            doc = pg.active_doc(kind, name)
            if doc is not None:
                ver = pg.version_row(doc["id"], doc["active_version"])
                if ver is not None:
                    self._one = (ver["body"],)
        elif norm.startswith("SELECT version, body, tags, created_at FROM versioned_document_versions"):
            (doc_id,) = params
            self._all = [
                (v["version"], v["body"], list(v["tags"]), v["created_at"])
                for v in sorted(pg.versions, key=lambda v: v["version"])
                if v["document_id"] == doc_id
            ]
        elif norm.startswith("SELECT d.active_version, v.body, v.tags, v.created_at FROM versioned_documents d"):
            kind, name, version = params
            doc = pg.active_doc(kind, name)
            if doc is not None:
                ver = pg.version_row(doc["id"], version)
                if ver is not None:
                    self._one = (doc["active_version"], ver["body"], list(ver["tags"]), ver["created_at"])
        elif norm.startswith("SELECT 1 FROM versioned_document_versions"):
            doc_id, version = params
            self._one = (1,) if pg.version_row(doc_id, version) is not None else None
        else:
            raise AssertionError(f"unhandled SQL in fake: {norm!r}")

    def _active_doc_tuple(self, params: tuple, cols: tuple[str, ...]) -> tuple | None:
        kind, name = params
        doc = self._pg.active_doc(kind, name)
        return None if doc is None else tuple(doc[c] for c in cols)

    async def fetchone(self) -> Any:
        return self._one

    async def fetchall(self) -> list:
        return self._all


class _FakeConn:
    def __init__(self, pg: FakeVersioningPg) -> None:
        self._pg = pg

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._pg)

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self._pg)


class FakeVersioningPg:
    """In-memory stand-in for the two versioned-document tables."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.versions: list[dict] = []
        self.executed: list[str] = []
        self.fault: tuple[str, Exception] | None = None
        self._doc_seq = 0
        self._ver_seq = 0
        self._time_seq = 0

    def connection(self) -> _FakeConn:
        return _FakeConn(self)

    def next_doc_id(self) -> int:
        self._doc_seq += 1
        return self._doc_seq

    def next_ver_id(self) -> int:
        self._ver_seq += 1
        return self._ver_seq

    def next_time(self) -> datetime:
        self._time_seq += 1
        return _BASE_TIME + timedelta(seconds=self._time_seq)

    def active_doc(self, kind: str, name: str) -> dict | None:
        return next((d for d in self.documents if d["kind"] == kind and d["name"] == name and d["is_active"]), None)

    def version_row(self, doc_id: int, version: int) -> dict | None:
        return next((v for v in self.versions if v["document_id"] == doc_id and v["version"] == version), None)


@pytest.fixture
def pg(monkeypatch) -> FakeVersioningPg:
    fake = FakeVersioningPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(store_module, "client_ctx", fake_client_ctx)
    return fake
