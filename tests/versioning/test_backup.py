"""Store-level versioned-document backup: the two-table SQL round-trip.

Postgres is faked at the kit ``client_ctx`` seam (a stateful in-memory model of
the ``versioned_documents`` + ``versioned_document_versions`` tables) so the
export/import SQL is exercised with no real database. The section is registered
``secret=True`` and is kind-agnostic — ONE section carries EVERY kind — which the
round-trip pins by populating rows across two kinds.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from tai42_kit.clients.impl.postgres import Json, PostgresClient

import tai42_skeleton.versioning.backup as versioning_backup
from tai42_skeleton.backup.registry import BackupRegistry
from tai42_skeleton.backup.sections import register_core_sections
from tai42_skeleton.versioning.backup import export_versioned_documents, import_versioned_documents

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _unwrap(value: Any) -> Any:
    return value.obj if isinstance(value, Json) else value


class _FakeCursor:
    def __init__(self, pg: _FakeVersioningBackupPg) -> None:
        self._pg = pg
        self._one: Any = None
        self._all: list = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, sql: str, params: tuple = ()) -> None:
        norm = " ".join(sql.split())
        pg = self._pg
        self._one = None
        self._all = []

        if norm.startswith("SELECT id, kind, name, active_version, is_active, created_at FROM versioned_documents"):
            self._all = [
                (d["id"], d["kind"], d["name"], d["active_version"], d["is_active"], d["created_at"])
                for d in sorted(pg.documents, key=lambda d: d["id"])
            ]
        elif norm.startswith(
            "SELECT id, document_id, version, body, tags, created_at FROM versioned_document_versions"
        ):
            self._all = [
                (v["id"], v["document_id"], v["version"], v["body"], list(v["tags"]), v["created_at"])
                for v in sorted(pg.versions, key=lambda v: v["id"])
            ]
        elif norm.startswith("SELECT id FROM versioned_documents"):
            self._all = [(d["id"],) for d in pg.documents]
        elif norm.startswith("INSERT INTO versioned_documents"):
            doc_id, kind, name, active_version, is_active, created_at = params
            existing = next((d for d in pg.documents if d["id"] == doc_id), None)
            row = {
                "id": doc_id,
                "kind": kind,
                "name": name,
                "active_version": active_version,
                "is_active": is_active,
                "created_at": created_at,
            }
            if existing is not None:
                existing.update(row)
            else:
                pg.documents.append(row)
        elif norm.startswith("INSERT INTO versioned_document_versions"):
            ver_id, document_id, version, body, tags, created_at = params
            row = {
                "id": ver_id,
                "document_id": document_id,
                "version": version,
                "body": _unwrap(body),
                "tags": list(tags),
                "created_at": created_at,
            }
            existing = next((v for v in pg.versions if v["id"] == ver_id), None)
            if existing is not None:
                existing.update(row)
            else:
                pg.versions.append(row)
        elif norm.startswith("SELECT setval"):
            # Sequence bump — nothing to model in-memory; accept and return a value.
            self._one = (0,)
        else:
            raise AssertionError(f"unhandled SQL in fake: {norm!r}")

    async def fetchone(self) -> Any:
        return self._one

    async def fetchall(self) -> list:
        return self._all


class _FakeConn:
    def __init__(self, pg: _FakeVersioningBackupPg) -> None:
        self._pg = pg

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._pg)


class _FakeVersioningBackupPg:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.versions: list[dict] = []

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


@pytest.fixture
def pg(monkeypatch) -> _FakeVersioningBackupPg:
    fake = _FakeVersioningBackupPg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield fake

    monkeypatch.setattr(versioning_backup, "client_ctx", fake_client_ctx)
    return fake


def _seed(pg: _FakeVersioningBackupPg) -> None:
    """Two kinds, multiple versions, non-trivial active pointers, plus a
    soft-deleted ghost — the state the round-trip must restore intact."""
    # kind=preset "wv": v1,v2,v3 with active pointer rolled back to 2.
    pg.documents.append(
        {"id": 1, "kind": "preset", "name": "wv", "active_version": 2, "is_active": True, "created_at": _T0}
    )
    for n, units in ((1, "a"), (2, "b"), (3, "c")):
        pg.versions.append(
            {"id": n, "document_id": 1, "version": n, "body": {"units": units}, "tags": ["cat"], "created_at": _T0}
        )
    # kind=ac_policy "guard": v1,v2 active at 2 — proves kind-agnostic coverage.
    pg.documents.append(
        {"id": 2, "kind": "ac_policy", "name": "guard", "active_version": 2, "is_active": True, "created_at": _T0}
    )
    pg.versions.append({"id": 4, "document_id": 2, "version": 1, "body": {"rule": "x"}, "tags": [], "created_at": _T0})
    pg.versions.append({"id": 5, "document_id": 2, "version": 2, "body": {"rule": "y"}, "tags": [], "created_at": _T0})
    # A soft-deleted ghost of "wv" with its own history — audit rows survive too.
    pg.documents.append(
        {"id": 3, "kind": "preset", "name": "wv", "active_version": 1, "is_active": False, "created_at": _T0}
    )
    pg.versions.append(
        {"id": 6, "document_id": 3, "version": 1, "body": {"units": "old"}, "tags": [], "created_at": _T0}
    )


async def test_backup_round_trip_restores_documents_and_history(pg: _FakeVersioningBackupPg):
    _seed(pg)
    original_docs = sorted((dict(d) for d in pg.documents), key=lambda d: d["id"])
    original_versions = sorted((dict(v) for v in pg.versions), key=lambda v: v["id"])

    payload = await export_versioned_documents()

    # Wipe both tables, then import the payload.
    pg.documents.clear()
    pg.versions.clear()
    report = await import_versioned_documents(payload)

    # Every document (across BOTH kinds + the soft-deleted ghost) restored with its
    # active_version pointer and is_active flag intact.
    restored_docs = sorted((dict(d) for d in pg.documents), key=lambda d: d["id"])
    assert restored_docs == original_docs
    # Every version row restored under its original document_id (the FK link).
    restored_versions = sorted((dict(v) for v in pg.versions), key=lambda v: v["id"])
    assert restored_versions == original_versions

    # Wipe → import is all-new.
    assert report["created"] == 3
    assert report["updated"] == 0
    assert report["errors"] == []


async def test_backup_reimport_over_existing_is_idempotent(pg: _FakeVersioningBackupPg):
    _seed(pg)
    payload = await export_versioned_documents()

    # Re-import over the already-present rows: same ids, so every document upserts
    # as "updated" and the tables are unchanged.
    report = await import_versioned_documents(payload)

    assert report["created"] == 0
    assert report["updated"] == 3
    assert len(pg.documents) == 3
    assert len(pg.versions) == 6


def test_versioned_documents_section_registered_secret():
    registry = BackupRegistry()
    register_core_sections(registry)
    section = next(s for s in registry.sections() if s.name == "versioned_documents")
    assert section.secret is True
