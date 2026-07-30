"""A stateful in-memory fake Postgres for the tool-metadata store tests.

``FakeToolMetaPg`` models the two tables the overlay store touches
(``tool_folders`` — a tree of real folder entities, and ``tool_meta`` — a per-tool
overlay row) and interprets the store's EXACT SQL by normalized text,
monkeypatched in over the pooled ``client_ctx`` so the REAL
:class:`~tai42_skeleton.tool_meta.store.PostgresToolMetaStore` runs against it with
no live database. It is faithful to the Postgres semantics the store leans on:

* the ``tool_folders_parent_name_unique`` constraint is enforced with
  ``NULLS NOT DISTINCT`` — two folders sharing a ``(parent_id, name)`` collide
  (root folders share the ``None`` parent, so duplicate root names collide too);
  the fake raises a real ``psycopg.errors.UniqueViolation`` on the offending
  INSERT/UPDATE, which the store maps to :class:`FolderNameConflictError`;
* ``tool_meta`` is keyed by ``tool_name`` (a PK), and ``upsert_meta``'s
  ``ON CONFLICT (tool_name) DO UPDATE`` re-writes the mutable overlay columns;
* ``transaction()`` snapshots both tables on enter and RESTORES them on an
  exception (a real rollback), so a raising multi-statement change leaves no
  orphan;
* a statement that raises INSIDE an open transaction poisons it exactly as real
  Postgres does — every later ``execute`` on that transaction raises
  ``psycopg.errors.InFailedSqlTransaction`` until the transaction unwinds (a
  rollback), so a store path that queries after a caught error is caught here too.

The same fake backs the router tests (imported from there), which drive the live
store through the operation/route layer.
"""

from __future__ import annotations

import copy
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pytest
from psycopg.errors import InFailedSqlTransaction, UniqueViolation
from tai42_kit.clients.impl.postgres import PostgresClient

import tai42_skeleton.tool_meta.store as store_module
from tai42_skeleton.tool_meta.store import PostgresToolMetaStore


class _FakeTxn:
    """Snapshot-and-restore savepoint: rolls both tables back on any exception, so
    a multi-statement mutation that raises leaves the store state untouched, and
    clears the connection's failed-transaction flag on unwind (the rollback)."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._pg = conn._pg
        self._snapshot: tuple[dict[str, dict], dict[str, dict]] | None = None

    async def __aenter__(self) -> _FakeTxn:
        self._snapshot = (copy.deepcopy(self._pg.folders), copy.deepcopy(self._pg.meta))
        self._conn._in_txn = True
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None and self._snapshot is not None:
            self._pg.folders, self._pg.meta = self._snapshot
        self._conn._in_txn = False
        self._conn._failed = False
        return False


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._pg = conn._pg
        self._one: Any = None
        self._all: list[Any] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        # An open transaction that already saw a statement raise is aborted: every
        # further statement fails until rollback, exactly as real Postgres does.
        if self._conn._failed:
            raise InFailedSqlTransaction()
        try:
            await self._dispatch(sql, params)
        except Exception:
            if self._conn._in_txn:
                self._conn._failed = True
            raise

    async def _dispatch(self, sql: str, params: tuple[Any, ...]) -> None:
        norm = " ".join(sql.split())
        pg = self._pg
        self._one = None
        self._all = []

        # -- tool_folders writes ---------------------------------------------
        if norm.startswith("INSERT INTO tool_folders"):
            name, parent_id = params
            pg.reject_folder_name_conflict(name, parent_id, exclude=None)
            folder = {"id": pg.new_folder_id(), "name": name, "parent_id": parent_id}
            pg.folders[folder["id"]] = folder
            self._one = (folder["id"], folder["name"], folder["parent_id"])
        elif norm.startswith("UPDATE tool_folders SET name"):
            name, folder_id = params
            folder = pg.folders[folder_id]
            pg.reject_folder_name_conflict(name, folder["parent_id"], exclude=folder_id)
            folder["name"] = name
            self._one = (folder["id"], folder["name"], folder["parent_id"])
        elif norm.startswith("UPDATE tool_folders SET parent_id"):
            parent_id, folder_id = params
            folder = pg.folders[folder_id]
            pg.reject_folder_name_conflict(folder["name"], parent_id, exclude=folder_id)
            folder["parent_id"] = parent_id
            self._one = (folder["id"], folder["name"], folder["parent_id"])
        elif norm.startswith("DELETE FROM tool_folders WHERE id"):
            (folder_id,) = params
            pg.folders.pop(folder_id, None)

        # -- tool_folders reads ----------------------------------------------
        elif norm == "SELECT id, name, parent_id FROM tool_folders ORDER BY name":
            self._all = [
                (f["id"], f["name"], f["parent_id"]) for f in sorted(pg.folders.values(), key=lambda f: f["name"])
            ]
        elif norm == "SELECT id, parent_id FROM tool_folders":
            self._all = [(f["id"], f["parent_id"]) for f in pg.folders.values()]
        elif norm == "SELECT name, parent_id FROM tool_folders WHERE id = %s":
            (folder_id,) = params
            folder = pg.folders.get(folder_id)
            self._one = None if folder is None else (folder["name"], folder["parent_id"])
        elif norm == "SELECT 1 FROM tool_folders WHERE parent_id = %s LIMIT 1":
            (parent_id,) = params
            self._one = (1,) if any(f["parent_id"] == parent_id for f in pg.folders.values()) else None
        elif norm == "SELECT 1 FROM tool_meta WHERE folder_id = %s LIMIT 1":
            (folder_id,) = params
            self._one = (1,) if any(m["folder_id"] == folder_id for m in pg.meta.values()) else None

        # -- tool_meta writes ------------------------------------------------
        elif norm == "INSERT INTO tool_meta (tool_name) VALUES (%s) ON CONFLICT (tool_name) DO NOTHING":
            # ``merge_meta`` materializes the row so the FOR UPDATE below has one to
            # lock; DO NOTHING when it already exists.
            (tool_name,) = params
            pg.meta.setdefault(
                tool_name,
                {"tool_name": tool_name, "display_name": None, "folder_id": None, "tags": [], "hidden": None},
            )
        elif norm.startswith("UPDATE tool_meta SET display_name"):
            display_name, folder_id, tags, hidden, tool_name = params
            row = pg.meta[tool_name]
            row.update({"display_name": display_name, "folder_id": folder_id, "tags": list(tags), "hidden": hidden})
            self._one = pg.meta_tuple(tool_name)
        elif norm.startswith("INSERT INTO tool_meta"):
            tool_name, display_name, folder_id, tags, hidden = params
            pg.meta[tool_name] = {
                "tool_name": tool_name,
                "display_name": display_name,
                "folder_id": folder_id,
                "tags": list(tags),
                "hidden": hidden,
            }
            self._one = pg.meta_tuple(tool_name)
        elif norm.startswith("DELETE FROM tool_meta WHERE tool_name"):
            (tool_name,) = params
            pg.meta.pop(tool_name, None)
        elif norm.startswith("UPDATE tool_meta SET tool_name"):
            new_name, old_name = params
            row = pg.meta.pop(old_name, None)
            if row is not None:
                row["tool_name"] = new_name
                pg.meta[new_name] = row

        # -- tool_meta reads -------------------------------------------------
        elif norm == "SELECT display_name, folder_id, tags, hidden FROM tool_meta WHERE tool_name = %s FOR UPDATE":
            (tool_name,) = params
            row = pg.meta.get(tool_name)
            self._one = (
                None if row is None else (row["display_name"], row["folder_id"], list(row["tags"]), row["hidden"])
            )
        elif norm == "SELECT tool_name, display_name, folder_id, tags, hidden FROM tool_meta WHERE tool_name = %s":
            (tool_name,) = params
            self._one = pg.meta_tuple(tool_name)
        elif norm == "SELECT tool_name, display_name, folder_id, tags, hidden FROM tool_meta ORDER BY tool_name":
            self._all = [pg.meta_tuple(name) for name in sorted(pg.meta)]
        else:
            raise AssertionError(f"unhandled SQL in fake: {norm!r}")

    async def fetchone(self) -> Any:
        return self._one

    async def fetchall(self) -> list[Any]:
        return self._all


class _FakeConn:
    def __init__(self, pg: FakeToolMetaPg) -> None:
        self._pg = pg
        self._in_txn = False
        self._failed = False

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakePool:
    def __init__(self, pg: FakeToolMetaPg) -> None:
        self._pg = pg

    @asynccontextmanager
    async def connection(self):
        yield _FakeConn(self._pg)


class FakeToolMetaPg:
    """In-memory ``tool_folders`` + ``tool_meta`` the store's SQL runs against."""

    def __init__(self) -> None:
        self.folders: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, dict[str, Any]] = {}

    def new_folder_id(self) -> str:
        return str(uuid.uuid4())

    def reject_folder_name_conflict(self, name: str, parent_id: str | None, *, exclude: str | None) -> None:
        """Raise a real ``UniqueViolation`` when another folder already holds
        ``(parent_id, name)`` — the ``NULLS NOT DISTINCT`` sibling-uniqueness the
        store maps to :class:`FolderNameConflictError`. ``exclude`` skips the row
        being updated in place."""
        for folder_id, folder in self.folders.items():
            if folder_id == exclude:
                continue
            if folder["parent_id"] == parent_id and folder["name"] == name:
                raise UniqueViolation()

    def meta_tuple(self, tool_name: str) -> tuple[Any, ...] | None:
        row = self.meta.get(tool_name)
        if row is None:
            return None
        return (row["tool_name"], row["display_name"], row["folder_id"], list(row["tags"]), row["hidden"])


def make_pg_ctx(pg: FakeToolMetaPg):
    """A ``client_ctx`` stand-in yielding a fake pool over ``pg`` — asserts the
    store still asks for a ``PostgresClient``."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake: {client_cls!r}")
        yield _FakePool(pg)

    return _ctx


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch) -> FakeToolMetaPg:
    fake = FakeToolMetaPg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(fake))
    return fake


@pytest.fixture
def store() -> PostgresToolMetaStore:
    return PostgresToolMetaStore()
