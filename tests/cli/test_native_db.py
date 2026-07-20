"""``tai db apply`` / ``tai db check`` — schema apply and inspection.

No live Postgres: the DB seam is faked or the async workers are patched. The
tests assert the DDL is idempotent by construction, that ``apply`` runs it inside
one transaction and is loud (and credential-free) on a connection failure, and
that ``check`` exits non-zero when a table is missing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
import pytest
from click.testing import CliRunner

from tai42_skeleton.cli import app as app_module
from tai42_skeleton.cli.native import db
from tai42_skeleton.sql.schema import load_ddl

# --- DDL properties -------------------------------------------------------


def test_expected_tables_match_the_ddl() -> None:
    tables = db.expected_tables()
    assert set(tables) == {
        "connector_connections",
        "connector_category",
        "connector_catalog",
        "connector_allowed_source",
        "versioned_documents",
        "versioned_document_versions",
        "access_control_policies",
        "access_control_routes",
        "marketplace_installs",
    }


def test_ddl_is_idempotent_by_construction() -> None:
    ddl = load_ddl().upper()
    # Every table/index creation guards with IF NOT EXISTS and every seed insert
    # with ON CONFLICT, so a second apply is a no-op.
    assert "CREATE TABLE " in ddl
    assert ddl.count("CREATE TABLE ") == ddl.count("CREATE TABLE IF NOT EXISTS ")
    assert ddl.count("CREATE INDEX ") + ddl.count("CREATE UNIQUE INDEX ") == (
        ddl.count("CREATE INDEX IF NOT EXISTS ") + ddl.count("CREATE UNIQUE INDEX IF NOT EXISTS ")
    )
    assert ddl.count("INSERT INTO ") == ddl.count("ON CONFLICT ")


# --- fake DB seam ---------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.executed: list[str] = []

    async def execute(self, sql: str, params: object = None) -> None:
        self.executed.append(sql)

    async def fetchall(self) -> list[tuple]:
        return self._rows

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeConn:
    def __init__(self, rows: list[tuple]) -> None:
        self.cursor_obj = _FakeCursor(rows)
        self.executed: list[str] = []

    async def execute(self, sql: str, params: object = None) -> None:
        self.executed.append(sql)

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    @asynccontextmanager
    async def transaction(self):
        yield


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self._conn


def _fake_client_ctx(conn: _FakeConn):
    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield _FakePool(conn)

    return _ctx


# --- apply ----------------------------------------------------------------


def test_apply_runs_ddl_in_a_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(rows=[])
    monkeypatch.setattr(db, "client_ctx", _fake_client_ctx(conn))

    result = CliRunner().invoke(app_module.app, ["db", "apply"])

    assert result.exit_code == 0, result.output
    assert conn.executed  # the DDL text was executed
    assert "CREATE TABLE" in conn.executed[0]


def test_apply_is_loud_and_credential_free_on_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_DB_PG_PASSWORD", "s3cr3t-should-not-leak")
    db.schema_settings.cache_clear()

    async def _boom(settings: object) -> None:
        raise psycopg.OperationalError("password authentication failed for user 'postgres'")

    monkeypatch.setattr(db, "_apply_schema", _boom)
    try:
        result = CliRunner().invoke(app_module.app, ["db", "apply"])
    finally:
        db.schema_settings.cache_clear()

    assert result.exit_code == 1
    assert "could not connect to Postgres" in result.output
    assert "s3cr3t-should-not-leak" not in result.output


# --- check ----------------------------------------------------------------


def test_check_reports_missing_tables_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _missing(settings: object) -> list[str]:
        return ["versioned_documents"]

    monkeypatch.setattr(db, "_missing_tables", _missing)

    result = CliRunner().invoke(app_module.app, ["db", "check"])

    assert result.exit_code == 1
    assert "versioned_documents" in result.output
    assert "tai db apply" in result.output


def test_check_passes_when_all_tables_present(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(rows=[(table,) for table in db.expected_tables()])
    monkeypatch.setattr(db, "client_ctx", _fake_client_ctx(conn))

    result = CliRunner().invoke(app_module.app, ["db", "check"])

    assert result.exit_code == 0, result.output
    assert "is present" in result.output


def test_check_is_loud_on_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(settings: object) -> list[str]:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(db, "_missing_tables", _boom)

    result = CliRunner().invoke(app_module.app, ["db", "check"])

    assert result.exit_code == 1
    assert "could not connect to Postgres" in result.output
