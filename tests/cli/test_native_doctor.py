"""``tai doctor`` — read-only diagnostics.

Covers the exit-code contract (non-zero when any dependency check fails) and the
credential-redaction rule (connection-URL passwords are masked, never echoed).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner
from redis.exceptions import ConnectionError as RedisConnectionError

from tai_skeleton.cli import app as app_module
from tai_skeleton.cli.native import doctor
from tai_skeleton.cli.native.db import SchemaAdminSettings
from tai_skeleton.cli.native.doctor import Check


def test_redact_url_masks_password() -> None:
    assert doctor._redact_url("redis://user:secret@host:6379/0") == "redis://user:***@host:6379/0"
    assert doctor._redact_url("redis://:secret@host:6379/0") == "redis://:***@host:6379/0"
    # No password -> unchanged; unset -> a clear marker, never a blank.
    assert doctor._redact_url("redis://host:6379/0") == "redis://host:6379/0"
    assert doctor._redact_url(None) == "(unset)"


def test_doctor_exits_non_zero_when_a_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _checks() -> list[Check]:
        return [Check("python", doctor._INFO, "3.13.0"), Check("redis", doctor._FAIL, "cannot connect to redis://h")]

    monkeypatch.setattr(doctor, "_run_checks", _checks)

    result = CliRunner().invoke(app_module.app, ["doctor"])

    assert result.exit_code == 1
    assert "redis" in result.output
    assert "fail" in result.output


def test_doctor_exits_zero_when_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _checks() -> list[Check]:
        return [Check("postgres", doctor._OK, "connected"), Check("redis", doctor._OK, "connected")]

    monkeypatch.setattr(doctor, "_run_checks", _checks)

    result = CliRunner().invoke(app_module.app, ["doctor"])

    assert result.exit_code == 0, result.output


def test_doctor_json_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    async def _checks() -> list[Check]:
        return [Check("postgres", doctor._OK, "connected")]

    monkeypatch.setattr(doctor, "_run_checks", _checks)

    result = CliRunner().invoke(app_module.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["check"] == "postgres"


def test_probe_redis_fails_and_redacts_password(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_settings = SimpleNamespace(redis=SimpleNamespace(redis_url="redis://:supersecret@cache:6379/0"))
    monkeypatch.setattr(
        "tai_skeleton.connectors.settings.connector_store_settings",
        lambda: fake_settings,
    )

    @asynccontextmanager
    async def _refusing(*args, **kwargs):
        raise RedisConnectionError("Connection refused")
        yield  # pragma: no cover - unreachable, marks this an async generator

    monkeypatch.setattr(doctor, "client_ctx", _refusing)

    check = asyncio.run(doctor._probe_redis())

    assert check.status == doctor._FAIL
    assert "supersecret" not in check.detail
    assert "***" in check.detail


# -- probe internals ---------------------------------------------------------

# A stand-in for the schema-admin settings the probes only read for a target
# description; the fake client makes the real connection fields irrelevant.
_TARGET = cast("SchemaAdminSettings", SimpleNamespace(pg_host="db", pg_port=5432, pg_db="tai"))


class _FakeCursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    async def execute(self, *args: object) -> None:
        return None

    async def fetchall(self) -> list[tuple[str]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[str]] | None = None) -> None:
        self._rows = rows or []

    async def execute(self, *args: object) -> None:
        return None

    @asynccontextmanager
    async def cursor(self):
        yield _FakeCursor(self._rows)


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self._conn


def _yielding(pool: _FakePool):
    @asynccontextmanager
    async def factory(*args: object, **kwargs: object):
        yield pool

    return factory


def _refusing():
    import psycopg

    @asynccontextmanager
    async def factory(*args: object, **kwargs: object):
        raise psycopg.OperationalError("connection refused")
        yield  # pragma: no cover - unreachable, marks this an async generator

    return factory


def test_probe_postgres_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "client_ctx", _yielding(_FakePool(_FakeConn())))
    check = asyncio.run(doctor._probe_postgres(_TARGET))
    assert check.status == doctor._OK
    assert "db:5432/tai" in check.detail


def test_probe_postgres_fail_reports_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "client_ctx", _refusing())
    check = asyncio.run(doctor._probe_postgres(_TARGET))
    assert check.status == doctor._FAIL
    assert "cannot connect to db:5432/tai" in check.detail


def test_probe_schema_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai_skeleton.cli.native import doctor as doctor_mod

    expected = doctor_mod.expected_tables()
    rows = [(table,) for table in expected]
    monkeypatch.setattr(doctor, "client_ctx", _yielding(_FakePool(_FakeConn(rows))))
    check = asyncio.run(doctor._probe_schema(_TARGET))
    assert check.status == doctor._OK
    assert str(len(expected)) in check.detail


def test_probe_schema_reports_missing_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai_skeleton.cli.native import doctor as doctor_mod

    expected = doctor_mod.expected_tables()
    # Report every table but the first as present, so the first is flagged missing.
    rows = [(table,) for table in expected[1:]]
    monkeypatch.setattr(doctor, "client_ctx", _yielding(_FakePool(_FakeConn(rows))))
    check = asyncio.run(doctor._probe_schema(_TARGET))
    assert check.status == doctor._FAIL
    assert "missing tables" in check.detail
    assert expected[0] in check.detail


def test_probe_schema_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "client_ctx", _refusing())
    check = asyncio.run(doctor._probe_schema(_TARGET))
    assert check.status == doctor._FAIL
    assert "cannot inspect schema" in check.detail


def test_probe_redis_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_settings = SimpleNamespace(redis=SimpleNamespace(redis_url="redis://cache:6379/0"))
    monkeypatch.setattr(
        "tai_skeleton.connectors.settings.connector_store_settings",
        lambda: fake_settings,
    )

    class _PingClient:
        async def ping(self) -> bool:
            return True

    @asynccontextmanager
    async def _ok(*args: object, **kwargs: object):
        yield _PingClient()

    monkeypatch.setattr(doctor, "client_ctx", _ok)
    check = asyncio.run(doctor._probe_redis())
    assert check.status == doctor._OK
    assert "connected to redis://cache:6379/0" in check.detail


def test_run_checks_includes_schema_when_postgres_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "schema_settings", lambda: _TARGET)

    async def _pg(settings: object) -> Check:
        return Check("postgres", doctor._OK, "connected")

    async def _schema(settings: object) -> Check:
        return Check("schema", doctor._OK, "all present")

    async def _redis() -> Check:
        return Check("redis", doctor._OK, "connected")

    monkeypatch.setattr(doctor, "_probe_postgres", _pg)
    monkeypatch.setattr(doctor, "_probe_schema", _schema)
    monkeypatch.setattr(doctor, "_probe_redis", _redis)

    checks = asyncio.run(doctor._run_checks())
    names = [c.name for c in checks]
    assert names == ["python", "config-mode", "postgres", "schema", "redis"]


def test_run_checks_skips_schema_when_postgres_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "schema_settings", lambda: _TARGET)

    async def _pg(settings: object) -> Check:
        return Check("postgres", doctor._FAIL, "down")

    async def _schema(settings: object) -> Check:  # pragma: no cover - must be skipped
        raise AssertionError("schema probe must be skipped when Postgres is unreachable")

    async def _redis() -> Check:
        return Check("redis", doctor._OK, "connected")

    monkeypatch.setattr(doctor, "_probe_postgres", _pg)
    monkeypatch.setattr(doctor, "_probe_schema", _schema)
    monkeypatch.setattr(doctor, "_probe_redis", _redis)

    checks = asyncio.run(doctor._run_checks())
    names = [c.name for c in checks]
    assert "schema" not in names
    assert names == ["python", "config-mode", "postgres", "redis"]
