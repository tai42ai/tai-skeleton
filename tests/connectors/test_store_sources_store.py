"""Read-only loader for the MCP-finder's allowed discovery sources."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import tai_skeleton.connectors.store.sources_store as sources_store
from tai_skeleton.connectors.store.sources_store import AllowedSource, fetch_allowed_sources


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        self.executed = " ".join(sql.split())

    async def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, pg):
        self._pg = pg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        cur = _FakeCursor(self._pg._rows)
        self._pg.cursors.append(cur)
        return cur


class _FakePg:
    def __init__(self, rows):
        self._rows = rows
        self.cursors: list = []

    def connection(self):
        return _FakeConn(self)


@pytest.fixture
def install_pg(monkeypatch):
    def _install(rows):
        pg = _FakePg(rows)

        @asynccontextmanager
        async def fake_client_ctx(client_cls, settings=None, **kwargs):
            yield pg

        monkeypatch.setattr(sources_store, "client_ctx", fake_client_ctx)
        return pg

    return _install


async def test_fetch_allowed_sources(install_pg):
    pg = install_pg([("github", "https://github.com"), ("pypi", "https://pypi.org")])
    sources = await fetch_allowed_sources()
    assert sources == [
        AllowedSource(id="github", url="https://github.com"),
        AllowedSource(id="pypi", url="https://pypi.org"),
    ]
    # Only enabled rows are read, ordered by id.
    executed = pg.cursors[0].executed
    assert "WHERE enabled" in executed
    assert "ORDER BY id" in executed


async def test_fetch_allowed_sources_empty(install_pg):
    install_pg([])
    assert await fetch_allowed_sources() == []
