"""Validated community catalog write path: create_category + add_provider."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from tai_contract.connectors.probe import ToolSummary, VerifyResult

import tai_skeleton.connectors.store.catalog_write as catalog_write
from tai_skeleton.app import instance
from tai_skeleton.connectors.store.catalog_store import ConnectorCategory
from tai_skeleton.connectors.store.catalog_write import add_provider, create_category
from tests._fakes.bus import FakeBus

from .conftest import make_noauth_http_descriptor, make_oauth_descriptor


class _FakeCursor:
    def __init__(self, pg: _FakePg) -> None:
        self._pg = pg
        self.rowcount = 1
        self._one = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._pg.executed.append((norm, tuple(params)))
        if "INSERT INTO connector_category" in norm:
            # Single INSERT..SELECT..RETURNING statement: a conflict returns no
            # row, success returns the computed sort_order.
            self._one = None if self._pg.category_conflict else (self._pg.next_sort,)
        elif "INSERT INTO connector_catalog" in norm:
            self.rowcount = 0 if self._pg.provider_conflict else 1

    async def fetchone(self):
        return self._one


class _FakeConn:
    def __init__(self, pg):
        self._pg = pg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self._pg)


class _FakePg:
    def __init__(self) -> None:
        self.next_sort = 5
        self.category_conflict = False
        self.provider_conflict = False
        self.executed: list[tuple[str, tuple]] = []

    def connection(self):
        return _FakeConn(self)

    def find(self, needle: str) -> tuple[str, tuple]:
        return next((norm, params) for norm, params in self.executed if needle in norm)


class _FakeAppImpl:
    def __init__(self, backend) -> None:
        # The app exposes the backend through the ``backends`` facet namespace,
        # never a flat ``tai_app.backend``. ``add_provider`` runs a FULL local reload
        # (``admin.reload_config``) then broadcasts it on the bus.
        self.backends = SimpleNamespace(backend=backend)
        self.reload_calls = 0
        self.admin = SimpleNamespace(reload_config=self._reload)
        self.bus: FakeBus | None = None

    def _reload(self) -> dict:
        self.reload_calls += 1
        return {"status": "ok"}


@pytest.fixture
def fake_pg(monkeypatch):
    pg = _FakePg()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield pg

    monkeypatch.setattr(catalog_write, "client_ctx", fake_client_ctx)
    return pg


@pytest.fixture
def bind_app(monkeypatch):
    from tai_contract.app import tai_app

    def _bind(backend) -> _FakeAppImpl:
        impl = _FakeAppImpl(backend)
        monkeypatch.setattr(tai_app, "_impl", impl)
        bus = FakeBus(origin="serve-x")
        monkeypatch.setattr(instance.app, "_bus", bus)
        impl.bus = bus
        return impl

    return _bind


# -- create_category ---------------------------------------------------------


async def test_create_category_success(fake_pg):
    out = await create_category("dev-tools", "Dev Tools")
    assert out == ConnectorCategory(id="dev-tools", display_name="Dev Tools", sort_order=5)


async def test_create_category_rejects_bad_id(fake_pg):
    with pytest.raises(ValueError, match="kebab-case"):
        await create_category("Bad_ID", "X")


async def test_create_category_rejects_empty_display_name(fake_pg):
    with pytest.raises(ValueError, match="non-empty"):
        await create_category("dev-tools", "   ")


async def test_create_category_conflict_raises(fake_pg):
    fake_pg.category_conflict = True
    with pytest.raises(ValueError, match="already exists"):
        await create_category("dev-tools", "Dev Tools")


async def test_create_category_reaching_sentinel_raises(fake_pg):
    # A computed sort_order at/after the 'other' sentinel (1000) cannot stay
    # sorted last — reject it (the insert rolls back).
    fake_pg.next_sort = 1000
    with pytest.raises(ValueError, match="sentinel"):
        await create_category("dev-tools", "Dev Tools")


async def test_create_category_advisory_lock_serializes(fake_pg):
    # The MAX(sort_order)+1 read and its insert run under a transaction-scoped
    # advisory lock so racing creates cannot collide on sort_order.
    await create_category("dev-tools", "Dev Tools")
    assert any("pg_advisory_xact_lock" in norm for norm, _ in fake_pg.executed)


# -- add_provider ------------------------------------------------------------


def _patch_verify(monkeypatch, result: VerifyResult):
    async def _verify(descriptor, sub_service, *, config_values=None):
        return result

    monkeypatch.setattr(catalog_write, "verify", _verify)


def _community_descriptor(provider_id="catprov"):
    desc = make_noauth_http_descriptor(provider_id=provider_id)
    object.__setattr__(desc, "origin", "community")
    return desc


async def test_add_provider_rejects_oauth_kind(fake_pg):
    desc = make_oauth_descriptor(provider_id="catprov")
    with pytest.raises(ValueError, match="kind='none'"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


async def test_add_provider_rejects_non_community(fake_pg):
    desc = make_noauth_http_descriptor(provider_id="catprov")  # origin=system
    with pytest.raises(ValueError, match="community rows"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


async def test_add_provider_rejects_empty_added_by(fake_pg):
    desc = _community_descriptor()
    with pytest.raises(ValueError, match="requires added_by"):
        await add_provider(desc, source_url="https://x", added_by="  ", config_values={})


async def test_add_provider_unknown_category_without_create(fake_pg, monkeypatch):
    monkeypatch.setattr(catalog_write, "fetch_categories", _async_return([]))
    desc = _community_descriptor()
    with pytest.raises(ValueError, match="unknown category"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


async def test_add_provider_collision_with_existing(fake_pg, monkeypatch):
    monkeypatch.setattr(
        catalog_write,
        "fetch_categories",
        _async_return([ConnectorCategory(id="data", display_name="Data", sort_order=1)]),
    )
    desc = _community_descriptor()
    # get_provider succeeds → collision.
    monkeypatch.setattr(catalog_write, "get_provider", lambda pid: desc)
    with pytest.raises(ValueError, match="already exists"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


async def test_add_provider_verification_failure(fake_pg, monkeypatch):
    monkeypatch.setattr(
        catalog_write,
        "fetch_categories",
        _async_return([ConnectorCategory(id="data", display_name="Data", sort_order=1)]),
    )
    _ensure_unknown_provider(monkeypatch)
    _patch_verify(monkeypatch, VerifyResult(ok=False, error="no answer"))
    desc = _community_descriptor()
    with pytest.raises(ValueError, match="verification failed"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


async def test_add_provider_success_reloads_fleet(fake_pg, monkeypatch, bind_app):
    monkeypatch.setattr(
        catalog_write,
        "fetch_categories",
        _async_return([ConnectorCategory(id="data", display_name="Data", sort_order=1)]),
    )
    _ensure_unknown_provider(monkeypatch)
    _patch_verify(monkeypatch, VerifyResult(ok=True, tools=[ToolSummary(name="t1")]))
    impl = bind_app(None)
    desc = _community_descriptor()
    tools = await add_provider(
        desc,
        source_url="https://x",
        added_by="me",
        config_values={"token": "v"},
    )
    assert [t.name for t in tools] == ["t1"]
    # The FULL local reload ran once (no narrower op exists), then it was broadcast.
    assert impl.reload_calls == 1
    assert impl.bus is not None
    assert impl.bus.publish_calls[0][0] == {"op": "reload_config"}

    # The catalog INSERT carries exactly (provider_id, descriptor_json, origin,
    # category, source_url, added_by) — and the stored jsonb must NOT embed the
    # origin/category columns (fetch_catalog rejects rows that do).
    insert_sql, params = fake_pg.find("INSERT INTO connector_catalog")
    assert "(provider_id, descriptor, origin, category, source_url, added_by)" in insert_sql
    provider_id, descriptor_json, origin, category, source_url, added_by = params
    assert provider_id == "catprov"
    assert origin == "community"
    assert category == "data"
    assert source_url == "https://x"
    assert added_by == "me"
    payload = descriptor_json.obj  # psycopg Json wrapper
    assert "origin" not in payload
    assert "category" not in payload


async def test_add_provider_creates_new_category(fake_pg, monkeypatch, bind_app):
    _ensure_unknown_provider(monkeypatch)
    _patch_verify(monkeypatch, VerifyResult(ok=True, tools=[]))
    bind_app(None)
    desc = _community_descriptor()
    object.__setattr__(desc, "category", "brand-new")
    tools = await add_provider(
        desc,
        source_url="https://x",
        added_by="me",
        config_values={},
        new_category_display_name="Brand New",
    )
    assert tools == []
    # The category is created in the same write path as the catalog insert, under
    # the advisory lock that serializes MAX(sort_order)+1.
    assert any("pg_advisory_xact_lock" in norm for norm, _ in fake_pg.executed)
    _cat_sql, cat_params = fake_pg.find("INSERT INTO connector_category")
    assert cat_params == ("brand-new", "Brand New", "other")
    assert any("INSERT INTO connector_catalog" in norm for norm, _ in fake_pg.executed)


async def test_add_provider_success_no_backend(fake_pg, monkeypatch, bind_app):
    monkeypatch.setattr(
        catalog_write,
        "fetch_categories",
        _async_return([ConnectorCategory(id="data", display_name="Data", sort_order=1)]),
    )
    _ensure_unknown_provider(monkeypatch)
    _patch_verify(monkeypatch, VerifyResult(ok=True, tools=[]))
    bind_app(None)  # tai_app.backend is None
    desc = _community_descriptor()
    tools = await add_provider(desc, source_url="https://x", added_by="me", config_values={})
    assert tools == []


async def test_add_provider_insert_conflict_raises(fake_pg, monkeypatch, bind_app):
    monkeypatch.setattr(
        catalog_write,
        "fetch_categories",
        _async_return([ConnectorCategory(id="data", display_name="Data", sort_order=1)]),
    )
    _ensure_unknown_provider(monkeypatch)
    _patch_verify(monkeypatch, VerifyResult(ok=True, tools=[]))
    bind_app(None)
    fake_pg.provider_conflict = True
    desc = _community_descriptor()
    with pytest.raises(ValueError, match="already exists"):
        await add_provider(desc, source_url="https://x", added_by="me", config_values={})


# -- helpers -----------------------------------------------------------------


def _async_return(value):
    async def _fn():
        return value

    return _fn


def _ensure_unknown_provider(monkeypatch):
    def _raise(pid):
        raise KeyError(pid)

    monkeypatch.setattr(catalog_write, "get_provider", _raise)
