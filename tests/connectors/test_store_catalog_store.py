"""Catalog loader: fetch_categories / fetch_catalog validation / refresh_catalog."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import tai_skeleton.connectors.store.catalog_store as catalog_store
from tai_skeleton.connectors.providers import registry
from tai_skeleton.connectors.store.catalog_store import (
    ConnectorCategory,
    fetch_catalog,
    fetch_categories,
    refresh_catalog,
)

from .conftest import make_noauth_http_descriptor


@pytest.fixture(autouse=True)
def _isolate_registry():
    cat = dict(registry._CATALOG_CACHE)
    reg = dict(registry._REGISTRY)
    registry._CATALOG_CACHE.clear()
    yield
    registry._CATALOG_CACHE.clear()
    registry._CATALOG_CACHE.update(cat)
    registry._REGISTRY.clear()
    registry._REGISTRY.update(reg)


class _FakeCursor:
    def __init__(self, pg: _FakePg) -> None:
        self._pg = pg
        self._rows: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._pg.executed.append(norm)
        if "FROM connector_category" in norm:
            self._rows = self._pg.categories
        elif "FROM connector_catalog" in norm:
            self._rows = self._pg.catalog
        else:
            self._rows = []

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
        return _FakeCursor(self._pg)


class _FakePg:
    def __init__(self, *, categories=None, catalog=None) -> None:
        self.categories = categories or []
        self.catalog = catalog or []
        self.executed: list[str] = []

    def connection(self):
        return _FakeConn(self)


@pytest.fixture
def install_pg(monkeypatch):
    def _install(*, categories=None, catalog=None):
        pg = _FakePg(categories=categories, catalog=catalog)

        @asynccontextmanager
        async def fake_client_ctx(client_cls, settings=None, **kwargs):
            yield pg

        monkeypatch.setattr(catalog_store, "client_ctx", fake_client_ctx)
        return pg

    return _install


def _catalog_descriptor_json(provider_id="catprov"):
    desc = make_noauth_http_descriptor(provider_id=provider_id)
    return desc.model_dump(mode="json", exclude={"origin", "category"})


# -- fetch_categories --------------------------------------------------------


async def test_fetch_categories(install_pg):
    pg = install_pg(categories=[("data", "Data", 1), ("other", "Other", 99)])
    cats = await fetch_categories()
    assert cats == [
        ConnectorCategory(id="data", display_name="Data", sort_order=1),
        ConnectorCategory(id="other", display_name="Other", sort_order=99),
    ]
    # Categories are read in display order.
    assert any("ORDER BY sort_order, id" in norm for norm in pg.executed)


# -- fetch_catalog -----------------------------------------------------------


async def test_fetch_catalog_round_trip(install_pg):
    pg = install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", _catalog_descriptor_json(), "system", "data", None)],
    )
    descriptors = await fetch_catalog()
    assert len(descriptors) == 1
    assert descriptors[0].id == "catprov"
    assert descriptors[0].origin == "system"
    assert descriptors[0].category == "data"
    # Only enabled catalog rows are loaded, ordered by provider_id.
    catalog_sql = next(norm for norm in pg.executed if "FROM connector_catalog" in norm)
    assert "WHERE enabled" in catalog_sql
    assert "ORDER BY provider_id" in catalog_sql


async def test_fetch_catalog_rejects_embedded_origin(install_pg):
    bad = _catalog_descriptor_json()
    bad["origin"] = "system"
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", bad, "system", "data", None)],
    )
    with pytest.raises(ValueError, match="must not embed"):
        await fetch_catalog()


async def test_fetch_catalog_rejects_unknown_category(install_pg):
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", _catalog_descriptor_json(), "system", "nope", None)],
    )
    with pytest.raises(ValueError, match="unknown category"):
        await fetch_catalog()


async def test_fetch_catalog_community_without_added_by(install_pg):
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", _catalog_descriptor_json(), "community", "data", None)],
    )
    with pytest.raises(ValueError, match="no added_by"):
        await fetch_catalog()


async def test_fetch_catalog_invalid_descriptor(install_pg):
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", {"id": "catprov"}, "system", "data", None)],
    )
    with pytest.raises(ValueError, match="invalid descriptor"):
        await fetch_catalog()


async def test_fetch_catalog_rejects_oauth_kind(install_pg):
    # A descriptor that is valid but kind != none.
    from .conftest import make_oauth_descriptor

    oauth_json = make_oauth_descriptor(provider_id="catprov").model_dump(mode="json", exclude={"origin", "category"})
    install_pg(
        categories=[("productivity", "Productivity", 1)],
        catalog=[("catprov", oauth_json, "system", "productivity", None)],
    )
    with pytest.raises(ValueError, match="kind='none'"):
        await fetch_catalog()


async def test_fetch_catalog_id_mismatch(install_pg):
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("different_id", _catalog_descriptor_json("catprov"), "system", "data", None)],
    )
    with pytest.raises(ValueError, match="does not match descriptor id"):
        await fetch_catalog()


# -- refresh_catalog ---------------------------------------------------------


async def test_refresh_catalog_publishes_to_registry(install_pg):
    install_pg(
        categories=[("data", "Data", 1)],
        catalog=[("catprov", _catalog_descriptor_json(), "system", "data", None)],
    )
    await refresh_catalog()
    assert registry.get_provider("catprov").id == "catprov"
