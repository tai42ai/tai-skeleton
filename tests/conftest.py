"""Suite-wide test infrastructure.

The connector catalog refresh is an app startup handler that runs whenever
connectors are in use (managed manifest entries, registered providers, or
``CONNECTORS_*``/``CONNECTOR_STORE_*`` env vars — which the connectors suite
sets), so a test entering ``app.app_context`` would otherwise open a real
Postgres pool for the ``connector_category`` / ``connector_catalog`` reads and
hang on the connection timeout. This suite is fully offline, so the autouse
fixture below injects a fake pooled Postgres client at the catalog store's
``client_ctx`` seam: the real ``refresh_catalog`` wiring runs, reads an empty
catalog, and returns instantly.

Tests that exercise the catalog DB behavior itself patch their own client seam
(e.g. the connectors token-store tests patch ``redis_pg.client_ctx``); this
fixture only covers the otherwise-incidental startup load.
"""

from __future__ import annotations

import os
import tempfile

# ``prometheus_client`` freezes its value backend (the multiprocess mmap class vs
# the in-process mutex class) the first time it is imported, choosing mmap only
# when ``PROMETHEUS_MULTIPROC_DIR`` is already set. The writer entry points assert
# the mmap backend is active, mirroring production where the launcher sets this
# before any import. Set it here — the earliest point in the test process, before
# any test module imports ``prometheus_client`` — so the suite runs mmap-frozen.
# No test writes counters to this scratch dir; render tests point the collector at
# their own tmp dirs.
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", os.path.join(tempfile.gettempdir(), "tai_prometheus_test"))

import logging
from contextlib import asynccontextmanager

import pytest
from tai_kit.settings import reset_all_settings

import tai_skeleton.connectors.store.catalog_store as catalog_store
from tests._fakes.interactions_redis import FakeRedis


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def fake_client_ctx(fake_redis: FakeRedis):
    """A drop-in for ``tai_kit.clients.client_ctx`` that yields the shared fake
    for any client class, ignoring pool/fresh."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake_redis

    return _ctx


class _FakeCursor:
    """Records nothing and returns no rows — the offline catalog is empty."""

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, *args, **kwargs) -> None:
        return None

    async def fetchall(self) -> list:
        return []


class _FakeConn:
    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor()


class _FakePool:
    def connection(self) -> _FakeConn:
        return _FakeConn()


@pytest.fixture(autouse=True)
def _offline_connector_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield _FakePool()

    monkeypatch.setattr(catalog_store, "client_ctx", fake_client_ctx)


class _ProbeRedis:
    """A plain-Redis stand-in: ``HGETALL`` on the probe key answers ``{}``, so the
    identity provider's ``healthcheck()`` passes without a real Redis."""

    async def hgetall(self, key: str) -> dict[str, str]:
        return {}


@pytest.fixture(autouse=True)
def _ensure_redis_identity_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test-side sane default for the extracted identity provider.

    The skeleton ships NO concrete identity provider — a deployment names one in its
    manifest ``lifecycle_modules``, which ``start()`` imports. Tests that don't boot a
    full app (adapter / router / management coverage) still resolve the default
    ``"redis"`` provider through the module-level registry. An app-boot test's
    ``start()`` calls ``reset_registry()`` and then re-registers from the manifest, so
    a minimal test manifest that omits the identity plugin would leave the registry
    empty — and the AC-enabled startup probe (``probe_identity_provider``) would fail
    to resolve the active provider. Emulate a manifest that lists the plugin: register
    ``"redis"`` before each test, AND wrap ``start()``'s registry reset so the default
    is re-registered after the clear (the reset still runs, so a manifest re-import's
    duplicate guard is unaffected). Suites that isolate the registry snapshot this
    baseline and restore it."""
    from tai_contract.access_control import registry
    from tai_identity_redis.redis_api_key_provider import RedisApiKeyProvider

    import tai_skeleton.app.lifecycle as lifecycle

    def _ensure() -> None:
        if "redis" not in registry._REGISTRY:
            registry.register_identity_provider("redis", RedisApiKeyProvider)

    real_reset = lifecycle.reset_identity_registry

    def _reset_then_ensure() -> None:
        real_reset()
        _ensure()

    monkeypatch.setattr(lifecycle, "reset_identity_registry", _reset_then_ensure)
    _ensure()


@pytest.fixture(autouse=True)
def _identity_probe_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The access-control startup probe runs whenever access control is enabled (the
    default). This offline suite has no real Redis, so point the provider's client
    seam at a plain-Redis fake — the real probe wiring runs and passes.
    ``probe_identity_provider`` awaits the active provider's ``healthcheck()``, which
    reaches Redis through the plugin's own ``client_ctx``, so patch that seam. The
    probe's own tests re-patch it to drive the failure branches."""
    import tai_identity_redis.redis_api_key_provider as redis_provider

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield _ProbeRedis()

    monkeypatch.setattr(redis_provider, "client_ctx", fake_client_ctx)


@pytest.fixture
def root_logger_restored():
    """Snapshot the root logger and restore it afterwards. Code under test may run
    ``setup_logging`` / ``apply_logging_settings`` (both ``force=True``), which
    replace the root logger's handlers and level; the restore keeps a failing
    assertion from leaking that mutation into later tests. The teardown also drops
    the settings caches, so a test's monkeypatched ``TAI_LOG_LEVEL`` never survives
    into a later test's ``logging_settings()`` read."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield root
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        reset_all_settings()
