"""The ``/ready`` readiness route.

The route pings exactly the backing stores this deployment wired, deduped by
connection identity and pinged concurrently. These tests fake ``client_ctx`` so no
live Redis/Postgres is needed, and drive ``_wired_connections`` directly (or via a
stub) to isolate the response/dedupe/aggregation behavior from the real gates.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import cast

from starlette.requests import Request
from tai_contract.access_control import registry
from tai_contract.access_control.identity import AuthIdentity, IdentityProvider, ReadinessTarget
from tai_kit.clients import PostgresConnectionSettings, RedisConnectionSettings
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient
from tai_kit.settings import reset_all_settings

from tai_skeleton.routers import health


class _RedisBoom(Exception):
    """Distinct exception type so the test can assert the type name surfaces."""


class _FakeRedis:
    def __init__(self, fail_type: type[Exception] | None) -> None:
        self._fail_type = fail_type

    async def ping(self) -> None:
        if self._fail_type is not None:
            raise self._fail_type("secret-redis-host:6379 unreachable")


class _FakeConn:
    def __init__(self, fail_type: type[Exception] | None) -> None:
        self._fail_type = fail_type

    async def execute(self, sql: str) -> None:
        if self._fail_type is not None:
            raise self._fail_type("secret-pg-host:5432 unreachable")


class _FakePool:
    def __init__(self, fail_type: type[Exception] | None) -> None:
        self._fail_type = fail_type

    def connection(self):
        conn = _FakeConn(self._fail_type)

        @asynccontextmanager
        async def _cm():
            yield conn

        return _cm()


def _make_client_ctx(calls: list, fail_idents: frozenset = frozenset(), fail_type: type[Exception] = RuntimeError):
    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings):
        kwargs = settings.client_kwargs()
        ident = kwargs.get("url") or kwargs.get("dsn")
        calls.append((client_cls.__name__, ident))
        boom = fail_type if ident in fail_idents else None
        if client_cls is RedisClient:
            yield _FakeRedis(boom)
        else:
            yield _FakePool(boom)

    return fake_client_ctx


def _request() -> Request:
    # The handler ignores the request; a bare stand-in cast to Request suffices.
    return cast(Request, object())


async def test_ready_all_healthy_returns_200(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(health, "client_ctx", _make_client_ctx(calls))
    wired = [
        ("access_control", RedisClient, RedisConnectionSettings(redis_url="redis://ac")),
        ("tool_runs", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
        ("interactions", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
        ("versioning", PostgresClient, PostgresConnectionSettings(pg_host="db")),
    ]
    monkeypatch.setattr(health, "_wired_connections", lambda: wired)

    resp = await health.readiness_check(_request())

    assert resp.status_code == 200
    body = json.loads(bytes(resp.body))
    assert body["status"] == "ready"
    assert body["checks"] == {
        "access_control": "ok",
        "tool_runs": "ok",
        "interactions": "ok",
        "versioning": "ok",
    }


async def test_ready_failure_returns_503_type_only(monkeypatch, caplog) -> None:
    calls: list = []
    monkeypatch.setattr(
        health,
        "client_ctx",
        _make_client_ctx(calls, fail_idents=frozenset({"redis://shared"}), fail_type=_RedisBoom),
    )
    wired = [
        ("access_control", RedisClient, RedisConnectionSettings(redis_url="redis://ac")),
        ("tool_runs", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
        ("interactions", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
    ]
    monkeypatch.setattr(health, "_wired_connections", lambda: wired)

    with caplog.at_level("WARNING", logger=health.logger.name):
        resp = await health.readiness_check(_request())

    assert resp.status_code == 503
    raw = bytes(resp.body)
    body = json.loads(raw)
    assert body["status"] == "not_ready"
    assert body["checks"]["access_control"] == "ok"
    # Both subsystems sharing the failed connection fail together, carrying only
    # the exception TYPE name.
    assert body["checks"]["tool_runs"] == "_RedisBoom"
    assert body["checks"]["interactions"] == "_RedisBoom"
    # The exception MESSAGE never reaches the public body...
    assert b"secret-redis-host" not in raw
    # ...but the full detail is in the logs, exactly one warning for the one failed
    # distinct connection.
    assert "secret-redis-host" in caplog.text
    assert sum("readiness ping failed" in rec.getMessage() for rec in caplog.records) == 1


async def test_ready_dedupes_shared_connection(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(health, "client_ctx", _make_client_ctx(calls))
    wired = [
        ("tool_runs", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
        ("interactions", RedisClient, RedisConnectionSettings(redis_url="redis://shared")),
    ]
    monkeypatch.setattr(health, "_wired_connections", lambda: wired)

    resp = await health.readiness_check(_request())

    assert resp.status_code == 200
    # One ping for the single shared connection, though two subsystems use it.
    assert len(calls) == 1
    body = json.loads(bytes(resp.body))
    assert body["checks"] == {"tool_runs": "ok", "interactions": "ok"}


async def test_wired_connections_gates_out_pg_and_inmemory_hooks(monkeypatch) -> None:
    # connectors + versioning not wired, hooks in-memory: no Postgres check and no
    # hooks check are produced.
    monkeypatch.setattr(health.instance, "connectors_in_use", lambda: False)
    monkeypatch.setattr(health.instance, "versioned_store_in_use", lambda: False)
    for key in list(os.environ):
        if key.startswith("HOOKS_"):
            monkeypatch.delenv(key, raising=False)

    conns = health._wired_connections()

    names = [name for name, _, _ in conns]
    classes = [cls for _, cls, _ in conns]
    assert "connectors" not in names
    assert "versioning" not in names
    assert "hooks" not in names
    assert PostgresClient not in classes


async def test_wired_connections_gates_in_stores_when_wired(monkeypatch) -> None:
    monkeypatch.setattr(health.instance, "connectors_in_use", lambda: True)
    monkeypatch.setattr(health.instance, "versioned_store_in_use", lambda: True)
    monkeypatch.setenv("HOOKS_REDIS_URL", "redis://hooks")
    reset_all_settings()
    try:
        conns = health._wired_connections()
    finally:
        reset_all_settings()

    names = [name for name, _, _ in conns]
    classes = [cls for _, cls, _ in conns]
    assert "connectors" in names
    assert "versioning" in names
    assert "hooks" in names
    # connectors contributes a Postgres connection; so does versioning.
    assert PostgresClient in classes


async def test_wired_connections_gates_sub_mcp_on_redis_url(monkeypatch) -> None:
    # The durable sub-MCP registration store joins the readiness set exactly when
    # SUB_MCP_REDIS_URL is set (same gate shape as hooks): a ``sub_mcp`` Redis check
    # appears with it set, and no ``sub_mcp`` check appears with it unset.
    monkeypatch.setattr(health.instance, "connectors_in_use", lambda: False)
    monkeypatch.setattr(health.instance, "versioned_store_in_use", lambda: False)
    for key in list(os.environ):
        if key.startswith("SUB_MCP_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("SUB_MCP_REDIS_URL", "redis://sub-mcp")
    reset_all_settings()
    try:
        wired = health._wired_connections()
    finally:
        reset_all_settings()
    sub_mcp = [(name, cls) for name, cls, _ in wired if name == "sub_mcp"]
    assert sub_mcp == [("sub_mcp", RedisClient)]

    monkeypatch.delenv("SUB_MCP_REDIS_URL", raising=False)
    reset_all_settings()
    try:
        wired_unset = health._wired_connections()
    finally:
        reset_all_settings()
    assert "sub_mcp" not in [name for name, _, _ in wired_unset]


async def test_wired_connections_enumerates_identity_provider_generically(monkeypatch) -> None:
    # Core enumerates the ACTIVE identity provider's declared readiness target(s)
    # through the IdentityProvider ABC — it does NOT string-match the provider name
    # "redis". A provider registered under any OTHER name, declaring any store, is
    # health-checked all the same.
    declared = RedisConnectionSettings(redis_url="redis://custom-idp-store")

    class _CustomProvider(IdentityProvider):
        def __init__(self, settings: object) -> None:
            self._settings = settings

        async def validate_token(self, token: str) -> AuthIdentity | None:
            return None

        def readiness_targets(self) -> tuple[ReadinessTarget, ...]:
            return (ReadinessTarget("access_control", RedisClient, declared),)

    registry.register_identity_provider("custom_idp", _CustomProvider)
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["custom_idp"]')
    reset_all_settings()
    try:
        conns = health._wired_connections()
    finally:
        registry._REGISTRY.pop("custom_idp", None)
        reset_all_settings()

    # The provider registered under "custom_idp" (never "redis") still contributes its
    # declared target, verbatim — proving core routes through the ABC, not a name match.
    ac = [conn for conn in conns if conn[0] == "access_control"]
    assert ac == [("access_control", RedisClient, declared)]
    assert ac[0][2] is declared


async def test_ready_nothing_wired_returns_200_empty(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(health, "client_ctx", _make_client_ctx(calls))
    monkeypatch.setattr(health, "_wired_connections", list)

    resp = await health.readiness_check(_request())

    assert resp.status_code == 200
    body = json.loads(bytes(resp.body))
    assert body == {"status": "ready", "checks": {}}
    assert calls == []
