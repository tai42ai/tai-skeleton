"""``tai doctor`` — read-only environment health diagnostics."""

from __future__ import annotations

import asyncio
import platform
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import typer
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import PostgresClient
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.cli.commands._common import app_context
from tai_skeleton.cli.native.db import SchemaAdminSettings, expected_tables, schema_settings
from tai_skeleton.cli.render import print_json, render_table
from tai_skeleton.config.config_mode import config_mode

_OK = "ok"
_FAIL = "fail"
_INFO = "info"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _pg_target(settings: SchemaAdminSettings) -> str:
    """A credential-free description of the Postgres target."""
    return f"{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"


def _redact_url(url: str | None) -> str:
    """Mask the password in a connection URL, keeping the rest for diagnosis."""
    if not url:
        return "(unset)"
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    netloc = f"{user}:***@{host}" if user else f":***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


async def _probe_postgres(settings: SchemaAdminSettings) -> Check:
    import psycopg

    try:
        async with (
            client_ctx(PostgresClient, settings, fresh=True) as pool,
            pool.connection() as conn,
        ):
            await conn.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        return Check("postgres", _FAIL, f"cannot connect to {_pg_target(settings)}: {exc}")
    return Check("postgres", _OK, f"connected to {_pg_target(settings)}")


async def _probe_schema(settings: SchemaAdminSettings) -> Check:
    import psycopg

    expected = expected_tables()
    try:
        async with (
            client_ctx(PostgresClient, settings, fresh=True) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                (expected,),
            )
            present = {row[0] for row in await cur.fetchall()}
    except psycopg.OperationalError as exc:
        return Check("schema", _FAIL, f"cannot inspect schema at {_pg_target(settings)}: {exc}")
    missing = [table for table in expected if table not in present]
    if missing:
        return Check("schema", _FAIL, f"missing tables (run 'tai db apply'): {', '.join(missing)}")
    return Check("schema", _OK, f"all {len(expected)} framework tables present")


async def _probe_redis() -> Check:
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from tai_skeleton.connectors.settings import connector_store_settings

    redis_settings = connector_store_settings().redis
    target = _redact_url(redis_settings.redis_url)
    try:
        async with client_ctx(RedisClient, redis_settings, fresh=True) as client:
            # redis-py types ``ping`` against its sync client (-> bool); the async
            # client returns the awaitable it is called on here.
            await cast("Awaitable[bool]", client.ping())
    except (RedisConnectionError, RedisTimeoutError, OSError) as exc:
        return Check("redis", _FAIL, f"cannot connect to {target}: {exc}")
    return Check("redis", _OK, f"connected to {target}")


async def _run_checks() -> list[Check]:
    settings = schema_settings()
    checks = [
        Check("python", _INFO, platform.python_version()),
        Check("config-mode", _INFO, config_mode()),
    ]
    postgres = await _probe_postgres(settings)
    checks.append(postgres)
    # The schema probe needs a live connection; skip it (rather than double-report
    # the connection failure) when Postgres itself is unreachable.
    if postgres.status == _OK:
        checks.append(await _probe_schema(settings))
    checks.append(await _probe_redis())
    return checks


def doctor(ctx: typer.Context) -> None:
    """Run read-only health diagnostics against the environment.

    Probes Python version, config mode, Postgres and Redis connectivity, and
    whether the schema is applied. Purely read-only. Credentials are redacted:
    connection-URL passwords are masked and no DSN is echoed. Exits non-zero when
    any dependency check fails.
    """
    app_ctx = app_context(ctx)
    checks = asyncio.run(_run_checks())
    records = [{"check": c.name, "status": c.status, "detail": c.detail} for c in checks]

    if app_ctx.json_output:
        print_json(records)
    else:
        typer.echo(render_table(records, ["check", "status", "detail"]))

    if any(c.status == _FAIL for c in checks):
        raise typer.Exit(1)
