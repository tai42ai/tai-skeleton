"""``tai db`` — apply and inspect the database schema.

``apply`` runs the bundled DDL (``sql/schema.py:load_ddl``) through the kit
``PostgresClient``; the DDL is written ``IF NOT EXISTS`` / ``ON CONFLICT DO
NOTHING`` throughout, so re-running it is a no-op. ``check`` verifies
connectivity plus that every table the DDL declares exists, reporting the missing
ones and exiting non-zero for CI / pre-deploy.

Both commands connect through the ``TAI_DB_*`` schema-admin connection — a
dedicated namespace for the migrator, which a deployment points at the same
Postgres its runtime stores (``VERSIONING_STORE_*`` / ``CONNECTOR_STORE_*``) use.
Keeping the migrator's credentials separate lets it run as a schema-owning role
distinct from the app's runtime role.
"""

from __future__ import annotations

import asyncio
import re
from typing import LiteralString, cast

import typer
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import PostgresConnectionSettings, client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient
from tai42_kit.settings import settings_cache

from tai42_skeleton.sql.schema import load_ddl

app = typer.Typer(
    name="db",
    help="Apply and inspect the database schema.",
    no_args_is_help=True,
)

# Table names the bundled DDL declares. Parsed from the DDL itself so the check
# never drifts from what ``apply`` creates.
_CREATE_TABLE_RE = re.compile(r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)", re.IGNORECASE)


class SchemaAdminSettings(PostgresConnectionSettings):
    """``TAI_DB_*`` Postgres connection for the schema migrator. No baked-in
    credential — supply the password via ``TAI_DB_PG_PASSWORD``."""

    model_config = SettingsConfigDict(env_prefix="TAI_DB_")

    pg_db: str = "tai"


@settings_cache
def schema_settings() -> SchemaAdminSettings:
    return SchemaAdminSettings()


def expected_tables() -> list[str]:
    """The tables the bundled DDL creates, in sorted order."""
    return sorted(set(_CREATE_TABLE_RE.findall(load_ddl())))


def _target(settings: SchemaAdminSettings) -> str:
    """A credential-free description of the connection target for messages."""
    return f"{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"


async def _apply_schema(settings: SchemaAdminSettings) -> None:
    ddl = load_ddl()
    # The DDL is a single multi-statement script with no parameters, so it runs in
    # one execute; the transaction makes the whole thing commit or roll back
    # together.
    async with (
        client_ctx(PostgresClient, settings, fresh=True) as pool,
        pool.connection() as conn,
        conn.transaction(),
    ):
        # ``load_ddl`` returns the trusted, packaged schema (never user input), so
        # it is a valid literal query; the cast satisfies psycopg's LiteralString
        # guard, which exists to catch injected dynamic SQL.
        await conn.execute(cast(LiteralString, ddl))


async def _missing_tables(settings: SchemaAdminSettings) -> list[str]:
    expected = expected_tables()
    async with (
        client_ctx(PostgresClient, settings, fresh=True) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (expected,),
        )
        present = {row[0] for row in await cur.fetchall()}
    return [table for table in expected if table not in present]


@app.command("apply")
def apply_command() -> None:
    """Apply the bundled DDL to Postgres. Idempotent; loud on a connection failure."""
    import psycopg

    settings = schema_settings()
    try:
        asyncio.run(_apply_schema(settings))
    except psycopg.OperationalError as exc:
        # ``exc`` carries host/user diagnostics but never the password (it is not
        # part of the driver's error text); the DSN itself is never echoed.
        typer.echo(f"Error: could not connect to Postgres at {_target(settings)}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Applied schema to {_target(settings)}.")


@app.command("check")
def check_command() -> None:
    """Verify DB connectivity and that every expected framework table exists."""
    import psycopg

    settings = schema_settings()
    try:
        missing = asyncio.run(_missing_tables(settings))
    except psycopg.OperationalError as exc:
        typer.echo(f"Error: could not connect to Postgres at {_target(settings)}: {exc}", err=True)
        raise typer.Exit(1) from exc

    if missing:
        typer.echo(f"Error: schema at {_target(settings)} is missing tables: {', '.join(missing)}", err=True)
        typer.echo("Run 'tai db apply' to create them.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Schema at {_target(settings)} is present ({len(expected_tables())} tables).")
