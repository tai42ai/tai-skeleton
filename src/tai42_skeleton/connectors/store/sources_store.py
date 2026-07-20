"""Read-only loader for the MCP-finder's allowed discovery sources.

The ``connector_allowed_source`` table holds one row per base URL the discovery
tools may search/fetch. It is runtime read-only: ops extend the list by editing
the table directly in Postgres — no tool, route, or UI writes it. ``enabled``
disables a source without deleting it; :func:`fetch_allowed_sources` returns
enabled rows only.

Postgres is reached through the app-pooled ``PostgresClient`` so it shares one
pool with the token / catalog stores (same DSN), closed centrally at shutdown.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient

from tai42_skeleton.connectors.settings import connector_store_settings

logger = logging.getLogger(__name__)


class AllowedSource(BaseModel):
    """One enabled ``connector_allowed_source`` row — a searchable base URL."""

    id: str
    url: str


async def fetch_allowed_sources() -> list[AllowedSource]:
    """Read every enabled allowed-source row, ordered by id."""
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT id, url FROM connector_allowed_source WHERE enabled ORDER BY id")
        rows = await cur.fetchall()
    return [AllowedSource(id=source_id, url=url) for source_id, url in rows]
