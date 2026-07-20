"""Validated, verified write path for community catalog rows.

The agent-tool add path is the ONLY runtime writer of ``connector_catalog``
(ops insert system rows directly in SQL). :func:`add_provider` gates every
insert: pydantic-valid no-auth descriptor, known (or explicitly created)
category, no collision with an existing provider, and a passing verbose
verification of every sub-service — each failure raises with the reason,
nothing is inserted on a reject. After the insert a full local reload runs (its
reload handlers re-run the catalog refresh) and the reload is broadcast on the
worker bus, so the new provider is visible on every process in the fleet.

:func:`create_category` is the category-create path of the same add flow; new
categories slot in before the ``other`` sentinel so ``other`` stays last.
"""

from __future__ import annotations

import logging
import re

from tai_contract.app import tai_app
from tai_contract.connectors.probe import ToolSummary
from tai_contract.connectors.providers import ProviderDescriptor
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import Json, PostgresClient

from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.connectors.providers.registry import get_provider
from tai_skeleton.connectors.runtime.probe import verify
from tai_skeleton.connectors.settings import connector_store_settings
from tai_skeleton.connectors.store.catalog_store import ConnectorCategory, fetch_categories
from tai_skeleton.operations._broadcast import broadcast

logger = logging.getLogger(__name__)

# connector_category ids are kebab-case slugs (e.g. "dev-tools").
_CATEGORY_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# Seed sentinel keeping "other" last; new categories must slot in below it.
_OTHER_CATEGORY_ID = "other"
# The seed row's sort_order (see sql/resources/tai_skeleton.init.sql); a new
# category reaching this value would sort at/after "other", which is an error.
_OTHER_SORT_ORDER = 1000

# Fixed key for the transaction-scoped advisory lock that serializes category
# creation. The MAX(sort_order)+1 read and its insert are two steps under READ
# COMMITTED, so without this lock two racing creates could read the same MAX and
# collide on sort_order. Category creation is a rare admin action, so serializing
# every create on one key is cheap; the lock releases at commit/rollback.
_CATEGORY_CREATE_ADVISORY_KEY = 0x7461695F636F6E6E  # "tai_conn"


def _validate_new_category(category_id: str, display_name: str) -> None:
    if not _CATEGORY_ID_RE.match(category_id):
        raise ValueError(f"category id {category_id!r} must be a kebab-case slug (^[a-z][a-z0-9-]*$)")
    if not display_name.strip():
        raise ValueError(f"category {category_id!r} display_name must be non-empty")


async def _insert_category(cur, category_id: str, display_name: str) -> ConnectorCategory:
    """Insert one category row (advisory-locked MAX+1) on an open cursor.

    The caller owns the transaction, so this can compose with the catalog insert
    in :func:`add_provider` (create + insert commit or roll back together)."""
    await cur.execute("SELECT pg_advisory_xact_lock(%s)", (_CATEGORY_CREATE_ADVISORY_KEY,))
    await cur.execute(
        "INSERT INTO connector_category (id, display_name, sort_order) "
        "SELECT %s, %s, COALESCE(MAX(sort_order), 0) + 1 "
        "FROM connector_category WHERE id <> %s "
        "ON CONFLICT (id) DO NOTHING "
        "RETURNING sort_order",
        (category_id, display_name, _OTHER_CATEGORY_ID),
    )
    row = await cur.fetchone()
    if row is None:
        raise ValueError(f"category {category_id!r} already exists")
    (sort_order,) = row
    if sort_order >= _OTHER_SORT_ORDER:
        # Raising rolls the insert back (the connection context commits only on
        # clean exit).
        raise ValueError(
            f"category sort_order {sort_order} reached the "
            f"{_OTHER_CATEGORY_ID!r} sentinel ({_OTHER_SORT_ORDER}); "
            "cannot keep it sorted last"
        )
    logger.info("connectors: created category %s", category_id)
    return ConnectorCategory(id=category_id, display_name=display_name, sort_order=sort_order)


async def create_category(category_id: str, display_name: str) -> ConnectorCategory:
    """Insert a new ``connector_category`` row and return it.

    The new category sorts after every existing non-``other`` category and
    before ``other``. Raises on a malformed id, an empty display name, or an
    id collision — never silently reuses an existing row.
    """
    _validate_new_category(category_id, display_name)

    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        return await _insert_category(cur, category_id, display_name)


async def add_provider(
    descriptor: ProviderDescriptor,
    *,
    source_url: str,
    added_by: str,
    config_values: dict[str, str],
    new_category_display_name: str | None = None,
) -> list[ToolSummary]:
    """Verify and insert a community provider, then propagate it fleet-wide.

    ``descriptor`` must already carry ``origin="community"`` and the target
    ``category``. When ``new_category_display_name`` is given the category is
    created in the SAME transaction as the catalog insert, AFTER verification, so
    a rejected add (verification failure, id collision) leaves nothing behind —
    both rows commit or roll back together. Otherwise the category must already
    exist. Every sub-service is verified with the client-supplied
    ``config_values`` and must answer — the first failure rejects the add.

    Returns the verified tool list (every sub-service's tools, in sub-service
    order) — the proof the provider answered.

    Raises ``ValueError`` on: oauth kind, non-community origin, empty
    ``added_by``, unknown category without a create payload, provider-id
    collision, failed verification, or a concurrent insert of the same id.
    """
    if descriptor.kind != "none":
        raise ValueError(f"community provider {descriptor.id!r} must have kind='none' (got {descriptor.kind!r})")
    if descriptor.origin != "community":
        raise ValueError(
            f"catalog add path only writes community rows; provider {descriptor.id!r} has origin {descriptor.origin!r}"
        )
    if not added_by.strip():
        raise ValueError(f"community provider {descriptor.id!r} requires added_by")

    # Validate the category BEFORE the network verification (cheap fail-fast),
    # but defer the actual category INSERT into the write transaction below so a
    # later reject never orphans a freshly-created category.
    if new_category_display_name is not None:
        _validate_new_category(descriptor.category, new_category_display_name)
    else:
        known_ids = {category.id for category in await fetch_categories()}
        if descriptor.category not in known_ids:
            raise ValueError(
                f"unknown category {descriptor.category!r}; pass a create "
                f"payload to add it (known: {', '.join(sorted(known_ids))})"
            )

    try:
        get_provider(descriptor.id)
    except KeyError:
        pass
    else:
        raise ValueError(f"provider {descriptor.id!r} already exists")

    tools: list[ToolSummary] = []
    for sub_service in descriptor.sub_services:
        result = await verify(descriptor, sub_service, config_values=config_values)
        if not result.ok:
            raise ValueError(
                f"verification failed for provider {descriptor.id!r} sub_service {sub_service!r}: {result.error}"
            )
        tools.extend(result.tools)

    # origin/category live only in their columns; the stored jsonb must not
    # embed them (fetch_catalog rejects rows that do).
    descriptor_json = descriptor.model_dump(mode="json", exclude={"origin", "category"})
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        # Create the new category (if any) and insert the provider row in one
        # transaction: a conflict on the insert rolls the category back too, so a
        # rejected add never leaves an orphan category behind.
        if new_category_display_name is not None:
            await _insert_category(cur, descriptor.category, new_category_display_name)
        await cur.execute(
            "INSERT INTO connector_catalog "
            "(provider_id, descriptor, origin, category, source_url, added_by) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (provider_id) DO NOTHING",
            (
                descriptor.id,
                Json(descriptor_json),
                descriptor.origin,
                descriptor.category,
                source_url,
                added_by,
            ),
        )
        if cur.rowcount == 0:
            raise ValueError(f"provider {descriptor.id!r} already exists")

    logger.info(
        "connectors: added community provider %s (category=%s, added_by=%s)",
        descriptor.id,
        descriptor.category,
        added_by,
    )

    # Local + fleet visibility: run the FULL local reload (its reload handlers re-run
    # refresh_catalog, so the new provider is in this worker's cache) then broadcast
    # the reload so every worker re-reads. The provider row is already committed, so a
    # failed local reload still broadcasts and re-raises with the fleet report — a
    # fleet never strands stale behind a local error.
    await broadcast(
        {"op": "reload_config"},
        None,
        lambda: reload_gate.run(tai_app.admin.reload_config),
        publish_on_local_failure=True,
    )

    return tools
