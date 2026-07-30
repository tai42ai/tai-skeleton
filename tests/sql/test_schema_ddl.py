"""Schema-resource tests for the de-tenanted connector token store.

Loads the bundled DDL via the package loader and asserts the single-namespace
shape required by the de-tenanted connector contract: the token-store table is
keyed by `connection_id` alone, with no tenant / client partition column.
"""

import re

from tai42_skeleton.sql.schema import load_ddl


def _connector_connections_block(ddl: str) -> str:
    """Return the text of the `connector_connections` CREATE TABLE (...) body."""
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS connector_connections\s*\((.*?)\);",
        ddl,
        re.DOTALL,
    )
    assert match is not None, "connector_connections table not found in DDL"
    return match.group(1)


def test_load_ddl_returns_nonempty_schema() -> None:
    ddl = load_ddl()
    assert isinstance(ddl, str)
    assert "CREATE TABLE IF NOT EXISTS connector_connections" in ddl


def test_token_store_primary_key_collapses_to_connection_id() -> None:
    block = _connector_connections_block(load_ddl())
    # Single-column PK on connection_id; no composite (client_name, connection_id) PK — the store is de-tenanted.
    assert re.search(r"PRIMARY KEY\s*\(\s*connection_id\s*\)", block) is not None
    assert "PRIMARY KEY (client_name" not in block


def test_token_store_has_no_tenant_columns() -> None:
    block = _connector_connections_block(load_ddl())
    for forbidden in ("client_name", "tenant_id", "tenant"):
        assert forbidden not in block, f"de-tenant violated: `{forbidden}` present in connector_connections"


def test_header_is_de_branded() -> None:
    ddl = load_ddl()
    assert "Nexus Platform" not in ddl
    assert "Tai Platform" in ddl


def test_role_audit_append_only_triggers_present() -> None:
    """The shipped DDL wires the three role_audit append-only triggers onto the
    versioned-document tables — the DB-level guard that a comment alone cannot give.
    Text-level guard so removing/renaming a trigger fails without a live Postgres."""
    ddl = load_ddl()
    # Trigger functions (idempotent CREATE OR REPLACE) and their row triggers.
    for name in (
        "versioned_document_versions_role_audit_immutable",
        "versioned_documents_role_audit_no_delete",
        "versioned_documents_role_audit_guard_update",
    ):
        assert f"CREATE OR REPLACE FUNCTION {name}()" in ddl, f"missing trigger function {name}"
        assert f"DROP TRIGGER IF EXISTS trg_{name}" in ddl, f"missing idempotent drop for trg_{name}"
        assert f"CREATE TRIGGER trg_{name}" in ddl, f"missing trigger trg_{name}"

    # The immutability trigger fires on both UPDATE and DELETE of version rows;
    # the doc-delete guard on DELETE; the update guard on UPDATE.
    assert "BEFORE UPDATE OR DELETE ON versioned_document_versions" in ddl
    assert "BEFORE DELETE ON versioned_documents" in ddl
    assert "BEFORE UPDATE ON versioned_documents" in ddl
    # Every guard keys strictly on kind='role_audit' — no other kind is affected.
    assert ddl.count("'role_audit'") >= 3


def test_marketplace_installs_evolves_added_columns_via_alter() -> None:
    """`contract_version`/`skeleton_version` sit inside the `CREATE TABLE IF NOT
    EXISTS marketplace_installs` block — a no-op on any database where the table
    already exists — so the shipped DDL MUST also add them with idempotent
    `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for existing deployments. Text-level
    guard so dropping either ALTER (which would break the store's SELECT/INSERT on
    an already-created table) fails without a live Postgres."""
    ddl = load_ddl()
    for column in ("contract_version", "skeleton_version"):
        assert f"ALTER TABLE marketplace_installs ADD COLUMN IF NOT EXISTS {column} TEXT;" in ddl, (
            f"missing idempotent ADD COLUMN for marketplace_installs.{column}"
        )
    # The ALTERs come AFTER the CREATE TABLE so the table exists when they run.
    create_at = ddl.index("CREATE TABLE IF NOT EXISTS marketplace_installs")
    for column in ("contract_version", "skeleton_version"):
        alter_at = ddl.index(f"ALTER TABLE marketplace_installs ADD COLUMN IF NOT EXISTS {column} TEXT;")
        assert alter_at > create_at, f"ALTER for {column} must follow the CREATE TABLE"


def test_tool_meta_overlay_tables_present() -> None:
    """The tool-metadata overlay ships two plain tables: a folder tree and a
    per-tool row. Text-level guards so a dropped/renamed table or a lost
    invariant fails without a live Postgres."""
    ddl = load_ddl()
    assert "CREATE TABLE IF NOT EXISTS tool_folders" in ddl
    assert "CREATE TABLE IF NOT EXISTS tool_meta" in ddl


def test_tool_folders_root_name_uniqueness_uses_nulls_not_distinct() -> None:
    """Root folders have `parent_id IS NULL`; without `NULLS NOT DISTINCT` the
    default unique would never fire for them, silently allowing duplicate root
    names. The modifier is REQUIRED on the sibling-name uniqueness constraint."""
    match = re.search(r"CREATE TABLE IF NOT EXISTS tool_folders\s*\((.*?)\n\);", load_ddl(), re.DOTALL)
    assert match is not None, "tool_folders table not found in DDL"
    block = match.group(1)
    assert re.search(r"UNIQUE\s+NULLS NOT DISTINCT\s*\(\s*parent_id\s*,\s*name\s*\)", block) is not None


def test_tool_meta_hidden_is_nullable_tristate() -> None:
    """`tool_meta.hidden` is a NULLABLE boolean — the tri-state that lets NULL mean
    "defer to the plugin-declared visibility" while TRUE/FALSE force it. A
    NOT-NULL column could not express the defer state."""
    match = re.search(r"CREATE TABLE IF NOT EXISTS tool_meta\s*\((.*?)\n\);", load_ddl(), re.DOTALL)
    assert match is not None, "tool_meta table not found in DDL"
    block = match.group(1)
    hidden_line = next(line for line in block.splitlines() if line.strip().startswith("hidden"))
    assert "NOT NULL" not in hidden_line
