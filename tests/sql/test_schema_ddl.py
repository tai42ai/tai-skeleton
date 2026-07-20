"""Schema-resource tests for the de-tenanted connector token store.

Loads the bundled DDL via the package loader and asserts the single-namespace
shape required by the de-tenanted connector contract: the token-store table is
keyed by `connection_id` alone, with no tenant / client partition column.
"""

import re

from tai_skeleton.sql.schema import load_ddl


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
