"""Connector-managed manifest entries: in-place add / remove mutators.

``add_managed_entries`` / ``remove_managed_entries`` edit a PRESERVED manifest
document in place and return the titles they touched. The read-modify-write
transaction, validation, reload, and fleet broadcast belong to
:class:`~tai_skeleton.config.service.ConfigService`, exercised in
``tests/config/test_service.py``; here we assert only the mutation semantics on a
plain document.
"""

from __future__ import annotations

from typing import Any

import pytest

from tai_skeleton.connectors.service.manifest_writer import (
    add_managed_entries,
    managed_title,
    remove_managed_entries,
)

from .conftest import CID, CID2, make_oauth_descriptor


def test_managed_title():
    assert managed_title("acme", "mail", "work") == "acme_mail_work"


def test_add_managed_entries_http():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    added = add_managed_entries(
        doc,
        descriptor=desc,
        enabled_sub_services=["mail", "cal"],
        alias="work",
        connection_id=CID,
    )
    assert added == ["acme_mail_work", "acme_cal_work"]
    titles = [e["title"] for e in doc["mcp"]]
    assert "acme_mail_work" in titles
    # http entry carries no auth header (injected at call time)
    mail = next(e for e in doc["mcp"] if e["title"] == "acme_mail_work")
    assert mail["config"]["type"] == "http"
    assert mail["managed"]["connection_id"] == CID


def test_add_managed_entries_stdio():
    from .conftest import make_noauth_stdio_descriptor

    desc = make_noauth_stdio_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(
        doc,
        descriptor=desc,
        enabled_sub_services=["search"],
        alias="main",
        connection_id=CID,
    )
    entry = doc["mcp"][0]
    assert entry["config"]["type"] == "stdio"
    assert entry["config"]["command"] == "uvx"


def test_add_managed_entries_appends_without_dropping_existing():
    """The mutator edits the passed document in place and preserves the entries
    (managed or hand-authored) already present."""
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": [{"title": "hand_authored", "config": {"type": "http", "url": "http://x"}}]}
    add_managed_entries(
        doc,
        descriptor=desc,
        enabled_sub_services=["mail"],
        alias="work",
        connection_id=CID,
    )
    titles = [e["title"] for e in doc["mcp"]]
    assert titles == ["hand_authored", "acme_mail_work"]


def test_add_managed_entries_missing_mcp_key():
    """A document with no ``mcp`` key yet gets one seeded with the managed entry."""
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {}
    added = add_managed_entries(
        doc,
        descriptor=desc,
        enabled_sub_services=["mail"],
        alias="work",
        connection_id=CID,
    )
    assert added == ["acme_mail_work"]
    assert [e["title"] for e in doc["mcp"]] == ["acme_mail_work"]


def test_add_managed_entries_idempotent():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail"], alias="work", connection_id=CID)
    second = add_managed_entries(
        doc,
        descriptor=desc,
        enabled_sub_services=["mail"],
        alias="work",
        connection_id=CID,
    )
    assert second == []  # already owned, left in place
    assert len(doc["mcp"]) == 1


def test_add_managed_entries_title_collision_raises():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail"], alias="work", connection_id=CID)
    with pytest.raises(ValueError, match="title collision"):
        add_managed_entries(
            doc,
            descriptor=desc,
            enabled_sub_services=["mail"],
            alias="work",
            connection_id=CID2,
        )


def test_remove_all_managed_entries():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail", "cal"], alias="work", connection_id=CID)
    removed = remove_managed_entries(doc, connection_id=CID)
    assert set(removed) == {"acme_mail_work", "acme_cal_work"}
    assert doc["mcp"] == []


def test_remove_subset_of_managed_entries():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail", "cal"], alias="work", connection_id=CID)
    removed = remove_managed_entries(doc, connection_id=CID, sub_services={"cal"})
    assert removed == ["acme_cal_work"]
    assert [e["title"] for e in doc["mcp"]] == ["acme_mail_work"]


def test_remove_leaves_other_connections_untouched():
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": []}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail"], alias="work", connection_id=CID)
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["cal"], alias="home", connection_id=CID2)
    remove_managed_entries(doc, connection_id=CID)
    titles = [e["title"] for e in doc["mcp"]]
    assert titles == ["acme_cal_home"]


def test_remove_leaves_hand_authored_entries_untouched():
    """A hand-authored entry (``managed is None``) is never removed."""
    desc = make_oauth_descriptor()
    doc: dict[str, Any] = {"mcp": [{"title": "hand_authored", "config": {"type": "http", "url": "http://x"}}]}
    add_managed_entries(doc, descriptor=desc, enabled_sub_services=["mail"], alias="work", connection_id=CID)
    remove_managed_entries(doc, connection_id=CID)
    assert [e["title"] for e in doc["mcp"]] == ["hand_authored"]
