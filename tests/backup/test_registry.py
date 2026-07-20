"""The concrete ``BackupRegistry`` — the impl body behind the ``app.backup`` facet.

Pins the registry contract directly (no app, no HTTP): registration order,
the duplicate-name guard, the unknown-name raises, and that a sync section's
exporter/importer run through verbatim.
"""

from __future__ import annotations

import pytest
from tai42_contract.backup import BackupSectionInfo

from tai42_skeleton.backup.registry import BackupRegistry


def test_sections_reports_registration_order_and_secret_flag():
    registry = BackupRegistry()
    registry.register_section("alpha", lambda: 1, lambda _p: {}, secret=True)
    registry.register_section("beta", lambda: 2, lambda _p: {})

    assert registry.sections() == [
        BackupSectionInfo(name="alpha", secret=True),
        BackupSectionInfo(name="beta", secret=False),
    ]


def test_register_duplicate_name_raises():
    registry = BackupRegistry()
    registry.register_section("dup", lambda: 1, lambda _p: {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register_section("dup", lambda: 2, lambda _p: {})


def test_export_section_runs_exporter():
    registry = BackupRegistry()
    registry.register_section("s", lambda: {"value": 42}, lambda _p: {})
    assert registry.export_section("s") == {"value": 42}


def test_import_section_runs_importer_with_payload():
    registry = BackupRegistry()
    seen: dict = {}

    def _importer(payload):
        seen["payload"] = payload
        return {"created": 1}

    registry.register_section("s", lambda: None, _importer)
    assert registry.import_section("s", {"a": 1}) == {"created": 1}
    assert seen["payload"] == {"a": 1}


def test_export_unknown_section_raises():
    registry = BackupRegistry()
    with pytest.raises(KeyError, match="unknown backup section"):
        registry.export_section("nope")


def test_import_unknown_section_raises():
    registry = BackupRegistry()
    with pytest.raises(KeyError, match="unknown backup section"):
        registry.import_section("nope", {})
