"""Op-level oracles for the backup operations — the document-content validation
branches the route round-trips do not reach (they always carry a well-formed
document)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError
from tai_skeleton.operations.backup import import_backup


async def test_import_rejects_wrong_version() -> None:
    with pytest.raises(BadRequestError, match="unsupported backup document version"):
        await import_backup({"version": 2, "sections": {}}, ["manifest"])


async def test_import_rejects_non_object_sections() -> None:
    # A well-formed envelope whose document carries a non-object ``sections`` is a
    # loud 400 before any section import runs.
    with pytest.raises(BadRequestError, match="document must contain a 'sections' object"):
        await import_backup({"version": 1, "sections": "not-a-dict"}, ["manifest"])


async def test_import_registered_section_absent_from_document_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A selected section this host DOES register, but which the document omits, is a
    # per-section report error (ok=False) — not a transport failure.
    backup = SimpleNamespace(sections=lambda: [SimpleNamespace(name="manifest", secret=False)])
    monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(backup=backup))

    result = await import_backup({"version": 1, "sections": {}}, ["manifest"])

    assert result["ok"] is False
    assert "not present in the backup document" in result["sections"]["manifest"]["errors"][0]
