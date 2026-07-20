"""Backup/restore operations — list the sections and import a backup document.

Two operations over the live ``tai42_app.backup`` registry. The registry lists its
sections (host sections plus any a plugin registered), and each section runs its
own subsystem's export/import through a thin exporter/importer pair.

* ``list_sections`` returns the registered sections as ``{name, secret}`` so the
  UI renders one checkbox per live section (plugins included).
* ``import_backup`` imports each SELECTED section from a backup ``document`` and
  collects a per-section report. A document whose ``version`` is not 1 is a loud
  400; a selected section that is unknown (not registered) or absent from the
  document, or one whose importer fails, carries its error in the report and makes
  the overall result ``ok: false`` — none is a transport error, because the
  request itself is well-formed. The import op is destructive AND
  authority-changing (a restore can mint keys / replace policy), so it is off the
  default MCP surface (tier 2) and reload-gated (section importers rebind live
  registries).

The export door is a downloadable-attachment content route (the raw document, not
the ``{"data": ...}`` envelope), so it stays a handler in the router and reaches
these shared helpers from here.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import BaseModel
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation

logger = logging.getLogger(__name__)

_DOCUMENT_VERSION = 1


class BackupImport(BaseModel):
    """Import request — a backup ``document`` produced by the export route and the
    section names to import from it."""

    document: dict[str, Any]
    sections: list[str]


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when a section's exporter/importer was async; a sync
    section returns its result directly."""
    if inspect.isawaitable(value):
        return await value
    return value


def _registered_section_names() -> set[str]:
    return {info.name for info in tai42_app.backup.sections()}


@operation(summary="List backup sections", tags=["backup"])
async def list_sections() -> list:
    return [{"name": info.name, "secret": info.secret} for info in tai42_app.backup.sections()]


@operation(
    summary="Import a backup document",
    tags=["backup"],
    destructive=True,
    authority_changing=True,
    reload_gated=True,
    errors=[BadRequestError],
    request_model=BackupImport,
)
async def import_backup(document: dict[str, Any], sections: list[str]) -> dict:
    # The envelope shape (a JSON object carrying a ``document`` object + a list of
    # section-name strings) is validated at the HTTP edge by the route's extractor;
    # the document CONTENT (version, its sections map) is the operation's own
    # validation, so it declares and raises ``BadRequestError`` for those.
    if document.get("version") != _DOCUMENT_VERSION:
        raise BadRequestError(f"unsupported backup document version: {document.get('version')!r}")
    document_sections = document.get("sections")
    if not isinstance(document_sections, dict):
        raise BadRequestError("document must contain a 'sections' object")

    registered = _registered_section_names()
    reports: dict[str, Any] = {}
    ok = True
    for name in sections:
        if name not in registered:
            # A well-formed request naming a section this host does not register —
            # a section report error, not a transport failure.
            reports[name] = {"created": 0, "updated": 0, "skipped": 0, "errors": [f"unknown section: {name!r}"]}
            ok = False
            continue
        if name not in document_sections:
            reports[name] = {
                "created": 0,
                "updated": 0,
                "skipped": 0,
                "errors": [f"section {name!r} is not present in the backup document"],
            }
            ok = False
            continue
        try:
            report = await _maybe_await(tai42_app.backup.import_section(name, document_sections[name]))
        except Exception as exc:  # a failed importer imports nothing for its section
            # Still surfaced in the returned report; also logged so a genuine
            # importer code bug is visible server-side, not just an absent
            # subsystem indistinguishable from it.
            logger.warning("backup import of section %r failed: %s", name, exc, exc_info=True)
            reports[name] = {"created": 0, "updated": 0, "skipped": 0, "errors": [str(exc)]}
            ok = False
            continue
        reports[name] = report
        if report.get("errors"):
            ok = False

    return {"ok": ok, "sections": reports}
