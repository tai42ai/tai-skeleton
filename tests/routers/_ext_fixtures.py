"""A second tools module for the tool-extensions router tests.

Provides a lone ``spare`` tool so a manifest can carry a SECOND ``tools`` config
(a distinct module) whose ``extensions`` map targets a tool the FIRST config
provides — the shape the union GET and the 409 mapping-consolidation guard need.
The extension wrappers themselves come from ``tests.presets._fixtures`` (loaded
as the ``extensions_modules`` entry)."""

from __future__ import annotations

from tai42_contract.app import tai42_app


@tai42_app.tools.tool
def spare(text: str) -> str:
    """A standalone tool owned by the second config."""
    return text
