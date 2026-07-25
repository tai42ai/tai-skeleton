"""The concrete ``AppBackup`` registry — the impl body behind the ``app.backup``
facet.

A section is a named ``(exporter, importer)`` pair plus a ``secret`` flag. The
registry stores sections in registration order, lists them for the UI, and runs
one section's exporter/importer by name. An unknown name raises loudly — never a
silent no-op — and a duplicate registration raises rather than overwrite.

The contract facet is synchronous, but a section's real work (redis, Postgres,
a run-tool call) is asynchronous. The exporter/importer callables are therefore
allowed to be either sync or async: :meth:`export_section` / :meth:`import_section`
run the callable and return its result verbatim (a coroutine for an async
section), and the async HTTP router awaits an awaitable result. The registry
itself neither awaits nor inspects the payload — it is a pure name-to-callable
dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tai42_contract.backup import BackupSectionInfo


@dataclass(frozen=True)
class _Section:
    """One registered section: its name, exporter/importer pair, and secrecy."""

    name: str
    exporter: Callable[[], Any]
    importer: Callable[[Any], Any]
    secret: bool


class BackupRegistry:
    """Ordered registry of named backup sections (``tai42_contract.app.AppBackup``)."""

    def __init__(self) -> None:
        # Insertion-ordered: ``sections()`` reports registration order, so the UI
        # renders host sections before any plugin-added ones.
        self._sections: dict[str, _Section] = {}

    def register_section(
        self,
        name: str,
        exporter: Callable[[], Any],
        importer: Callable[[Any], Any],
        *,
        secret: bool = False,
    ) -> None:
        """Register a section under ``name``. A duplicate name raises rather than
        silently overwrite an existing section."""
        if name in self._sections:
            raise ValueError(f"backup section {name!r} is already registered")
        self._sections[name] = _Section(name=name, exporter=exporter, importer=importer, secret=secret)

    def sections(self) -> list[BackupSectionInfo]:
        """Every registered section as a ``BackupSectionInfo``, in registration order."""
        return [BackupSectionInfo(name=section.name, secret=section.secret) for section in self._sections.values()]

    def export_section(self, name: str) -> Any:
        """Run ``name``'s exporter and return its payload. Unknown name raises."""
        return self._require(name).exporter()

    def import_section(self, name: str, payload: Any) -> Any:
        """Run ``name``'s importer over ``payload`` and return its report. Unknown
        name raises."""
        return self._require(name).importer(payload)

    def _require(self, name: str) -> _Section:
        try:
            return self._sections[name]
        except KeyError:
            raise KeyError(f"unknown backup section: {name!r}") from None
