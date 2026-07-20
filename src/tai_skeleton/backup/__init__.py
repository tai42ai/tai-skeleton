"""The backup subsystem — the concrete ``AppBackup`` registry plus the host's
own core sections.

:class:`~tai_skeleton.backup.registry.BackupRegistry` is the concrete
``tai_contract.app.AppBackup`` impl exposed behind the ``app.backup`` facet: a
plugin (or the host) registers a named section by supplying an ``exporter()`` /
``importer(payload)`` pair, and the registry lists sections and runs one
section's export/import by name. The host is its own first consumer —
:func:`~tai_skeleton.backup.sections.register_core_sections` registers the
skeleton's built-in sections through the SAME registry.
"""

from tai_skeleton.backup.registry import BackupRegistry
from tai_skeleton.backup.sections import register_core_sections

__all__ = ["BackupRegistry", "register_core_sections"]
