"""Out-of-tree call log for the importer single-import test.

Lives OUTSIDE any reloaded package, so its list survives the pop+reimport that
``import_or_reload_package`` performs on the package under test.
"""

from __future__ import annotations

INIT_CALLS: list[str] = []
