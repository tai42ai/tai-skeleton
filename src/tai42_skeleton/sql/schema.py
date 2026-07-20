"""Centralized SQL schema loader for the Tai platform.

Usage::

    from tai42_skeleton.sql.schema import load_ddl

    ddl = load_ddl()       # tai42_skeleton.init.sql contents
"""

from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def load_ddl() -> str:
    """Return the full platform DDL (connector store, plus the bundled tables)."""
    return (_RESOURCES_DIR / "tai42_skeleton.init.sql").read_text(encoding="utf-8")
