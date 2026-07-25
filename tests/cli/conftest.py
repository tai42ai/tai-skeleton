"""CLI launcher tests drive code that writes process environment variables
directly (``TAI_MANIFEST_PATH``, ``TAI_TRANSPORT``, the backend manifest key,
``PROMETHEUS_MULTIPROC_DIR``). Those writes are real launcher behavior, not
test setup, so they bypass ``monkeypatch`` and would otherwise leak into the
rest of the suite. This autouse fixture snapshots and restores ``os.environ``
around every CLI test to keep them hermetic.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    saved = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
