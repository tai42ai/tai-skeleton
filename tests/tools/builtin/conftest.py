"""Test infrastructure for the builtin management tools.

The builtin modules register through the ``tai42_app`` handle at import time
(``@tai42_app.tools.tool``), exactly as external plugins do, so the handle must be
bound before those modules import. Binding the process app here — before the test
modules are collected — lets each test import the builtin tool functions at module
top level; an unstarted app has no manifest, so the decorator returns each tool
function unchanged (no registration side effect) and the tests call it directly.

``bind_app`` swaps in a fake app impl for the duration of a test (the fan-out
tools reach ``app.tools`` / ``app.admin`` / ``app.backends``) and restores the
previous binding afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance

tai42_app.bind(instance.build_app())


@pytest.fixture
def bind_app() -> Iterator[object]:
    """Yield a binder that installs a fake ``tai42_app`` impl and restores the
    previous one on teardown."""
    previous = object.__getattribute__(tai42_app, "_impl")

    def _bind(fake: object) -> object:
        tai42_app.bind(fake)
        return fake

    try:
        yield _bind
    finally:
        tai42_app.bind(previous)
