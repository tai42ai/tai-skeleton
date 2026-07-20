"""Auto-reset every cached settings accessor between tests so a
``monkeypatch.setenv`` in one test can't bleed into the next via a cached value.

Does not undo the import-time constant capture in
``tools/adapters/mcp_tool_to_func.py`` — that needs a re-import, out of scope here.

Also binds the process app singleton to the ``tai_app`` contract handle before
the test modules are imported: the router modules (imported by
``test_universal_webhook``) register their routes through the handle at import
time, exactly as external plugins do, and the runtime imports them only after
``start()`` binds the handle.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tai_contract.app import tai_app
from tai_kit.settings import reset_all_settings

from tai_skeleton.app import instance

tai_app.bind(instance.build_app())


@pytest.fixture(autouse=True)
def _reset_settings_caches_between_tests() -> Iterator[None]:
    reset_all_settings()
    yield
    reset_all_settings()
