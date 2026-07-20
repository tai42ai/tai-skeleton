"""Edge-path coverage for the manifest impl: the unknown-module ImportError in
the prefix walk, and the per-entry error handling in ``_build_map`` (a malformed
config row fails the whole build LOUDLY rather than being silently omitted from
the derived maps while the build reports success)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from tai_skeleton.manifest import Manifest


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "tools": [
                {"title": "Math", "module": "pkg.math_tools", "include": ["add"]},
            ],
        }
    )


def test_should_include_unknown_module_raises_import_error() -> None:
    """A module with no manifest entry (and no registered prefix) raises loudly
    rather than silently passing the filter."""
    m = _manifest()
    with pytest.raises(ImportError, match="not found in manifest"):
        m.should_include_tool("add", "totally.unrelated.module")


def test_build_map_raises_on_a_broken_config_entry() -> None:
    """A config object that raises on attribute access fails the whole build
    loudly: a malformed entry must not be silently omitted from the maps while
    the build reports success (that would quietly change which entries load)."""

    class _BoomConfig:
        include: ClassVar[list[str]] = []
        exclude: ClassVar[list[str]] = []

        @property
        def module(self) -> str:
            raise RuntimeError("malformed entry")

        title = "boom"

    class _GoodConfig:
        module = "pkg.ok"
        title = "ok"
        include: ClassVar[list[str]] = ["x"]
        exclude: ClassVar[list[str]] = []

    m = _manifest()

    with pytest.raises(ValueError, match="failed to build map entries") as excinfo:
        m._build_map([_BoomConfig(), _GoodConfig()], key="module")
    # The underlying cause is chained, and the malformed entry's error is named.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "malformed entry" in str(excinfo.value)
