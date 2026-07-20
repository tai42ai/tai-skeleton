"""Resilience/strip invariants.

The headline guarantee after the "strip" change: a failed MCP is recorded
and surfaced as *title + coarse status only* — never the MCP config
(credentials) or raw exception text — on every path that is LLM-callable,
logged, or broadcast. These tests lock that invariant and the reload result
shapes so a future edit re-introducing a `config`/`error-text` field fails
the suite instead of silently leaking.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from tai_contract.manifest import TaiMCPConfig

from tai_skeleton.app.lifecycle import TaiMCPLifecycleMixin
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.manifest import Manifest
from tai_skeleton.tools.registry import ToolRegistry


class _NoManifestConfig:
    """Embedded/test runtime with no external manifest file: ``read_manifest``
    raises ``FileNotFoundError`` so ``_refresh_manifest_mcp`` keeps its in-memory
    rows."""

    def read_manifest(self):
        raise FileNotFoundError("no external manifest")


class _Mixin(TaiMCPLifecycleMixin):
    """Concrete-enough subclass to exercise the failed-MCP/reload logic
    without an event server, network, or the full app."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_manager = _NoManifestConfig()  # pyright: ignore[reportAttributeAccessIssue]

    def _mcp_tools(self, config, tools):  # abstract in the mixin
        self._mcp_bound_tools[config.title] = {f"{config.title}_t"}


def _cfg(title="x", **config):
    config.setdefault("type", "http")
    config.setdefault("url", "http://user:SECRET@h:9000/mcp")
    config.setdefault("headers", {"Authorization": "Bearer LEAKTOKEN"})
    return TaiMCPConfig.model_validate({"title": title, "include": [], "config": config})


class TestSoftValidation:
    def test_raises_for_missing(self):
        with pytest.raises(TaiValidationError):
            ToolRegistry(requested_tools={"foo"}, tool_extensions={}).validation()

    def test_ignore_suppresses_owned_missing(self):
        ToolRegistry(requested_tools={"foo"}, tool_extensions={}).validation(ignore=frozenset({"foo"}))

    def test_ignore_does_not_hide_other_missing(self):
        with pytest.raises(TaiValidationError):
            ToolRegistry(requested_tools={"foo", "bar"}, tool_extensions={}).validation(ignore=frozenset({"foo"}))


class TestFailedMcpStrip:
    def test_record_stores_only_title_to_status(self):
        m = _Mixin()
        cfg = _cfg("redis")
        m._record_failed_mcp(cfg, "TimeoutError")
        # Only the title is stored, mapped to a coarse status — the config
        # (and thus its credentials) never enters process state.
        assert m._failed_mcps == {"redis": "unavailable"}

    def test_list_failed_mcps_shape_has_no_config_or_reason(self):
        m = _Mixin()
        m._failed_mcps = {"a": "unavailable", "b": "unavailable"}
        out = m._list_failed_mcps()
        assert out == [
            {"title": "a", "status": "unavailable"},
            {"title": "b", "status": "unavailable"},
        ]
        # Hard guard: no entry may carry any key beyond title/status, and no
        # secret substring may appear anywhere in the serialized result.
        for entry in out:
            assert set(entry) == {"title", "status"}
        assert "SECRET" not in repr(out)
        assert "LEAKTOKEN" not in repr(out)


class TestReloadShapes:
    def test_unknown_title_is_structured_error(self):
        m = _Mixin()
        m._manifest = Manifest.model_validate({})
        r = asyncio.run(m._reload_mcp_async("nope"))
        assert r["title"] == "nope"
        assert r["status"] == "error"
        assert r["error"].startswith("Unknown MCP 'nope'")
        assert set(r) == {"title", "status", "error"}

    def test_reload_failed_keeps_siblings_and_strips_text(self):
        m = _Mixin()
        m._manifest = Manifest.model_validate({"mcp": [_cfg("a").model_dump(), _cfg("b").model_dump()]})
        m._failed_mcps = {"a": "unavailable", "b": "unavailable"}
        # Probe result is irrelevant here — the apply half is patched below.
        m._probe_mcp = AsyncMock(return_value=[])

        async def fake_apply(title, config, tools):
            if title == "b":
                raise RuntimeError("boom with SECRET in message")
            return {"title": title, "status": "ok"}

        m._apply_reloaded_mcp = fake_apply
        out = asyncio.run(m._reload_failed_mcps_async())
        # One bad title must not discard the other; the failure is coarse in the
        # RESULT (no exception text — strip policy); the trace goes to the log only.
        assert out == [
            {"title": "a", "status": "ok"},
            {"title": "b", "status": "error"},
        ]
        assert "SECRET" not in repr(out)
