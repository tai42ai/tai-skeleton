"""A single MCP tool advertising an unusable schema is SKIPPED with a loud log,
never allowed to take down the whole binding pass — every other tool the server
advertises still binds. A genuine ``bind_tool_func`` failure still raises, and a
malformed schema-depth setting surfaces as a loud config error, not a per-tool skip."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, cast

import mcp
import pytest
from tai_contract.manifest import MCPConfig, TaiMCPConfig

from tai_skeleton.app.instance import app
from tai_skeleton.manifest import Manifest


class _GoodTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "ok"
        self.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}
        self.outputSchema: dict = {}


class _EmptyAnyOfTool:
    name = "broken"
    description = "malformed"
    # An empty union crashes the converter (``ValueError`` from the kit) — the
    # malformed-schema case the per-tool skip must contain.
    inputSchema: ClassVar[dict] = {"anyOf": []}
    outputSchema: ClassVar[dict] = {}


def _cfg(title: str = "svc") -> TaiMCPConfig:
    return TaiMCPConfig(title=title, include=[], config=MCPConfig(type="http", url="http://x/mcp"))


def test_malformed_tool_skipped_good_tools_still_bind(caplog, monkeypatch):
    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):
            # This test isolates the per-tool skip; seed the manifest include-gate
            # so the fake ``svc`` server default-includes every advertised tool.
            manifest = app._tool_binding._require_manifest()
            manifest.include_title_mcp_tools_map["svc"] = frozenset()
            manifest.exclude_title_mcp_tools_map["svc"] = frozenset()
            tools = [
                cast(mcp.types.Tool, _GoodTool("alpha")),
                cast(mcp.types.Tool, _EmptyAnyOfTool()),
                cast(mcp.types.Tool, _GoodTool("beta")),
            ]
            with caplog.at_level(logging.ERROR):
                # Raises NOTHING despite the malformed middle tool.
                app._tool_binding.mcp_tools(_cfg(), tools)

            bound = {t.name for t in await app._fast_mcp.local_provider.list_tools()}
            assert "svc_alpha" in bound
            assert "svc_beta" in bound
            assert "svc_broken" not in bound

            # The skip is loud: an ERROR log naming the server title + tool name.
            skip_logs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
            assert any("svc" in m and "broken" in m for m in skip_logs)

    asyncio.run(run())


def test_bind_tool_func_failure_still_raises(monkeypatch):
    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):
            manifest = app._tool_binding._require_manifest()
            manifest.include_title_mcp_tools_map["svc"] = frozenset()
            manifest.exclude_title_mcp_tools_map["svc"] = frozenset()

            # ``mcp_tool_to_func`` succeeds (outside the guard-covered path), but a
            # genuine registration bug inside ``bind_tool_func`` must NOT be
            # swallowed — the guard wraps only the adaptation, not the bind.
            def _boom(*args, **kwargs):
                raise RuntimeError("registration bug")

            monkeypatch.setattr(app._tool_binding, "bind_tool_func", _boom)
            with pytest.raises(RuntimeError, match="registration bug"):
                app._tool_binding.mcp_tools(_cfg(), [cast(mcp.types.Tool, _GoodTool("alpha"))])

    asyncio.run(run())


def test_malformed_schema_max_depth_env_surfaces_as_config_error(monkeypatch):
    # A malformed TAI_MCP_SCHEMA_MAX_DEPTH is a CONFIG error: it must surface loudly
    # from the binding pass, NOT be caught by the per-tool skip guard and mis-logged
    # as every tool advertising an unusable schema. The depth setting is resolved
    # once, up front, OUTSIDE the guard, so a bad value fails before any tool is bound.
    from pydantic import ValidationError
    from tai_kit.settings import reset_all_settings

    async def run() -> None:
        async with app.app_context(Manifest.model_validate({})):
            manifest = app._tool_binding._require_manifest()
            manifest.include_title_mcp_tools_map["svc"] = frozenset()
            manifest.exclude_title_mcp_tools_map["svc"] = frozenset()
            monkeypatch.setenv("TAI_MCP_SCHEMA_MAX_DEPTH", "0")  # violates gt=0
            reset_all_settings()
            try:
                # Raised loudly from the binding pass (the depth setting is resolved
                # before the loop). Reverting the resolve back inside the per-tool
                # guard would catch this and skip the tool instead — no raise.
                with pytest.raises(ValidationError):
                    app._tool_binding.mcp_tools(_cfg(), [cast(mcp.types.Tool, _GoodTool("alpha"))])
            finally:
                monkeypatch.delenv("TAI_MCP_SCHEMA_MAX_DEPTH", raising=False)
                reset_all_settings()

    asyncio.run(run())
