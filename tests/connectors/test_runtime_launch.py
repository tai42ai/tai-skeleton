"""Generic stdio launch-spec synthesis (resolve_mcp_server)."""

from __future__ import annotations

import pytest
from tai_contract.connectors.providers import (
    ProviderDescriptor,
    SubServiceDescriptor,
)

from tai_skeleton.connectors.runtime.launch import resolve_mcp_server

from .conftest import make_noauth_http_descriptor, make_noauth_stdio_descriptor


def _stdio_provider(*, pkg_manager, pkg_version=None, entry_point="tai-mcp-x"):
    return ProviderDescriptor(
        id="prov",
        display_name="Prov",
        icon_url="https://prov.test/i.png",
        kind="none",
        origin="system",
        category="data",
        pkg_manager=pkg_manager,
        pkg_version=pkg_version,
        sub_services={
            "svc": SubServiceDescriptor(
                id="svc",
                display_name="Svc",
                entry_point=entry_point,
            ),
        },
    )


def test_returns_declared_mcp_server_as_is():
    desc = make_noauth_http_descriptor()
    server = resolve_mcp_server(desc, "main")
    assert server.type == "http"
    assert server.url == "https://httpsvc.test/mcp"


def test_uvx_with_version():
    server = resolve_mcp_server(_stdio_provider(pkg_manager="uvx", pkg_version="1.2.3"), "svc")
    assert server.type == "stdio"
    assert server.command == "uvx"
    assert server.args == ["--from", "tai-mcp-x==1.2.3", "tai-mcp-x"]


def test_uvx_without_version():
    server = resolve_mcp_server(_stdio_provider(pkg_manager="uvx"), "svc")
    assert server.command == "uvx"
    assert server.args == ["tai-mcp-x"]


def test_npx_with_version():
    server = resolve_mcp_server(_stdio_provider(pkg_manager="npx", pkg_version="2.0.0"), "svc")
    assert server.command == "npx"
    assert server.args == ["-y", "tai-mcp-x@2.0.0"]


def test_npx_without_version():
    server = resolve_mcp_server(_stdio_provider(pkg_manager="npx"), "svc")
    assert server.command == "npx"
    assert server.args == ["-y", "tai-mcp-x"]


def test_default_stdio_descriptor_uses_uvx_pinned():
    server = resolve_mcp_server(make_noauth_stdio_descriptor(), "search")
    assert server.args == ["--from", "tai-mcp-widgets==1.2.3", "tai-mcp-widgets"]


def test_unknown_pkg_manager_raises():
    desc = _stdio_provider(pkg_manager="npx")
    object.__setattr__(desc, "pkg_manager", "pip")
    with pytest.raises(ValueError, match="expected 'uvx' or 'npx'"):
        resolve_mcp_server(desc, "svc")


def test_missing_pkg_manager_raises():
    desc = _stdio_provider(pkg_manager="npx")
    object.__setattr__(desc, "pkg_manager", None)
    with pytest.raises(ValueError, match="expected 'uvx' or 'npx'"):
        resolve_mcp_server(desc, "svc")


def test_neither_mcp_server_nor_entry_point_raises():
    """Belt-and-braces guard: a sub-service with neither launch path set."""
    desc = make_noauth_stdio_descriptor()
    # Force the invariant-broken state past the contract validator.
    object.__setattr__(desc.sub_services["search"], "entry_point", None)
    with pytest.raises(ValueError, match="neither mcp_server nor entry_point"):
        resolve_mcp_server(desc, "search")


def test_leading_dash_entry_point_rejected():
    desc = _stdio_provider(pkg_manager="uvx", entry_point="ok")
    object.__setattr__(desc.sub_services["svc"], "entry_point", "-evil")
    with pytest.raises(ValueError, match="entry_point"):
        resolve_mcp_server(desc, "svc")


def test_leading_dash_pkg_version_rejected():
    desc = _stdio_provider(pkg_manager="uvx", pkg_version="1.0")
    object.__setattr__(desc, "pkg_version", "-1.0")
    with pytest.raises(ValueError, match="pkg_version"):
        resolve_mcp_server(desc, "svc")
