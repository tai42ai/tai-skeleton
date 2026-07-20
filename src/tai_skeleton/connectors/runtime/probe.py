"""Pull-based reachability checks for a managed sub-service's MCP server.

Two variants over the same transport build:

* :func:`probe` — bool liveness for the API status path (never the tool-call
  hot path). Best-effort: it never raises and logs at DEBUG.
* :func:`verify` — verbose result for the community provider add path: the
  served tool list on success, the specific failure reason otherwise — never
  a bare bool, so a rejected add can say WHY.

Both open the same MCP transport the runtime uses for the sub-service and run
the MCP handshake plus a ``tools/list`` round-trip under a short timeout. For
``http``/``websocket`` servers the access token is carried on the
``Authorization`` header; for ``stdio`` servers the child process is spawned and
answers ``tools/list`` without per-call auth. Liveness is "the server
initialised and answered".

The client is the app-pooled ``FastMCPClient`` taken ``fresh=True`` — a one-shot
client built outside the shared pool and closed on exit, so a probe (which may
spawn a stdio child) never holds or leaks a shared-pool connection.
"""

from __future__ import annotations

import asyncio
import logging

from tai_contract.connectors.probe import ToolSummary, VerifyResult
from tai_contract.connectors.providers import ProviderDescriptor
from tai_contract.manifest import MCPConfig, TaiMCPConfig
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.mcp import FastMCPClient

from tai_skeleton.connectors.runtime.launch import resolve_mcp_server

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 5.0


def _build_probe_config(
    descriptor: ProviderDescriptor,
    sub_service: str,
    access_token: str | None,
    config_values: dict[str, str],
) -> TaiMCPConfig:
    """Build the MCP transport config for a liveness probe of ``sub_service``.

    Mirrors a real managed call: OAuth → bearer ``Authorization`` header (http)
    or tokenless stdio. No-auth → the client's ``config_values`` injected on the
    sub-service's transport channel (env for stdio, headers for http).
    """
    server = resolve_mcp_server(descriptor, sub_service)
    if server.type == "stdio":
        inner = MCPConfig(
            type="stdio",
            command=server.command,
            args=list(server.args),
            env={**dict(server.env), **config_values},
        )
    else:
        headers = dict(server.extra_headers)
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        headers.update(config_values)
        inner = MCPConfig(type=server.type, url=server.url, headers=headers)
    return TaiMCPConfig(title=f"probe_{descriptor.id}_{sub_service}", config=inner)


def _probe_client(config: TaiMCPConfig):
    """A one-shot pooled MCP client for ``config`` (built off-pool, closed on exit)."""
    return client_ctx(FastMCPClient, config=config.model_dump(mode="json"), fresh=True)


async def probe(
    descriptor: ProviderDescriptor,
    sub_service: str,
    *,
    access_token: str | None = None,
    config_values: dict[str, str] | None = None,
) -> bool:
    """Return whether the MCP server for ``sub_service`` is currently live.

    Opens the MCP transport, completes ``initialize`` + ``tools/list`` under a
    short timeout, and returns whether the server answered. Works for both stdio
    and http/websocket sub-services, OAuth and no-auth.

    Never raises: any error (unknown sub-service, transport failure, spawn
    failure, timeout) is logged at DEBUG and reported as unreachable.
    """
    if sub_service not in descriptor.sub_services:
        logger.debug(
            "connectors: probe — unknown sub_service %r for provider %s",
            sub_service,
            descriptor.id,
        )
        return False

    try:
        config = _build_probe_config(descriptor, sub_service, access_token, config_values or {})
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            async with _probe_client(config) as client:
                await client.list_tools()
    except Exception:
        logger.debug(
            "connectors: probe — %s sub_service %s unreachable",
            descriptor.id,
            sub_service,
            exc_info=True,
        )
        return False
    return True


async def verify(
    descriptor: ProviderDescriptor,
    sub_service: str,
    *,
    access_token: str | None = None,
    config_values: dict[str, str] | None = None,
) -> VerifyResult:
    """Verify the MCP server for ``sub_service`` and report the result verbosely.

    Same transport build and ``initialize`` + ``tools/list`` round-trip as
    :func:`probe`, but instead of collapsing to a bool it returns the served
    tool list on success and the specific failure reason (timeout, transport
    error, malformed response, unknown sub-service) on failure — the community
    add path rejects on it and must say why.
    """
    if sub_service not in descriptor.sub_services:
        return VerifyResult(
            ok=False,
            error=f"unknown sub_service {sub_service!r} for provider {descriptor.id!r}",
        )

    try:
        config = _build_probe_config(descriptor, sub_service, access_token, config_values or {})
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            async with _probe_client(config) as client:
                tools = await client.list_tools()
    except TimeoutError:
        return VerifyResult(
            ok=False,
            error=f"timeout: no response within {_PROBE_TIMEOUT_SECONDS}s",
        )
    except Exception:
        # The raw exception text is attacker-influenceable (it can echo upstream
        # server output), so return a fixed reason and keep the detail in the
        # DEBUG log only.
        logger.debug(
            "connectors: verify — %s sub_service %s failed",
            descriptor.id,
            sub_service,
            exc_info=True,
        )
        return VerifyResult(
            ok=False,
            error="transport error: could not complete the MCP handshake",
        )

    return VerifyResult(
        ok=True,
        tools=[ToolSummary(name=tool.name, description=tool.description or "") for tool in tools],
    )
