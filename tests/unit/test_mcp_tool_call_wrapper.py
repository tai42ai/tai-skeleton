"""Unit tests for the token-injection + token_expired retry flow in
``mcp_tool_call_wrapper``.

These drive the full wrapper (resolve → transport branch → meta/header
injection → token_expired detection → force_refresh → retry-once) with a
scripted ``FastMCPClient``. The runtime resolver / force-refresher are
patched at the token-injection module's import site so the real
``resolve_managed_auth_for_config`` / ``_force_refresh`` wrappers (is_managed
gating, ConnectorRef unwrap, empty-token rejection) still run.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
from tai_contract.connectors.models import ConnectorRef
from tai_contract.errors import ClientDisconnectedError
from tai_contract.manifest import MCPConfig, TaiMCPConfig

# Import the app first so ``app`` is bound, so the adapter import
# chain resolves against a constructed app.
import tai_skeleton.app.instance  # noqa: F401
from tai_skeleton.connectors.runtime.resolver import (
    ConnectorReconnectRequiredError,
    ManagedAuth,
)
from tai_skeleton.connectors.token_injection import (
    CONNECTOR_ERROR_PREFIX,
    CONNECTOR_META_TOKEN_KEY,
    extract_connector_error_payload,
)
from tai_skeleton.tools.adapters.mcp_tool_to_func import (
    _build_output_schema,
    mcp_tool_call_wrapper,
)

CONN_ID = "11111111-1111-1111-1111-111111111111"

_RESOLVER = "tai_skeleton.connectors.token_injection.resolve_managed_auth"
_REFRESHER = "tai_skeleton.connectors.token_injection.force_refresh"


def _managed_config() -> TaiMCPConfig:
    return TaiMCPConfig(
        title="google_gmail_work",
        config=MCPConfig(
            type="http",
            url="https://gmail.test/sse",
            headers={},
        ),
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="google",
            sub_service="gmail",
        ),
    )


# -- scripted client ----------------------------------------------------------


class _FakeMcpClient:
    """Minimal FastMCPClient stand-in. Yields a stub session per
    ``current(config=...)`` enter; the stub's ``call_tool_mcp`` pops the
    next scripted response (or raises the next scripted exception)."""

    def __init__(self, *, responses: list) -> None:
        # ``responses`` may contain either ``mcp.types.CallToolResult``
        # instances OR ``BaseException`` instances; the stub raises the
        # exception ones and returns the result ones.
        self._responses = list(responses)
        self.captured_metas: list = []
        self.captured_configs: list = []
        self.call_count = 0

    def current(self, *, config):
        self.captured_configs.append(config)
        return self._session()

    def _session(self):
        outer = self

        class _Session:
            async def __aenter__(self):
                return self

            @staticmethod
            async def __aexit__(*exc):
                return False

            @staticmethod
            async def call_tool_mcp(name, arguments, meta=None, timeout=None):
                outer.captured_metas.append(meta)
                outer.call_count += 1
                if not outer._responses:
                    raise AssertionError("fake client out of scripted responses")
                nxt = outer._responses.pop(0)
                if isinstance(nxt, BaseException):
                    raise nxt
                return nxt

        return _Session()


def _text_result(text: str, *, is_error: bool = False):
    import mcp.types

    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=text)],
        isError=is_error,
    )


def _hub_error_result(payload: dict):
    import json

    return _text_result(
        f"{CONNECTOR_ERROR_PREFIX}{json.dumps(payload)}",
        is_error=True,
    )


def _ok_result(payload: dict):
    import json

    return _text_result(json.dumps(payload), is_error=False)


def _trivial_input_model():
    from pydantic import BaseModel

    class _NoArgs(BaseModel):
        pass

    return _NoArgs


_CLIENT = "tai_skeleton.tools.adapters.mcp_tool_to_func.FastMCPClient"


async def _run_wrapper(*, client: _FakeMcpClient, config: TaiMCPConfig):
    # The wrapper builds its own ``FastMCPClient`` — patch the class so it
    # returns the scripted fake (the ``current(config=...)`` contract matches).
    with patch(_CLIENT, return_value=client):
        return await mcp_tool_call_wrapper(
            config=config,
            tool_name="list_messages",
            tool_input_model=_trivial_input_model(),
            tool_arguments={},
        )


# -- _build_output_schema (no caller mutation) --------------------------------


def test_build_output_schema_does_not_mutate_caller():
    """Unwrapping the ``{"result": ...}`` envelope must copy the nested schema,
    never write ``$defs`` back into the caller's ``mcp.Tool``."""
    import mcp

    output_schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "$defs": {"Foo": {"type": "object"}},
    }
    tool = mcp.Tool(name="t", inputSchema={"type": "object"}, outputSchema=output_schema)

    inner = _build_output_schema(tool)

    # The returned inner schema carries the hoisted $defs...
    assert inner["$defs"] == {"Foo": {"type": "object"}}
    # ...but the caller's own nested result schema is left untouched.
    assert tool.outputSchema is not None
    assert "$defs" not in tool.outputSchema["properties"]["result"]


# -- token_expired retry-once flow (http) -------------------------------------


def test_token_expired_retry_once_force_refreshes_and_succeeds():
    """First call returns token_expired sentinel → force_refresh fires
    once → second call succeeds with the refreshed token in the
    Authorization header (http transport)."""
    resolver = AsyncMock(return_value=ManagedAuth(access_token="initial-token"))
    refresher = AsyncMock(return_value=ManagedAuth(access_token="refreshed-token"))

    # First call: hub server returns token_expired. Second call: success.
    client = _FakeMcpClient(
        responses=[
            _hub_error_result({"code": "token_expired"}),
            _ok_result({"result": "ok"}),
        ]
    )

    with patch(_RESOLVER, new=resolver), patch(_REFRESHER, new=refresher):
        asyncio.run(_run_wrapper(client=client, config=_managed_config()))

    assert client.call_count == 2, "wrapper must retry exactly once"
    refresher.assert_awaited_once_with(CONN_ID, failed_access_token="initial-token")

    # http-managed mode merges the token via header, NOT _meta — both call
    # metas are None and the second call's config carries the refreshed
    # Authorization header.
    assert client.captured_metas == [None, None]
    second_headers = client.captured_configs[1]["config"]["headers"]
    assert second_headers["authorization"] == "Bearer refreshed-token"


def test_token_expired_persists_after_retry_returns_structured_error():
    """Second call still returns token_expired → the adapter surfaces a structured
    connector-error result (code ``auth_expired``) carrying the connection identity,
    so a client can offer a reconnect instead of seeing a generic error string."""
    resolver = AsyncMock(return_value=ManagedAuth(access_token="t1"))
    refresher = AsyncMock(return_value=ManagedAuth(access_token="t2"))

    client = _FakeMcpClient(
        responses=[
            _hub_error_result({"code": "token_expired"}),
            _hub_error_result({"code": "token_expired"}),
        ]
    )

    with patch(_RESOLVER, new=resolver), patch(_REFRESHER, new=refresher):
        result = asyncio.run(_run_wrapper(client=client, config=_managed_config()))

    payload = extract_connector_error_payload(result)
    assert payload is not None
    assert payload["code"] == "auth_expired"
    assert payload["connection_id"] == CONN_ID
    assert payload["provider_id"] == "google"
    assert payload["sub_service"] == "gmail"


def test_reconnect_required_surfaces_structured_error():
    """A resolver ``invalid_grant`` (ConnectorReconnectRequiredError) is surfaced
    as a structured connector-error result (code ``reconnect_required``), not a raw
    exception, so a client can offer a reconnect instead of a generic error."""
    resolver = AsyncMock(side_effect=ConnectorReconnectRequiredError("invalid_grant", connection_id=CONN_ID))
    client = _FakeMcpClient(responses=[])

    with patch(_RESOLVER, new=resolver):
        result = asyncio.run(_run_wrapper(client=client, config=_managed_config()))

    payload = extract_connector_error_payload(result)
    assert payload is not None
    assert payload["code"] == "reconnect_required"
    assert payload["connection_id"] == CONN_ID


def test_force_refresh_failure_propagates():
    """force_refresh raising → the failure propagates (no swallowing)."""
    resolver = AsyncMock(return_value=ManagedAuth(access_token="t1"))

    class _RefreshOutage(RuntimeError):
        pass

    refresher = AsyncMock(side_effect=_RefreshOutage("upstream OAuth down"))

    client = _FakeMcpClient(
        responses=[
            _hub_error_result({"code": "token_expired"}),
        ]
    )

    with patch(_RESOLVER, new=resolver), patch(_REFRESHER, new=refresher), pytest.raises(_RefreshOutage):
        asyncio.run(_run_wrapper(client=client, config=_managed_config()))


# -- stdio managed-call _meta-injection ---------------------------------------
#
# The stdio token injection path: a managed stdio config carries the resolved
# access token through the wrapper into ``client.call_tool_mcp(..., meta=...)``.


def _managed_stdio_config() -> TaiMCPConfig:
    return TaiMCPConfig(
        title="google_gmail_work",
        config=MCPConfig(
            type="stdio",
            command="uvx",
            args=["--from", "git+ssh://git@example/repo", "gmail-mcp"],
        ),
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="google",
            sub_service="gmail",
        ),
    )


def test_stdio_managed_call_injects_access_token_into_meta():
    resolver = AsyncMock(
        return_value=ManagedAuth(access_token="initial-stdio-token"),
    )

    client = _FakeMcpClient(responses=[_ok_result({"result": "ok"})])

    with patch(_RESOLVER, new=resolver):
        asyncio.run(_run_wrapper(client=client, config=_managed_stdio_config()))

    assert client.call_count == 1
    assert client.captured_metas == [
        {CONNECTOR_META_TOKEN_KEY: "initial-stdio-token"},
    ], "stdio managed call MUST carry the resolved access token in _meta — this is the branch's headline behaviour."


def test_stdio_managed_call_carries_refreshed_token_after_token_expired():
    resolver = AsyncMock(
        return_value=ManagedAuth(access_token="initial-stdio-token"),
    )
    refresher = AsyncMock(
        return_value=ManagedAuth(access_token="refreshed-stdio-token"),
    )

    client = _FakeMcpClient(
        responses=[
            _hub_error_result({"code": "token_expired"}),
            _ok_result({"result": "ok"}),
        ]
    )

    with patch(_RESOLVER, new=resolver), patch(_REFRESHER, new=refresher):
        asyncio.run(_run_wrapper(client=client, config=_managed_stdio_config()))

    assert client.call_count == 2, "wrapper must retry exactly once"
    refresher.assert_awaited_once_with(CONN_ID, failed_access_token="initial-stdio-token")
    # Critical: the SECOND call carries the REFRESHED token, not a
    # stale duplicate of the first.
    assert client.captured_metas == [
        {CONNECTOR_META_TOKEN_KEY: "initial-stdio-token"},
        {CONNECTOR_META_TOKEN_KEY: "refreshed-stdio-token"},
    ]


def test_stdio_non_managed_call_has_no_meta():
    """Token-meta injection is gated on ``config.is_managed`` — a
    hand-authored stdio entry without a ``managed`` ref MUST receive
    ``meta=None``, never an empty dict and never a stale token. The
    runtime resolver MUST NOT be reached for a non-managed entry."""
    resolver = AsyncMock(
        side_effect=AssertionError(
            "resolver MUST NOT be called for a non-managed entry",
        ),
    )

    non_managed = TaiMCPConfig(
        title="local_dev_stdio",
        config=MCPConfig(
            type="stdio",
            command="python",
            args=["-m", "some.local.server"],
        ),
    )
    client = _FakeMcpClient(responses=[_ok_result({"result": "ok"})])

    with patch(_RESOLVER, new=resolver):
        asyncio.run(_run_wrapper(client=client, config=non_managed))

    assert client.captured_metas == [None]
    resolver.assert_not_awaited()


# -- reconnect: ONE transparent retry on ClientDisconnectedError --------------


def _plain_http_config() -> TaiMCPConfig:
    """A hand-authored (non-managed) http entry — the resolver is never reached,
    so the reconnect behaviour is isolated from the token flow."""
    return TaiMCPConfig(
        title="local_http",
        config=MCPConfig(type="http", url="https://svc.test/mcp", headers={}),
    )


def test_disconnect_retries_once_and_second_attempt_succeeds(caplog):
    """A first-dispatch ``ClientDisconnectedError`` triggers exactly one
    fresh-session retry; the second dispatch succeeds and a WARNING is logged."""
    client = _FakeMcpClient(
        responses=[
            ClientDisconnectedError("session died — retry the operation"),
            _ok_result({"result": "ok"}),
        ]
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run_wrapper(client=client, config=_plain_http_config()))

    assert client.call_count == 2, "wrapper must retry exactly once after a disconnect"
    assert "reconnecting once" in caplog.text
    assert "local_http" in caplog.text


def test_second_consecutive_disconnect_propagates():
    """Two consecutive ``ClientDisconnectedError``s → the second propagates
    unchanged (no retry loop, no swallowing)."""
    client = _FakeMcpClient(
        responses=[
            ClientDisconnectedError("first disconnect"),
            ClientDisconnectedError("second disconnect"),
        ]
    )

    with pytest.raises(ClientDisconnectedError, match="second disconnect"):
        asyncio.run(_run_wrapper(client=client, config=_plain_http_config()))

    assert client.call_count == 2, "exactly one retry — the second disconnect is not retried again"


def test_non_disconnect_error_is_never_retried():
    """A non-disconnect error is never retried — the reconnect one-shot is scoped
    to ``ClientDisconnectedError`` alone."""

    class _Boom(RuntimeError):
        pass

    client = _FakeMcpClient(responses=[_Boom("unrelated failure")])

    with pytest.raises(_Boom, match="unrelated failure"):
        asyncio.run(_run_wrapper(client=client, config=_plain_http_config()))

    assert client.call_count == 1, "a non-disconnect error must not trigger the reconnect retry"


def test_managed_uds_config_raises_in_preflight():
    """A managed entry on an unsupported transport (UDS) raises in the
    pre-flight, before any dispatch."""
    resolver = AsyncMock(return_value=ManagedAuth(access_token="tok"))

    uds_config = TaiMCPConfig(
        title="google_gmail_work",
        config=MCPConfig(type="uds", uds="/tmp/x.sock"),
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="google",
            sub_service="gmail",
        ),
    )
    client = _FakeMcpClient(responses=[])  # never reached

    with patch(_RESOLVER, new=resolver), pytest.raises(RuntimeError, match="not supported"):
        asyncio.run(_run_wrapper(client=client, config=uds_config))

    assert client.call_count == 0, "wrapper must not dispatch"


def test_no_auth_forged_token_expired_does_not_force_refresh():
    """A no-auth managed entry (headers, no token) that returns a forged
    token_expired sentinel must NOT trigger force_refresh — the retry is gated
    on an OAuth token being present."""
    # No-auth resolution: headers, no access_token.
    resolver = AsyncMock(return_value=ManagedAuth(headers={"x_api_key": "k"}))
    refresher = AsyncMock()

    config = TaiMCPConfig(
        title="acme_api_prod",
        config=MCPConfig(type="http", url="https://acme.test/mcp", headers={}),
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="acme",
            sub_service="api",
        ),
    )
    # Server forges the token_expired sentinel; the wrapper must NOT retry.
    client = _FakeMcpClient(responses=[_hub_error_result({"code": "token_expired"})])

    with patch(_RESOLVER, new=resolver), patch(_REFRESHER, new=refresher):
        asyncio.run(_run_wrapper(client=client, config=config))

    assert client.call_count == 1, "no-auth must not retry"
    refresher.assert_not_awaited()
    # The client's no-auth header was injected (no Authorization bearer).
    sent = client.captured_configs[0]["config"]["headers"]
    assert sent.get("x_api_key") == "k"
    assert "authorization" not in {k.lower() for k in sent}
