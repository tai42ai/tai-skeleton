"""OAuth client: PKCE, redirect allow-list, authorize URL, exchange/refresh/revoke."""

from __future__ import annotations

import base64
import hashlib
from contextlib import asynccontextmanager
from typing import cast
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from tai42_contract.connectors.errors import OperatorMisconfiguredError

import tai42_skeleton.connectors.oauth.client as client_mod
from tai42_skeleton.connectors.oauth import client
from tai42_skeleton.connectors.oauth.client import (
    CodeExchangeFailedError,
    RedirectUriNotAllowedError,
    RefreshTokenMissingError,
    TokenRefreshFailedError,
)

from .conftest import FakeHttp, FakeHttpResponse, make_oauth_descriptor


@pytest.fixture
def install_http(monkeypatch):
    """Install a queued fake pooled-HTTP client; return the FakeHttp for asserts."""

    def _install(responses):
        fake = FakeHttp(responses)

        @asynccontextmanager
        async def fake_http():
            yield fake

        monkeypatch.setattr(client_mod, "_http", fake_http)
        return fake

    return _install


# -- PKCE --------------------------------------------------------------------


def test_generate_pkce_pair_is_valid_s256():
    verifier, challenge = client.generate_pkce_pair()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    assert challenge == expected
    # fresh each call
    assert client.generate_pkce_pair()[0] != verifier


# -- Redirect allow-list -----------------------------------------------------


def test_validate_redirect_uri_accepts_https_in_allowlist():
    out = client.validate_redirect_uri("https://app.example.com/oauth-bridge.html")
    assert out == "https://app.example.com/oauth-bridge.html"


def test_validate_redirect_uri_accepts_localhost_http():
    out = client.validate_redirect_uri("http://localhost:5173/oauth-bridge.html")
    assert out == "http://localhost:5173/oauth-bridge.html"


def test_validate_redirect_uri_returns_uri_unchanged():
    # No normalization: OAuth requires the redirect_uri to be byte-identical
    # across authorize and token exchange, so a trailing slash is preserved.
    assert client.validate_redirect_uri("https://app.example.com/") == "https://app.example.com/"


def test_validate_redirect_uri_rejects_empty():
    with pytest.raises(RedirectUriNotAllowedError, match="non-empty"):
        client.validate_redirect_uri("")


def test_validate_redirect_uri_rejects_unparseable():
    with pytest.raises(RedirectUriNotAllowedError, match="scheme and host"):
        client.validate_redirect_uri("notaurl")


def test_validate_redirect_uri_rejects_http_for_non_local():
    with pytest.raises(RedirectUriNotAllowedError, match="https"):
        client.validate_redirect_uri("http://app.example.com/cb")


def test_validate_redirect_uri_rejects_origin_not_in_allowlist():
    with pytest.raises(RedirectUriNotAllowedError, match="not in the allow-list"):
        client.validate_redirect_uri("https://evil.example.com/cb")


def test_validate_redirect_uri_rejects_unallowlisted_bridge_origin():
    # A configured OAuth bridge origin must ALSO appear in
    # CONNECTORS_REDIRECT_URI_ALLOWLIST, or every Connect flow bounced through it
    # fails closed here at the token exchange.
    with pytest.raises(RedirectUriNotAllowedError, match="not in the allow-list"):
        client.validate_redirect_uri("https://bridge.tai42.ai/oauth-bridge.html")


def test_validate_redirect_uri_rejects_empty_allowlist(monkeypatch):
    from tai42_kit.settings import reset_all_settings

    monkeypatch.setenv("CONNECTORS_REDIRECT_URI_ALLOWLIST", "")
    reset_all_settings()
    with pytest.raises(RedirectUriNotAllowedError, match="allow-list"):
        client.validate_redirect_uri("https://app.example.com/cb")


# -- Authorize URL -----------------------------------------------------------


def test_build_authorize_url(oauth_client_env):
    desc = make_oauth_descriptor(extra_authorize_params={"access_type": "offline"})
    url = client.build_authorize_url(
        descriptor=desc,
        scopes=["mail.read", "mail.send"],
        state="signed-state",
        code_challenge="chal",
        redirect_uri="https://app.example.com/oauth-bridge.html",
    )
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "acme.test"
    q = parse_qs(parsed.query)
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["client-123"]
    assert q["scope"] == ["mail.read mail.send"]
    assert q["state"] == ["signed-state"]
    assert q["code_challenge"] == ["chal"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["access_type"] == ["offline"]


def test_build_authorize_url_enforces_allowlist(oauth_client_env):
    desc = make_oauth_descriptor()
    with pytest.raises(RedirectUriNotAllowedError):
        client.build_authorize_url(
            descriptor=desc,
            scopes=["mail.read"],
            state="s",
            code_challenge="c",
            redirect_uri="https://evil.example.com/cb",
        )


# -- Operator credential reads -----------------------------------------------


def test_client_id_missing_env_raises_operator_misconfigured():
    desc = make_oauth_descriptor()
    with pytest.raises(OperatorMisconfiguredError) as ei:
        client._client_id(desc)
    assert ei.value.env_var == "ACME_CLIENT_ID"
    assert ei.value.provider_id == "acme"


def test_client_id_without_env_field_raises_runtime():
    desc = make_oauth_descriptor()
    object.__setattr__(desc, "client_id_env", None)
    with pytest.raises(RuntimeError, match="not an oauth provider"):
        client._client_id(desc)


def test_client_secret_without_env_field_raises_runtime():
    desc = make_oauth_descriptor()
    object.__setattr__(desc, "client_secret_env", None)
    with pytest.raises(RuntimeError, match="not an oauth"):
        client._client_secret(desc)


# -- _error_detail -----------------------------------------------------------


def test_error_detail_non_json():
    # FakeHttpResponse is a minimal stand-in; httpx.Response is a concrete class,
    # so pyright cannot accept it structurally — cast at the _error_detail seam.
    resp = FakeHttpResponse(status_code=500, raise_on_json=True, content=b"oops")
    assert "non_json" in client._error_detail(cast(httpx.Response, resp))


def test_error_detail_non_object():
    resp = FakeHttpResponse(json_body=[1, 2], content=b"[1,2]")
    assert "non_object" in client._error_detail(cast(httpx.Response, resp))


def test_error_detail_truncates_long_description():
    long_desc = "x" * 200
    resp = FakeHttpResponse(json_body={"error": "bad", "error_description": long_desc})
    detail = client._error_detail(cast(httpx.Response, resp))
    assert "bad" in detail
    assert "…" in detail


def test_error_detail_error_only():
    resp = FakeHttpResponse(json_body={"error": "bad"})
    assert client._error_detail(cast(httpx.Response, resp)) == "error='bad'"


def test_error_detail_no_error_field():
    resp = FakeHttpResponse(json_body={"other": 1})
    assert client._error_detail(cast(httpx.Response, resp)) == "no_error_field"


# -- exchange_code -----------------------------------------------------------


async def test_exchange_code_success(oauth_client_env, install_http):
    install_http(
        [
            FakeHttpResponse(
                json_body={
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_in": 7200,
                    "scope": "mail.read mail.send",
                }
            ),
        ]
    )
    desc = make_oauth_descriptor()
    resp = await client.exchange_code(
        descriptor=desc,
        code="auth-code",
        code_verifier="v",
        redirect_uri="https://app.example.com/cb",
    )
    assert resp.access_token == "at"
    assert resp.refresh_token == "rt"
    assert resp.granted_scopes == ["mail.read", "mail.send"]


async def test_exchange_code_transport_error(oauth_client_env, install_http):
    install_http([httpx.ConnectError("boom")])
    with pytest.raises(CodeExchangeFailedError, match="transport error"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_non_200(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=400, json_body={"error": "invalid_grant"})])
    with pytest.raises(CodeExchangeFailedError, match="status 400"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_non_json_200(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=200, raise_on_json=True, content=b"notjson")])
    with pytest.raises(CodeExchangeFailedError, match="non-JSON"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_missing_refresh_token(oauth_client_env, install_http):
    install_http([FakeHttpResponse(json_body={"access_token": "at", "expires_in": 3600})])
    with pytest.raises(RefreshTokenMissingError):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_missing_access_token(oauth_client_env, install_http):
    install_http([FakeHttpResponse(json_body={"expires_in": 3600})])
    with pytest.raises(CodeExchangeFailedError, match="missing access_token"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_non_string_access_token_raises(oauth_client_env, install_http):
    # A non-string access_token is a malformed body, classified rather than
    # escaping as a raw 500.
    install_http([FakeHttpResponse(json_body={"access_token": 12345, "refresh_token": "rt", "expires_in": 3600})])
    with pytest.raises(CodeExchangeFailedError, match="non-string access_token"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_list_scope_raises_classified(oauth_client_env, install_http):
    # A list-typed scope would AttributeError on ``.split``; it must surface as the
    # documented CodeExchangeFailedError instead.
    install_http(
        [
            FakeHttpResponse(
                json_body={"access_token": "at", "refresh_token": "rt", "expires_in": 3600, "scope": ["mail.read"]}
            )
        ]
    )
    with pytest.raises(CodeExchangeFailedError, match="non-string scope"):
        await client.exchange_code(
            descriptor=make_oauth_descriptor(),
            code="c",
            code_verifier="v",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_allows_missing_refresh_when_not_required(oauth_client_env, install_http):
    install_http([FakeHttpResponse(json_body={"access_token": "at", "expires_in": 3600})])
    resp = await client.exchange_code(
        descriptor=make_oauth_descriptor(),
        code="c",
        code_verifier="v",
        redirect_uri="https://app.example.com/cb",
        require_refresh_token=False,
    )
    assert resp.refresh_token is None


# -- refresh -----------------------------------------------------------------


async def test_refresh_success(oauth_client_env, install_http):
    install_http(
        [
            FakeHttpResponse(
                json_body={
                    "access_token": "new-at",
                    "refresh_token": "new-rt",
                    "expires_in": 3600,
                    "scope": "mail.read",
                }
            )
        ]
    )
    resp = await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="old-rt")
    assert resp.access_token == "new-at"
    assert resp.refresh_token == "new-rt"


async def test_refresh_keeps_old_refresh_token_when_omitted(oauth_client_env, install_http):
    install_http([FakeHttpResponse(json_body={"access_token": "new-at", "expires_in": 3600})])
    resp = await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="old-rt")
    assert resp.refresh_token == "old-rt"


async def test_refresh_transport_error_is_transient(oauth_client_env, install_http):
    install_http([httpx.ConnectError("boom")])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"


async def test_refresh_5xx_is_transient(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=503)])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"
    assert ei.value.http_status == 503


async def test_refresh_invalid_grant_is_terminal(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=400, json_body={"error": "invalid_grant"})])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "invalid_grant"


async def test_refresh_other_4xx_is_transient(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=429, json_body={"error": "slow_down"})])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"


async def test_refresh_non_json_4xx_is_transient(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=400, raise_on_json=True, content=b"x")])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"


async def test_refresh_non_json_200_is_transient(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=200, raise_on_json=True, content=b"x")])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"


async def test_refresh_list_scope_is_reclassified_transient(oauth_client_env, install_http):
    # The shared parser raises CodeExchangeFailedError on a malformed 200 body;
    # refresh re-maps it to its own contract so it never escapes unclassified.
    install_http([FakeHttpResponse(json_body={"access_token": "at", "expires_in": 3600, "scope": ["mail.read"]})])
    with pytest.raises(TokenRefreshFailedError) as ei:
        await client.refresh(descriptor=make_oauth_descriptor(), refresh_token="rt")
    assert ei.value.reason == "transient"


# -- revoke ------------------------------------------------------------------


async def test_revoke_skipped_when_no_endpoint(oauth_client_env):
    desc = make_oauth_descriptor(revoke=None)
    outcome = await client.revoke(descriptor=desc, token="t")
    assert outcome.outcome == "skipped"


async def test_revoke_success(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=200)])
    outcome = await client.revoke(descriptor=make_oauth_descriptor(), token="t")
    assert outcome.outcome == "success"
    assert outcome.http_status == 200


async def test_revoke_non_2xx_is_failed(oauth_client_env, install_http):
    install_http([FakeHttpResponse(status_code=400, json_body={"error": "bad"})])
    outcome = await client.revoke(descriptor=make_oauth_descriptor(), token="t")
    assert outcome.outcome == "failed"
    assert outcome.http_status == 400


async def test_revoke_transport_error_is_failed(oauth_client_env, install_http):
    install_http([httpx.ConnectError("boom")])
    outcome = await client.revoke(descriptor=make_oauth_descriptor(), token="t")
    assert outcome.outcome == "failed"
