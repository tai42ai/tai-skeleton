"""OAuth 2.0 + PKCE primitives for Connectors (stdlib + pooled HTTP, no
3rd-party OAuth lib).

Covers PKCE pair generation, authorize-URL construction with redirect-URI
allow-list enforcement, code/refresh token exchange, and revocation. Every
failure path logs once at WARNING/ERROR before raising a typed exception.

Outbound HTTP goes through the app-pooled :class:`HttpxClient` so the OAuth
session/connection pool is shared and torn down centrally at app shutdown.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse

import httpx
from tai42_contract.connectors.errors import OperatorMisconfiguredError
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.http import HttpxClient

from tai42_skeleton.connectors.settings import connector_engine_config

if TYPE_CHECKING:
    from tai42_contract.connectors.models import UpstreamRevokeOutcome
    from tai42_contract.connectors.providers import OAuthEndpoints, ProviderDescriptor

logger = logging.getLogger(__name__)

_OAUTH_HTTP_TIMEOUT_SECONDS = 20.0

# Free-form ``error_description`` (RFC 6749) can echo the auth code or token;
# truncate before logging so a misbehaving provider cannot leak one.
_ERROR_DESCRIPTION_MAX_LEN = 64


def _http() -> AbstractAsyncContextManager[httpx.AsyncClient]:
    """The app-pooled OAuth HTTP client (one TLS session-cache + connection pool
    shared across all OAuth calls, closed centrally at shutdown)."""
    return client_ctx(HttpxClient, timeout=_OAUTH_HTTP_TIMEOUT_SECONDS)


def _error_detail(resp: httpx.Response) -> str:
    """Log-safe summary of a non-2xx OAuth body: only ``error`` + a truncated
    ``error_description``; everything else is dropped to avoid leaking tokens."""
    try:
        payload = resp.json()
    except Exception:
        return f"non_json body_len={len(resp.content)}"
    if not isinstance(payload, dict):
        return f"non_object body_len={len(resp.content)}"
    error = payload.get("error")
    desc = payload.get("error_description")
    if isinstance(desc, str) and len(desc) > _ERROR_DESCRIPTION_MAX_LEN:
        desc = desc[:_ERROR_DESCRIPTION_MAX_LEN] + "…"
    if isinstance(error, str) and isinstance(desc, str):
        return f"error={error!r} error_description={desc!r}"
    if isinstance(error, str):
        return f"error={error!r}"
    return "no_error_field"


# -- Operator-supplied OAuth client credentials -------------------------------


def _required_env(env_var: str, *, provider_id: str) -> str:
    """Read a required operator-supplied env var, or raise loudly.

    Client credentials are operator-supplied at the API process environment, not
    baked into the descriptor. An unset/empty value raises
    :class:`OperatorMisconfiguredError` (carrying the offending env-var name)
    rather than returning a silent default.
    """
    value = os.environ.get(env_var, "")
    if not value:
        raise OperatorMisconfiguredError(env_var=env_var, provider_id=provider_id)
    return value


def _client_id(descriptor: ProviderDescriptor) -> str:
    if not descriptor.client_id_env:
        raise RuntimeError(f"provider {descriptor.id!r} has no client_id_env (not an oauth provider)")
    return _required_env(descriptor.client_id_env, provider_id=descriptor.id)


def _client_secret(descriptor: ProviderDescriptor) -> str:
    if not descriptor.client_secret_env:
        raise RuntimeError(f"provider {descriptor.id!r} has no client_secret_env (not an oauth provider)")
    return _required_env(descriptor.client_secret_env, provider_id=descriptor.id)


def _oauth_endpoints(descriptor: ProviderDescriptor) -> OAuthEndpoints:
    if descriptor.oauth is None:
        raise RuntimeError(f"provider {descriptor.id!r} has no oauth endpoints (not an oauth provider)")
    return descriptor.oauth


# -- Errors -------------------------------------------------------------------


class OAuthError(RuntimeError):
    """Base class for connector OAuth failures."""


class RedirectUriNotAllowedError(OAuthError):
    """The configured/computed redirect URI is not on the allow-list."""


class CodeExchangeFailedError(OAuthError):
    """Provider rejected the authorization-code → token exchange."""


class RefreshTokenMissingError(OAuthError):
    """Provider returned no refresh_token on first consent."""


class TokenRefreshFailedError(OAuthError):
    """Provider rejected the refresh_token grant.

    ``reason`` is ``"invalid_grant"`` when the refresh token is revoked/expired
    (→ Reconnect required) or ``"transient"`` for network/5xx errors to retry.
    The resolver uses ``reason`` to dispatch the right state transition.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        http_status: int | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.http_status = http_status


# -- PKCE --------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` using RFC 7636 S256.

    ``token_urlsafe(64)`` yields an 86-char verifier, inside the 43-128 range
    RFC 7636 §4.1 requires, so no length clamping is needed.
    """
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


# -- Redirect-URI allow-list -------------------------------------------------


def _is_local_host(host: str) -> bool:
    # Literal-only set so an attacker-registered ``evil.local`` cannot match.
    return host in {"localhost", "127.0.0.1", "::1"}


def validate_redirect_uri(redirect_uri: str) -> str:
    """Validate a redirect URI against the configured allow-list.

    Requires an exact origin match and https:// for any non-localhost host.
    Returns the normalised URI; raises :class:`RedirectUriNotAllowedError` on any
    rule violation (logged redacted to scheme+host).
    """
    if not redirect_uri:
        raise RedirectUriNotAllowedError("redirect_uri must be non-empty")

    parsed = urlparse(redirect_uri)
    if not parsed.scheme or not parsed.netloc:
        logger.warning(
            "connectors: redirect_uri rejected (unparseable): %s://%s",
            parsed.scheme,
            parsed.netloc,
        )
        raise RedirectUriNotAllowedError("redirect_uri must include scheme and host")

    host = parsed.hostname or ""
    if parsed.scheme != "https" and not _is_local_host(host):
        logger.warning(
            "connectors: redirect_uri rejected (non-https for non-local host): %s://%s",
            parsed.scheme,
            host,
        )
        raise RedirectUriNotAllowedError("redirect_uri must use https:// for non-local hosts")

    allowlist = connector_engine_config().redirect_uri_allowlist_origins
    if not allowlist:
        logger.warning(
            "connectors: redirect_uri rejected because CONNECTORS_REDIRECT_URI_ALLOWLIST is unset",
        )
        raise RedirectUriNotAllowedError(
            "CONNECTORS_REDIRECT_URI_ALLOWLIST env var is empty — operator must "
            "configure the allow-list before any Connect flow can start"
        )

    # Exact origin (scheme://host[:port]) match; path is ignored because the
    # callback page is served on a fixed path the deployment narrows.
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in allowlist:
        logger.warning(
            "connectors: redirect_uri rejected (origin %s not in allow-list)",
            origin,
        )
        raise RedirectUriNotAllowedError(f"redirect_uri origin {origin!r} is not in the allow-list")

    # Returned unchanged: OAuth requires the redirect_uri to be byte-identical
    # across authorize and token exchange, so this must not normalize (e.g. strip
    # a trailing slash) — every caller uses the exact validated string.
    return redirect_uri


# -- Authorize URL -----------------------------------------------------------


def build_authorize_url(
    *,
    descriptor: ProviderDescriptor,
    scopes: list[str],
    state: str,
    code_challenge: str,
    redirect_uri: str,
) -> str:
    """Construct the provider's authorize URL with PKCE, enforcing the allow-list."""
    validate_redirect_uri(redirect_uri)

    params = {
        "response_type": "code",
        "client_id": _client_id(descriptor),
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    params.update(descriptor.extra_authorize_params)
    return f"{_oauth_endpoints(descriptor).authorize}?{urlencode(params)}"


# -- Token responses ---------------------------------------------------------


@dataclass(frozen=True)
class TokenResponse:
    access_token: str
    refresh_token: str | None
    expires_at: datetime
    granted_scopes: list[str]
    raw: dict


def _parse_token_response(payload: dict, *, fallback_refresh_token: str | None = None) -> TokenResponse:
    access = payload.get("access_token")
    if not access:
        raise CodeExchangeFailedError("provider response is missing access_token")
    if not isinstance(access, str):
        raise CodeExchangeFailedError(f"provider returned a non-string access_token (got {type(access).__name__})")
    raw_expires_in = payload.get("expires_in", 3600)
    try:
        expires_in = int(raw_expires_in)
    except (TypeError, ValueError) as exc:
        raise CodeExchangeFailedError(f"provider returned a non-numeric expires_in: {raw_expires_in!r}") from exc
    expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)
    refresh = payload.get("refresh_token") or fallback_refresh_token
    # A missing/null scope is tolerated (empty grant list); a present-but-wrong
    # type (e.g. a list) is a malformed body, not a silent AttributeError.
    raw_scope = payload.get("scope")
    if raw_scope is None:
        granted: list[str] = []
    elif isinstance(raw_scope, str):
        granted = [s for s in raw_scope.split(" ") if s]
    else:
        raise CodeExchangeFailedError(f"provider returned a non-string scope (got {type(raw_scope).__name__})")
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        granted_scopes=granted,
        raw=payload,
    )


# -- Code → token exchange ---------------------------------------------------


async def exchange_code(
    *,
    descriptor: ProviderDescriptor,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    require_refresh_token: bool = True,
) -> TokenResponse:
    """Exchange an authorization code for tokens.

    ``require_refresh_token``: the first Connect must return a refresh token;
    if it doesn't we abort rather than persist a half-broken record.
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": _client_id(descriptor),
        "client_secret": _client_secret(descriptor),
        "code_verifier": code_verifier,
    }
    try:
        async with _http() as http:
            resp = await http.post(_oauth_endpoints(descriptor).token, data=body)
    except httpx.HTTPError as exc:
        logger.error(
            "connectors: token exchange transport error (%s)",
            descriptor.id,
            exc_info=True,
        )
        raise CodeExchangeFailedError(f"transport error: {exc}") from exc

    if resp.status_code != 200:
        logger.warning(
            "connectors: token exchange failed (%s) status=%s detail=%s",
            descriptor.id,
            resp.status_code,
            _error_detail(resp),
        )
        raise CodeExchangeFailedError(f"provider returned status {resp.status_code}")

    try:
        raw = resp.json()
    except ValueError as exc:
        logger.warning(
            "connectors: token exchange returned non-JSON 200 body (%s)",
            descriptor.id,
            exc_info=True,
        )
        raise CodeExchangeFailedError("provider returned a non-JSON body") from exc

    parsed = _parse_token_response(raw)
    if require_refresh_token and not parsed.refresh_token:
        logger.warning(
            "connectors: provider %s returned no refresh_token on first consent",
            descriptor.id,
        )
        raise RefreshTokenMissingError("provider returned no refresh_token; aborting Connect")
    return parsed


# -- Refresh -----------------------------------------------------------------


async def refresh(*, descriptor: ProviderDescriptor, refresh_token: str) -> TokenResponse:
    """Exchange a refresh_token for a fresh access token.

    Raises :class:`TokenRefreshFailedError` on every error, with ``reason`` set to
    ``"invalid_grant"`` on revocation/expiry or ``"transient"`` on network/5xx
    so the caller can dispatch to the right state transition.
    """
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": _client_id(descriptor),
        "client_secret": _client_secret(descriptor),
    }
    try:
        async with _http() as http:
            resp = await http.post(_oauth_endpoints(descriptor).token, data=body)
    except httpx.HTTPError as exc:
        logger.warning(
            "connectors: refresh transport error (%s)",
            descriptor.id,
            exc_info=True,
        )
        raise TokenRefreshFailedError(f"transport error: {exc}", reason="transient") from exc

    if 500 <= resp.status_code < 600:
        logger.warning(
            "connectors: refresh transient failure (%s) status=%s",
            descriptor.id,
            resp.status_code,
        )
        raise TokenRefreshFailedError(
            f"provider returned status {resp.status_code}",
            reason="transient",
            http_status=resp.status_code,
        )

    if resp.status_code != 200:
        # RFC 6749 §5.2: only ``error == "invalid_grant"`` means the refresh
        # token itself is dead (terminal → Reconnect). Everything else (401,
        # 403, 429, other 400s) is operational — back off and retry.
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        err = payload.get("error", "")
        reason = "invalid_grant" if err == "invalid_grant" else "transient"
        # Every invalid_grant is terminal → RECONNECT_REQUIRED. Correct for the
        # shipped non-rotating-token provider, where a refresh token stays valid
        # until the user revokes it.
        logger.warning(
            "connectors: refresh %s (%s) status=%s error=%r",
            reason,
            descriptor.id,
            resp.status_code,
            err,
        )
        raise TokenRefreshFailedError(
            f"provider returned status {resp.status_code} error={err!r}",
            reason=reason,
            http_status=resp.status_code,
        )

    # Some providers (Google) omit refresh_token on refresh — keep the old one.
    try:
        raw = resp.json()
    except ValueError as exc:
        logger.warning(
            "connectors: refresh returned non-JSON 200 body (%s)",
            descriptor.id,
            exc_info=True,
        )
        raise TokenRefreshFailedError(
            "provider returned a non-JSON body",
            reason="transient",
        ) from exc
    try:
        return _parse_token_response(raw, fallback_refresh_token=refresh_token)
    except CodeExchangeFailedError as exc:
        # ``_parse_token_response`` is shared with the exchange path and raises
        # CodeExchangeFailedError on a malformed body; re-map it to the refresh path's
        # single documented error contract so it never escapes unclassified.
        logger.warning(
            "connectors: refresh returned a malformed 200 body (%s)",
            descriptor.id,
            exc_info=True,
        )
        raise TokenRefreshFailedError(str(exc), reason="transient") from exc


# -- Revoke ------------------------------------------------------------------


@dataclass(frozen=True)
class RevokeOutcome:
    outcome: UpstreamRevokeOutcome
    http_status: int | None = None


async def revoke(*, descriptor: ProviderDescriptor, token: str) -> RevokeOutcome:
    """Best-effort upstream token revocation. NEVER raises — the caller proceeds
    with the local purge regardless of outcome."""
    if descriptor.oauth is None or not descriptor.oauth.revoke:
        return RevokeOutcome(outcome="skipped")

    try:
        async with _http() as http:
            resp = await http.post(
                descriptor.oauth.revoke,
                data={"token": token},
                timeout=_OAUTH_HTTP_TIMEOUT_SECONDS,
            )
    except httpx.HTTPError:
        logger.warning(
            "connectors: revoke transport error (%s)",
            descriptor.id,
            exc_info=True,
        )
        return RevokeOutcome(outcome="failed")

    if 200 <= resp.status_code < 300:
        return RevokeOutcome(outcome="success", http_status=resp.status_code)
    logger.warning(
        "connectors: revoke non-2xx (%s) status=%s detail=%s",
        descriptor.id,
        resp.status_code,
        _error_detail(resp),
    )
    return RevokeOutcome(outcome="failed", http_status=resp.status_code)
