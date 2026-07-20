"""Pure helpers for composing the OAuth callback redirect URI. No import-time
side effects (unlike the router): settings are read at call time, so it is safe
to import from anywhere."""

from __future__ import annotations

import logging

from starlette.requests import Request

from tai42_skeleton.connectors.oauth.client import RedirectUriNotAllowedError
from tai42_skeleton.connectors.settings import connector_engine_config

logger = logging.getLogger(__name__)

# Static chromeless callback page registered with the OAuth provider. The
# provider redirects the popup back to this path — on the originating deployment
# (single-deployment default) or on the central bridge origin when one is
# configured. The page reads the signed ``state`` and hands the code onward.
CALLBACK_PATH = "/oauth-bridge.html"


def _resolve_origin(request: Request) -> str:
    """This deployment's own origin, from the browser Origin header (single-origin
    dev) then ``request.base_url`` (direct-API callers; needs uvicorn
    ``--proxy-headers`` behind a TLS proxy). Read at call time."""
    origin = (request.headers.get("origin") or "").strip().rstrip("/")
    if origin and origin.startswith(("http://", "https://")):
        return origin
    return str(request.base_url).rstrip("/")


def compute_deployment_origin(request: Request) -> str:
    """The originating deployment's own origin — signed into the OAuth ``state``
    so a callback routed through the central bridge knows where to bounce the code
    back. Always this deployment, never the bridge override."""
    return _resolve_origin(request)


def validate_origin_allowed(origin: str) -> str:
    """Fail-closed check that ``origin`` is on the operator's redirect allow-list.

    The origin is signed into the OAuth ``state`` and a central bridge trusts it
    to bounce the auth code back, so an off-list origin (e.g. a spoofed ``Origin``
    header) must never be signed. Mirrors ``validate_redirect_uri``'s rule; raises
    :class:`RedirectUriNotAllowedError` when the origin is absent from the list."""
    allowlist = connector_engine_config().redirect_uri_allowlist_origins
    if origin not in allowlist:
        logger.warning("connectors: origin rejected (not in redirect allow-list)")
        raise RedirectUriNotAllowedError(f"origin {origin!r} is not in the redirect_uri allow-list")
    return origin


def compute_redirect_uri(request: Request) -> str:
    """Compose the absolute callback URL the operator must allow-list — the URL the
    provider redirects to.

    With ``CONNECTORS_OAUTH_BRIDGE_URL`` set, the provider redirects to that shared
    bridge origin instead of this deployment; the bridge then bounces the code back
    to the origin signed into ``state``. Unset, the provider redirects here
    directly — a single deployment needs no central bridge. The chosen origin must
    pass ``validate_redirect_uri``'s allow-list either way.
    """
    bridge = connector_engine_config().oauth_bridge_url
    base = bridge.strip().rstrip("/") if bridge else _resolve_origin(request)
    return f"{base}{CALLBACK_PATH}"
