"""OAuth callback redirect-URI composition + the central-bridge override."""

from __future__ import annotations

from typing import cast

import pytest
from starlette.requests import Request

import tai42_skeleton.connectors.oauth.redirect as redirect_mod
from tai42_skeleton.connectors.oauth.client import RedirectUriNotAllowedError
from tai42_skeleton.connectors.oauth.redirect import (
    CALLBACK_PATH,
    compute_deployment_origin,
    compute_redirect_uri,
    validate_origin_allowed,
)


class _FakeRequest:
    def __init__(self, *, origin: str | None, base_url: str) -> None:
        self.headers = {"origin": origin} if origin is not None else {}
        self.base_url = base_url


class _FakeEngineConfig:
    def __init__(
        self,
        oauth_bridge_url: str | None = None,
        redirect_uri_allowlist_origins: list[str] | None = None,
    ) -> None:
        self.oauth_bridge_url = oauth_bridge_url
        self.redirect_uri_allowlist_origins = (
            redirect_uri_allowlist_origins
            if redirect_uri_allowlist_origins is not None
            else ["https://app.example.com"]
        )


@pytest.fixture
def engine_cfg(monkeypatch):
    """Isolate the redirect helpers from the process-global settings cache: the
    override starts unset and a test flips it via ``cfg.oauth_bridge_url``."""
    cfg = _FakeEngineConfig()
    monkeypatch.setattr(redirect_mod, "connector_engine_config", lambda: cfg)
    return cfg


# compute_redirect_uri reads request.headers.get("origin"), request.base_url, and
# connector_engine_config().oauth_bridge_url. starlette Request is a concrete
# class (not a Protocol), so the structural stand-in is cast to it.
def test_origin_header_preferred(engine_cfg):
    req = _FakeRequest(origin="https://app.example.com", base_url="https://api.x/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://app.example.com{CALLBACK_PATH}"


def test_origin_trailing_slash_stripped(engine_cfg):
    req = _FakeRequest(origin="https://app.example.com/", base_url="https://api.x/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://app.example.com{CALLBACK_PATH}"


def test_falls_back_to_base_url_when_no_origin(engine_cfg):
    req = _FakeRequest(origin=None, base_url="https://api.example.com/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://api.example.com{CALLBACK_PATH}"


def test_falls_back_when_origin_not_http_scheme(engine_cfg):
    """A non-http(s) Origin (e.g. a file:// or null origin) is ignored."""
    req = _FakeRequest(origin="null", base_url="https://api.example.com/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://api.example.com{CALLBACK_PATH}"


def test_falls_back_when_origin_blank(engine_cfg):
    req = _FakeRequest(origin="   ", base_url="https://api.example.com/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://api.example.com{CALLBACK_PATH}"


# -- Central-bridge override -------------------------------------------------


def test_bridge_override_used_as_redirect_uri_when_set(engine_cfg):
    """With the bridge configured, the provider redirect points at the bridge
    origin — NOT this deployment's request origin."""
    engine_cfg.oauth_bridge_url = "https://bridge.tai42.ai"
    req = _FakeRequest(origin="https://app.example.com", base_url="https://api.x/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://bridge.tai42.ai{CALLBACK_PATH}"


def test_bridge_override_trailing_slash_stripped(engine_cfg):
    engine_cfg.oauth_bridge_url = "https://bridge.tai42.ai/"
    req = _FakeRequest(origin="https://app.example.com", base_url="https://api.x/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://bridge.tai42.ai{CALLBACK_PATH}"


def test_unset_override_preserves_origin_behavior(engine_cfg):
    """An empty override is treated as unset — the request origin still wins."""
    engine_cfg.oauth_bridge_url = ""
    req = _FakeRequest(origin="https://app.example.com", base_url="https://api.x/")
    assert compute_redirect_uri(cast(Request, req)) == f"https://app.example.com{CALLBACK_PATH}"


# -- Deployment origin (signed into state) -----------------------------------


def test_deployment_origin_is_this_deployment_not_the_bridge(engine_cfg):
    """The origin signed into state is always this deployment's own origin, even
    when the provider redirect is bounced through the bridge."""
    engine_cfg.oauth_bridge_url = "https://bridge.tai42.ai"
    req = _FakeRequest(origin="https://app.example.com", base_url="https://api.x/")
    assert compute_deployment_origin(cast(Request, req)) == "https://app.example.com"


def test_deployment_origin_falls_back_to_base_url(engine_cfg):
    req = _FakeRequest(origin=None, base_url="https://api.example.com/")
    assert compute_deployment_origin(cast(Request, req)) == "https://api.example.com"


# -- Origin allow-list validation (fail-closed) ------------------------------


def test_validate_origin_allowed_passes_for_listed_origin(engine_cfg):
    engine_cfg.redirect_uri_allowlist_origins = ["https://app.example.com", "http://localhost:5173"]
    assert validate_origin_allowed("https://app.example.com") == "https://app.example.com"


def test_validate_origin_allowed_rejects_spoofed_origin(engine_cfg):
    """A spoofed Origin header (bridge mode) must not be signed into state — an
    off-list origin is rejected fail-closed before the flow can start."""
    engine_cfg.redirect_uri_allowlist_origins = ["https://app.example.com"]
    with pytest.raises(RedirectUriNotAllowedError, match="not in the redirect_uri allow-list"):
        validate_origin_allowed("https://evil.com")


def test_validate_origin_allowed_rejects_when_allowlist_empty(engine_cfg):
    engine_cfg.redirect_uri_allowlist_origins = []
    with pytest.raises(RedirectUriNotAllowedError):
        validate_origin_allowed("https://app.example.com")
