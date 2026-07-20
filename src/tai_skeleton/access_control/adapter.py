import logging

from fastmcp.server.auth import AccessToken, TokenVerifier
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from tai_contract.access_control.identity import IdentityProvider
from tai_contract.access_control.registry import get_identity_provider_factory

from tai_skeleton.access_control.backend import AccessControlAuthBackend, AuthorizationError
from tai_skeleton.access_control.middleware import ResourceGuardMiddleware
from tai_skeleton.access_control.roles import SkeletonAccountsAdminServices
from tai_skeleton.access_control.settings import AccessControlSettings
from tai_skeleton.access_control.verifier import AccessControlVerifier

logger = logging.getLogger(__name__)


def handle_auth_error(conn: HTTPConnection, exc: Exception) -> JSONResponse:
    # Log the full failure server-side (loud), but never surface the exception
    # string to the caller — it can carry internal detail (redis host:port, jq
    # internals). An authenticated-but-denied caller (AuthorizationError) gets a
    # 403; an authentication failure gets a 401. Both bodies are fixed and generic.
    logger.error("Authentication middleware error: %s", exc, exc_info=exc)
    if isinstance(exc, AuthorizationError):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


class AuthAdapter(TokenVerifier):
    def __init__(self, settings: AccessControlSettings):
        super().__init__()
        self.settings = settings

        # Install the application's AccountsAdminServices onto the settings' runtime
        # slot BEFORE any accounts-provider factory receives the object. This is the
        # one place the settings are handed to provider factories, so every downstream
        # factory — the verifier chain, the login aggregator/logout dispatcher, and any
        # accounts-provider factory — reaches these services as ``settings.admin`` (the
        # AccountsProviderSettings Protocol), never by importing the application.
        settings.admin = SkeletonAccountsAdminServices()

        # Provider resolution is DEFERRED, not done here. AuthAdapter is built in
        # build_app(), which runs BEFORE start() imports the manifest's identity
        # plugin modules that register the providers in the module-level registry.
        # An eager registry lookup at construction would therefore raise "unknown
        # provider" on every boot. The verifier resolves the providers through
        # _get_identity_providers on the FIRST verify_token — by then start() has
        # populated the registry — and memoizes them.
        self._internal_verifier = AccessControlVerifier(settings, provider_factories=self._get_identity_providers)

        self._middleware_stack = self._get_access_control_middleware()

    def _get_identity_providers(self) -> list[IdentityProvider]:
        # Resolve EVERY configured provider through the module-level identity-provider
        # registry, which the manifest's identity plugins populate at import, preserving
        # the configured order (the verifier tries them in turn). An unknown name raises
        # LOUDLY here (surfaced as a fail-closed deny by the backend's error handling),
        # never a boot crash and never a silent allow.
        return [get_identity_provider_factory(name)(self.settings) for name in self.settings.auth_providers]

    def _get_access_control_middleware(self) -> list[Middleware]:
        if not self.settings.enable:
            return []

        return [
            # 1. Identity & Policy (Reads cached Redis data)
            Middleware(
                AuthenticationMiddleware,
                backend=AccessControlAuthBackend(self._internal_verifier, self.settings),
                on_error=handle_auth_error,  # Renders denials as generic 401/403, hiding internal detail
            ),
            # 2. Context Helper (Populates request.user / request.auth)
            Middleware(AuthContextMiddleware),
            # 3. Route Authorization (Resource Guard)
            Middleware(
                ResourceGuardMiddleware,
                verifier=self._internal_verifier,
                public_resource_id=self.settings.public_resource_id,
                authenticated_always_allowed_paths=self.settings.authenticated_always_allowed_paths,
            ),
        ]

    async def verify_token(self, token: str) -> AccessToken | None:
        return await self._internal_verifier.verify_token(token)

    def get_middleware(self) -> list[Middleware]:
        return self._middleware_stack
