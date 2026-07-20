"""Login/logout operations fanning out over the accounts-provider registry.

- ``login_methods`` aggregates every registered accounts provider's declared
  ``LoginMethod`` metadata plus a bootstrap flag, so a generic login screen can
  render without knowing which providers are installed.
- ``logout`` revokes the caller's session by fanning ``revoke_session`` out over
  every presented credential candidate and every registered accounts provider;
  the first ``True`` wins, and no match is a loud 404 (:class:`NotFoundError`).
- ``exchange_claim_token`` burns a one-time claim token and returns the raw API key it
  carried — the public QR-onboarding exchange leg, ``authority_changing`` so it never
  projects as an MCP tool.

Provider factories are instantiated per call and NOT cached: the login fetch is
low-frequency, and ``needs_bootstrap`` must be FRESH so the create-owner screen
disappears the moment the owner exists. Provider errors propagate (loud, never a
silently empty methods list or a silent logout no-op).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from tai_contract.access_control import get_current_user_id
from tai_contract.accounts import iter_accounts_provider_factories

from tai_skeleton.access_control.claim_links import ClaimLinkError
from tai_skeleton.access_control.claim_links import exchange_claim_token as _exchange_claim_token
from tai_skeleton.access_control.settings import access_control_settings
from tai_skeleton.operations import NotFoundError, operation

logger = logging.getLogger(__name__)


class ClaimExchange(BaseModel):
    """Exchange a one-time claim token for its API key."""

    token: str = Field(min_length=1)


@operation(summary="List available login methods", tags=["login"])
async def login_methods() -> dict:
    """Aggregate every registered accounts provider's login methods + bootstrap flag.

    ``authed=False`` is OpenAPI truth-telling only; runtime public-ness comes from the
    always-public ``/api/login`` prefix. Each method is serialized with
    ``model_dump(exclude_none=True)`` so a ``None``-valued optional (icon/autocomplete)
    is OMITTED, never ``null`` (the Studio's zod schemas accept absent but reject
    ``null``). An empty registry yields ``{"methods": [], "bootstrap": false}``.
    Provider errors propagate (loud, never a silently empty methods list)."""
    settings = access_control_settings()
    methods: list[dict] = []
    bootstrap = False
    for _name, factory in iter_accounts_provider_factories():
        provider = factory(settings)
        for method in provider.login_methods():
            methods.append(method.model_dump(exclude_none=True))
        if await provider.needs_bootstrap():
            bootstrap = True
    return {"methods": methods, "bootstrap": bootstrap}


@operation(
    summary="Exchange a one-time claim token for its API key",
    tags=["login"],
    authority_changing=True,
    errors=[NotFoundError],
    request_model=ClaimExchange,
)
async def exchange_claim_token(token: str) -> dict:
    """Burn a one-time claim token and return the raw API key it carried — the public
    exchange leg of QR onboarding (``authed=False``; runtime public-ness comes from the
    always-public ``/api/login`` prefix).

    The claim record is single-use: a used / unknown / expired token all answer the SAME
    404 (no oracle distinguishing them). The handed-out key is guaranteed not-REVOKED
    (re-validated at exchange), NOT owner-alive — owner-death is enforced by the gate on
    every request. Response mirrors the ``loginResult`` wire shape:
    ``{"data": {"token": <raw key>, "user_id": ...}}``.

    ``authority_changing=True`` keeps this OFF the default MCP tool surface: a
    credential-exchange login door is not an agent tool (it sits outside the
    ``/api/auth/*`` prefix, so the flag is what excludes it, not the prefix)."""
    try:
        return await _exchange_claim_token(token)
    except ClaimLinkError as exc:
        # The store's only exchange failure is the uniform 404 miss.
        raise NotFoundError(exc.message) from exc


@operation(
    summary="Log out the current session",
    tags=["access-control"],
    destructive=True,
    errors=[NotFoundError],
)
async def logout(candidates: list[str]) -> dict:
    """Revoke the caller's session by fanning ``revoke_session`` out over every
    presented credential candidate and every registered accounts provider.

    Iterating all candidates is required: a client may present a stale value in
    ``Authorization`` alongside its live session in ``X-Api-Key``, so checking only the
    first would leave that session alive. The FIRST ``True`` → ``{"revoked": true}``.
    Only when EVERY candidate against EVERY provider returns ``False`` → 404 (an
    ``sk-`` API key cannot "log out" — loud, not a silent no-op), with a server-side
    log naming the caller so probing/replay stays visible. Provider errors propagate
    (fail closed)."""
    settings = access_control_settings()
    providers = [factory(settings) for _name, factory in iter_accounts_provider_factories()]

    # Tokens outer, providers inner (registry order): the first provider to own any
    # presented candidate revokes it and wins.
    for token in candidates:
        for provider in providers:
            if await provider.revoke_session(token):
                return {"revoked": True}

    logger.info(
        "access_control: logout by %s matched no accounts provider (presented credential is not a revocable session)",
        get_current_user_id(),
    )
    raise NotFoundError("Not a revocable session")
