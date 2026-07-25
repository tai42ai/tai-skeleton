"""The public login-methods aggregator and the authed logout dispatcher.

Two doors fanning out over the accounts-provider registry:

- ``GET /api/login/methods`` — PUBLIC (the always-public ``/api/login`` prefix makes
  it reachable pre-auth with no route rows). Aggregates every registered accounts
  provider's declared ``LoginMethod`` metadata plus a bootstrap flag, so a generic
  login screen can render without knowing which providers are installed. An empty
  registry answers ``{"methods": [], "bootstrap": false}`` — the Studio's
  key-paste-only signal.
- ``POST /api/login/claim`` — PUBLIC (same always-public prefix). Burns a one-time
  claim token and returns the raw API key it carried — the QR-onboarding exchange leg.
- ``POST /api/auth/logout`` — the single AUTHED logout dispatcher (logout is
  application surface so two installed accounts plugins never race to own the route).
  It fans out ``revoke_session`` over every presented credential candidate and every
  registered accounts provider; the first ``True`` wins.

The doors are thin adapters over operations in
``tai42_skeleton.operations.login``. The logout candidates are HTTP-edge input —
extracted from the request's credential headers/cookies here and passed to the
operation as a flat ``candidates`` argument, so the operation stays request-free.
"""

from __future__ import annotations

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control.backend import extract_credential_candidates
from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.login import exchange_claim_token as _exchange_claim_token_op
from tai42_skeleton.operations.login import login_methods as _login_methods_op
from tai42_skeleton.operations.login import logout as _logout_op

login_methods = register_operation_route(
    tai42_app,
    operation_metadata_of(_login_methods_op),
    path="/api/login/methods",
    method="GET",
    authed=False,
)

# PUBLIC (the always-public ``/api/login`` prefix): the exchange leg of a one-time claim
# link. The operation validates its own ``ClaimExchange`` body; ``authority_changing``
# on the operation keeps it off the MCP tool surface.
exchange_claim_token = register_operation_route(
    tai42_app,
    operation_metadata_of(_exchange_claim_token_op),
    path="/api/login/claim",
    method="POST",
    authed=False,
)


async def _logout_candidates(request: Request) -> dict:
    """The credential candidates the logout dispatcher fans out over — every value
    presented in ``Authorization`` / ``X-Api-Key`` / a session cookie, in the
    registry-order the operation iterates."""
    return {"candidates": extract_credential_candidates(request)}


logout = register_operation_route(
    tai42_app,
    operation_metadata_of(_logout_op),
    path="/api/auth/logout",
    method="POST",
    context_extractor=_logout_candidates,
    action="write",
)
