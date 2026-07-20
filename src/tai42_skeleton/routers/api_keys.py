"""HTTP surface for provisioning access-control keys and scopes — the authed
doors the Studio's API-keys settings tab consumes.

Routes (all AUTHED), prefixed ``/api/auth`` — each a thin adapter over an operation
in :mod:`tai42_skeleton.operations.api_keys`; no key/scope logic lives here:

- ``GET    /scopes``              — every non-public route mapping as ``{url: scope_id}``.
- ``POST   /scopes``             — map a url to a scope (optional dynamic ``pattern``).
- ``DELETE /scopes/urls``        — unmap one url (registered BEFORE ``/scopes/{scope_id}``
                                    so Starlette matches it, not the id capture).
- ``DELETE /scopes/{scope_id}``  — delete a scope; unknown (no urls) is a loud 404.
- ``GET    /routes``             — the app's own HTTP routes with each route's scope
                                    mapping (``mapped: null`` = the unassigned bucket);
                                    ``Mount`` entries excluded.
- ``GET    /public-routes``      — every route pinned to the public marker.
- ``POST   /public-routes``      — pin a url public (optional dynamic ``pattern``).
- ``DELETE /public-routes``      — unpin a public url; not-pinned is a loud 404.
- ``GET    /tokens-payload``     — every provisioned key's identity + policy (NEVER key material).
- ``POST   /api-keys``           — provision a key; the raw ``sk-…`` is returned ONCE.
- ``PUT    /api-keys/{user_id}`` — partial edit of a key's description/policy in place.
- ``DELETE /api-keys/{user_id}`` — revoke a key (immediate: next request fails to auth).
- ``POST   /claim-links``        — mint a one-time claim link that carries a key to
                                    another device (the QR onboarding leg); token once.
- ``GET    /me``                 — the caller's derived capability projection.
- ``GET    /capabilities``       — whether any configured provider can mint keys, per provider.
- ``GET    /roles``              — the role templates (name/scopes/condition/description).
- ``POST   /validate-condition`` — compile (and optionally sample-evaluate) a jq condition
                                    WITHOUT persisting it (the fail-closed lock-out guard).
- ``GET    /api-keys/{user_id}/policy/versions``  — the user's policy version history; ADMIN-ONLY.
- ``POST   /api-keys/{user_id}/policy/rollback``  — re-point the enforced policy to a prior
                                    version; ADMIN-ONLY.

Each mutating door parses and validates its body at the HTTP edge into the
operation's flat arguments (producing an explicit 400 surface), then the
operation owns the ownership rules, the enforced-store write, the cache-buster bump,
and the durable version-history record. Success bodies are ``{"data": ...}``; failures
are ``{"error": "<message>"}``. Owner-aware ownership rules and the policy-versioning
ordering are documented on the operation module.
"""

from __future__ import annotations

from json import JSONDecodeError
from typing import Any, cast

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.access_control.user import TaiUser
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.api_keys import add_scope_url as _add_scope_url_op
from tai42_skeleton.operations.api_keys import create_api_key as _create_api_key_op
from tai42_skeleton.operations.api_keys import create_claim_link as _create_claim_link_op
from tai42_skeleton.operations.api_keys import delete_scope as _delete_scope_op
from tai42_skeleton.operations.api_keys import edit_api_key as _edit_api_key_op
from tai42_skeleton.operations.api_keys import get_capabilities as _get_capabilities_op
from tai42_skeleton.operations.api_keys import get_me as _get_me_op
from tai42_skeleton.operations.api_keys import list_policy_versions as _list_policy_versions_op
from tai42_skeleton.operations.api_keys import list_public_routes as _list_public_routes_op
from tai42_skeleton.operations.api_keys import list_roles as _list_roles_op
from tai42_skeleton.operations.api_keys import list_routes as _list_routes_op
from tai42_skeleton.operations.api_keys import list_scopes as _list_scopes_op
from tai42_skeleton.operations.api_keys import list_tokens_payload as _list_tokens_payload_op
from tai42_skeleton.operations.api_keys import pin_public_route as _pin_public_route_op
from tai42_skeleton.operations.api_keys import remove_scope_url as _remove_scope_url_op
from tai42_skeleton.operations.api_keys import revoke_api_key as _revoke_api_key_op
from tai42_skeleton.operations.api_keys import rollback_policy as _rollback_policy_op
from tai42_skeleton.operations.api_keys import unpin_public_route as _unpin_public_route_op
from tai42_skeleton.operations.api_keys import validate_condition as _validate_condition_op

# -- HTTP-edge body parsing --------------------------------------------------


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise BadRequestError(f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise BadRequestError("request body must be a JSON object")
    return body


def _require_str(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise BadRequestError(f"{key} must be a non-empty string")
    return value


def _require_str_list(body: dict, key: str) -> list[str]:
    value = body.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise BadRequestError(f"{key} must be a list of strings")
    return value


def _opt_str(body: dict, key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BadRequestError(f"{key} must be a string")
    return value


def _opt_dict(body: dict, key: str) -> dict[str, Any] | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BadRequestError(f"{key} must be a JSON object")
    return value


def _opt_int(body: dict, key: str) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    # A bool is an ``int`` subclass in Python; reject it explicitly so ``true`` is a
    # loud 400, not a silent 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise BadRequestError(f"{key} must be an integer")
    return value


# -- Extractors: raw request -> the operation's flat kwargs -------------------


async def _extract_add_scope_url(request: Request) -> dict:
    body = await _json_body(request)
    return {
        "scope_id": _require_str(body, "scope_id"),
        "url": _require_str(body, "url"),
        "pattern": _opt_str(body, "pattern"),
    }


async def _extract_remove_scope_url(request: Request) -> dict:
    body = await _json_body(request)
    return {"url": _require_str(body, "url")}


async def _extract_pin_public_route(request: Request) -> dict:
    body = await _json_body(request)
    return {"url": _require_str(body, "url"), "pattern": _opt_str(body, "pattern")}


async def _extract_unpin_public_route(request: Request) -> dict:
    body = await _json_body(request)
    return {"url": _require_str(body, "url")}


async def _extract_list_routes(request: Request) -> dict:
    # The op stays request-free: hand it the app's live route table. At runtime
    # ``request.app`` is the route-bearing Starlette app (its ``.routes`` is the live
    # table the catalog reads); the sub-MCP Mount is filtered in the op.
    return {"routes": list(request.app.routes)}


async def _extract_create_api_key(request: Request) -> dict:
    body = await _json_body(request)
    return {
        "user_id": _require_str(body, "user_id"),
        "description": _require_str(body, "description"),
        "scopes": _require_str_list(body, "scopes"),
        "policy_data": _opt_dict(body, "policy_data"),
        "condition": _opt_str(body, "condition"),
        "condition_id": _opt_str(body, "condition_id"),
        "condition_kwargs": _opt_dict(body, "condition_kwargs"),
        "owner_user_id": _opt_str(body, "owner_user_id"),
    }


async def _extract_edit_api_key(request: Request) -> dict:
    # A PATCH-style partial edit: only the fields actually PRESENT in the body are
    # collected, so an absent field is preserved by the operation at its stored value
    # (distinct from an explicit ``null``/``{}``/``""`` that clears an optional gate).
    body = await _json_body(request)
    updates: dict[str, Any] = {}
    if "description" in body:
        updates["description"] = _require_str(body, "description")
    if "scopes" in body:
        updates["scopes"] = _require_str_list(body, "scopes")
    if "policy_data" in body:
        updates["policy_data"] = _opt_dict(body, "policy_data")
    if "condition" in body:
        updates["condition"] = _opt_str(body, "condition")
    if "condition_id" in body:
        updates["condition_id"] = _opt_str(body, "condition_id")
    if "condition_kwargs" in body:
        updates["condition_kwargs"] = _opt_dict(body, "condition_kwargs")
    return {"updates": updates}


async def _extract_create_claim_link(request: Request) -> dict:
    body = await _json_body(request)
    return {"api_key": _require_str(body, "api_key"), "ttl_seconds": _opt_int(body, "ttl_seconds")}


async def _extract_validate_condition(request: Request) -> dict:
    body = await _json_body(request)
    return {
        "condition": _opt_str(body, "condition"),
        "condition_id": _opt_str(body, "condition_id"),
        "condition_kwargs": _opt_dict(body, "condition_kwargs"),
        "sample_context": _opt_dict(body, "sample_context"),
    }


async def _extract_rollback_policy(request: Request) -> dict:
    body = await _json_body(request)
    version = body.get("version")
    # A bool is an ``int`` subclass in Python; reject it explicitly so ``true`` is a
    # loud 400, not a silent version 1.
    if not isinstance(version, int) or isinstance(version, bool):
        raise BadRequestError("version must be an integer")
    return {"version": version}


async def _extract_me(request: Request) -> dict:
    # Derive the caller's identity from the authenticated request so the operation stays
    # request-free. With the gate OFF there is no auth middleware (``request.user`` would
    # be unbound), so signal "no identity" and let the operation return the synthetic
    # total projection. With the gate ON the authenticated-always-allowed carve-out has
    # already guaranteed an authenticated principal here.
    if not access_control_settings().enable:
        return {"user_id": None, "effective_scopes": None, "claims": None}
    user = cast(TaiUser, request.user)
    return {
        "user_id": user.identity,
        "effective_scopes": list(request.auth.scopes),
        "claims": dict(user.token.claims),
    }


# -- Scopes ------------------------------------------------------------------


list_scopes = register_operation_route(
    tai42_app, operation_metadata_of(_list_scopes_op), path="/api/auth/scopes", method="GET"
)

add_scope_url = register_operation_route(
    tai42_app,
    operation_metadata_of(_add_scope_url_op),
    path="/api/auth/scopes",
    method="POST",
    context_extractor=_extract_add_scope_url,
)

# Registered BEFORE ``/scopes/{scope_id}`` — Starlette matches routes in order, so the
# literal ``/scopes/urls`` must win over the ``{scope_id}`` capture.
remove_scope_url = register_operation_route(
    tai42_app,
    operation_metadata_of(_remove_scope_url_op),
    path="/api/auth/scopes/urls",
    method="DELETE",
    context_extractor=_extract_remove_scope_url,
)

delete_scope = register_operation_route(
    tai42_app, operation_metadata_of(_delete_scope_op), path="/api/auth/scopes/{scope_id}", method="DELETE"
)


# -- Route catalog -----------------------------------------------------------


list_routes = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_routes_op),
    path="/api/auth/routes",
    method="GET",
    context_extractor=_extract_list_routes,
)


# -- Public route pins -------------------------------------------------------


list_public_routes = register_operation_route(
    tai42_app, operation_metadata_of(_list_public_routes_op), path="/api/auth/public-routes", method="GET"
)

pin_public_route = register_operation_route(
    tai42_app,
    operation_metadata_of(_pin_public_route_op),
    path="/api/auth/public-routes",
    method="POST",
    context_extractor=_extract_pin_public_route,
)

unpin_public_route = register_operation_route(
    tai42_app,
    operation_metadata_of(_unpin_public_route_op),
    path="/api/auth/public-routes",
    method="DELETE",
    context_extractor=_extract_unpin_public_route,
)


# -- Keys --------------------------------------------------------------------


list_tokens_payload = register_operation_route(
    tai42_app, operation_metadata_of(_list_tokens_payload_op), path="/api/auth/tokens-payload", method="GET"
)

create_api_key = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_api_key_op),
    path="/api/auth/api-keys",
    method="POST",
    context_extractor=_extract_create_api_key,
)

edit_api_key = register_operation_route(
    tai42_app,
    operation_metadata_of(_edit_api_key_op),
    path="/api/auth/api-keys/{user_id}",
    method="PUT",
    context_extractor=_extract_edit_api_key,
)

revoke_api_key = register_operation_route(
    tai42_app, operation_metadata_of(_revoke_api_key_op), path="/api/auth/api-keys/{user_id}", method="DELETE"
)

create_claim_link = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_claim_link_op),
    path="/api/auth/claim-links",
    method="POST",
    context_extractor=_extract_create_claim_link,
)


# -- Mint capabilities + role templates --------------------------------------


get_capabilities = register_operation_route(
    tai42_app, operation_metadata_of(_get_capabilities_op), path="/api/auth/capabilities", method="GET"
)

get_me = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_me_op),
    path="/api/auth/me",
    method="GET",
    context_extractor=_extract_me,
)

list_roles = register_operation_route(
    tai42_app, operation_metadata_of(_list_roles_op), path="/api/auth/roles", method="GET"
)


# -- Policy condition validation --------------------------------------------


validate_condition = register_operation_route(
    tai42_app,
    operation_metadata_of(_validate_condition_op),
    path="/api/auth/validate-condition",
    method="POST",
    context_extractor=_extract_validate_condition,
)


# -- Policy version history + rollback --------------------------------------


list_policy_versions = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_policy_versions_op),
    path="/api/auth/api-keys/{user_id}/policy/versions",
    method="GET",
)

rollback_policy = register_operation_route(
    tai42_app,
    operation_metadata_of(_rollback_policy_op),
    path="/api/auth/api-keys/{user_id}/policy/rollback",
    method="POST",
    context_extractor=_extract_rollback_policy,
)
