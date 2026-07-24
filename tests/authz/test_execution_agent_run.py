"""The route-mode RUN authorizer for the agent-run door (``POST /api/agents/{name}/runs``),
decided against the synthetic execution identity a fire runs as.

The door is a ``custom_route``, not a registered operation, so ``check`` cannot decide it:
``authorize_execution_agent_run`` names the concrete path and runs the same post-pin tail,
so a run reached as a door is fenced no more loosely than the same operation as a tool. The
fenced leg re-stamps the registered route's action to exercise the per-tag hard fence.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM

from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.authz.execution import authorize_execution_agent_run
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.operations.errors import PermissionDenied

# alru caches are held across the several ``asyncio.run`` loops each test opens — a benign
# loop-reset artifact, since a real process serves one loop for its lifetime.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_AGENT = "faker"
_RUN_PATH = f"/api/agents/{_AGENT}/runs"
_ROUTE_KEY = ("/api/agents/{name}/runs", ("POST",))
_SCOPE = "agents"  # the scope (and feature tag) the run door is registered under


def _settings() -> AccessControlSettings:
    return AccessControlSettings(enable=True)


def _identity(key: str, *, fingerprint: str | None = None, owner: str | None = None) -> CallerIdentity:
    """The execution identity a fire runs as, carrying the bound per-mint fingerprint the
    tail re-asserts against the live policy."""
    claims = {} if owner is None else {OWNER_USER_ID_CLAIM: owner}
    return CallerIdentity(user_id=key, claims=claims, execution_key_fingerprint=fingerprint or f"fp-{key}")


@contextmanager
def _run_door_action(action: str) -> Iterator[None]:
    """Re-stamp the run-door route's action-class for the duration; the resolver index must
    be rebuilt either side or the stamp is invisible."""
    original = route_registry._routes[_ROUTE_KEY]
    route_registry._routes[_ROUTE_KEY] = dataclasses.replace(original, action=action)
    reset_route_index()
    try:
        yield
    finally:
        route_registry._routes[_ROUTE_KEY] = original
        reset_route_index()


# -- the grantable run door ---------------------------------------------------


def test_an_authorized_exec_key_may_run_the_agent(ac_env, bound_app) -> None:
    ac_env.add_route(_RUN_PATH, _SCOPE)
    ac_env.add_policy("k-run", scopes=[_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-run"})
    asyncio.run(authorize_execution_agent_run(_identity("k-run"), _AGENT, settings=_settings()))


def test_a_key_lacking_the_run_scope_is_denied(ac_env, bound_app) -> None:
    ac_env.add_route(_RUN_PATH, _SCOPE)
    ac_env.add_policy("k-none", scopes=["other"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-none"})
    with pytest.raises(PermissionDenied, match="insufficient scope"):
        asyncio.run(authorize_execution_agent_run(_identity("k-none"), _AGENT, settings=_settings()))


# -- liveness: a revoked / reminted key ---------------------------------------


def test_a_fingerprint_mismatched_key_is_denied(ac_env, bound_app) -> None:
    # A remint of the same ``user_id`` writes a fresh fingerprint, so the bound record's
    # stale one fails the live equality and never inherits the reminted key's authority.
    ac_env.add_route(_RUN_PATH, _SCOPE)
    ac_env.add_policy("k-run", scopes=[_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-run"})
    stale = _identity("k-run", fingerprint="fp-STALE")
    with pytest.raises(PermissionDenied, match="no longer matches the bound key identity"):
        asyncio.run(authorize_execution_agent_run(stale, _AGENT, settings=_settings()))


def test_a_deleted_key_is_denied(ac_env, bound_app) -> None:
    # A key deleted after the fire opened reads as an empty policy and is refused.
    ac_env.add_route(_RUN_PATH, _SCOPE)
    with pytest.raises(PermissionDenied, match="principal has no policy"):
        asyncio.run(authorize_execution_agent_run(_identity("ghost"), _AGENT, settings=_settings()))


# -- the per-tag fence: a fenced door is admin-only ---------------------------


def test_a_fenced_run_door_is_admin_only(ac_env, bound_app) -> None:
    ac_env.add_route(_RUN_PATH, _SCOPE)
    # The key HOLDS the door's scope, so the deny can only be the per-tag LEVEL fence.
    ac_env.add_policy("k-run", scopes=[_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-run"})
    ac_env.add_policy("k-admin", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-admin"})
    settings = _settings()
    with _run_door_action("fenced"):
        with pytest.raises(PermissionDenied, match="is not permitted"):
            asyncio.run(authorize_execution_agent_run(_identity("k-run"), _AGENT, settings=settings))
        # ALLOW parity: the deny above is the fence, not an unreachable route.
        asyncio.run(authorize_execution_agent_run(_identity("k-admin"), _AGENT, settings=settings))


# -- access control disabled short-circuits -----------------------------------


def test_disabled_access_control_allows(ac_env, bound_app) -> None:
    # Matches the HTTP edge with no middleware installed: returns before any store read.
    asyncio.run(authorize_execution_agent_run(_identity("k-run"), _AGENT, settings=AccessControlSettings(enable=False)))
