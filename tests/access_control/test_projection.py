"""The derived capability projection: the coverage matrix, the jq-exactness, the
sub-MCP / tools / agents / pattern derivation, the version-keyed cache, and the core
``projection ⊆ gate`` property pin.

Every scenario runs the REAL enforcer + verifier over the ``FakeAccessControlPg`` store
and ``FakeRedis``; the live-registry seams (route registry, sub-MCP store, tool/agent
registries) are the module's monkeypatch points so a scenario controls the surface.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from starlette.authentication import AuthenticationError
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import JqAuthContext

from tai42_skeleton.access_control import management as management_module
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import projection
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.projection import build_projection
from tai42_skeleton.access_control.roles import EDITOR_JQ, VIEWER_JQ
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.access_control.verifier import AccessControlVerifier

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx


def _areturn(value):
    async def _f():
        return value

    return _f


def _routes(*specs: tuple[str, list[str]]):
    return [SimpleNamespace(path=path, methods=list(methods)) for path, methods in specs]


class _Env:
    def __init__(self, pg: FakeAccessControlPg, redis: FakeRedis, mp: pytest.MonkeyPatch) -> None:
        self.pg = pg
        self.redis = redis
        self.mp = mp

    def routes(self, *specs: tuple[str, list[str]]) -> None:
        self.mp.setattr(projection, "_registry_routes", lambda: _routes(*specs))

    def sub_mcp(self, mapping: dict) -> None:
        self.mp.setattr(projection, "_sub_mcp_routes", _areturn(mapping))

    def tools(self, names: list[str]) -> None:
        self.mp.setattr(projection, "_all_registry_tools", _areturn(list(names)))

    def agents(self, names: list[str]) -> None:
        self.mp.setattr(projection, "_all_agent_names", lambda: list(names))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, pg: FakeAccessControlPg, bound_app) -> Iterator[_Env]:
    redis = FakeRedis(strings={}, hashes={})
    rctx = make_client_ctx(redis)
    monkeypatch.setattr(policy_module, "client_ctx", rctx)
    monkeypatch.setattr(verifier_module, "client_ctx", rctx)
    monkeypatch.setattr(management_module, "client_ctx", rctx)
    monkeypatch.setattr(projection, "client_ctx", rctx)
    # Default-empty live seams; each test overrides what it exercises.
    monkeypatch.setattr(projection, "_registry_routes", list)
    monkeypatch.setattr(projection, "_sub_mcp_routes", _areturn({}))
    monkeypatch.setattr(projection, "_all_registry_tools", _areturn([]))
    monkeypatch.setattr(projection, "_all_agent_names", list)
    projection.reset_projection_cache()
    try:
        yield _Env(pg, redis, monkeypatch)
    finally:
        projection.reset_projection_cache()


# -- pattern sampler + digest + frozen wrapper (pure helpers) ----------------


def test_sample_path_for_pattern_variants():
    assert projection._sample_path_for_pattern(r"^/api/x/\d+$") == "/api/x/1"
    assert projection._sample_path_for_pattern(r"/a/[^/]+") == "/a/a"
    sample = projection._sample_path_for_pattern(r"^/(foo|bar)/\w+$")
    assert sample is not None
    assert re.compile(r"^/(foo|bar)/\w+$").fullmatch(sample)
    # A UUID-shaped class samples its range floor and validates.
    uuid_sample = projection._sample_path_for_pattern(r"^/x/[0-9a-f]{8}$")
    assert uuid_sample is not None
    assert re.compile(r"^/x/[0-9a-f]{8}$").fullmatch(uuid_sample)


def test_sample_path_for_pattern_unsupported_returns_none():
    assert projection._sample_path_for_pattern(r"^(a)\1$") is None  # backreference
    assert projection._sample_path_for_pattern(r"[") is None  # invalid regex


def test_claims_digest_is_stable_and_order_independent():
    assert projection._claims_digest({"a": 1, "b": 2}) == projection._claims_digest({"b": 2, "a": 1})
    assert projection._claims_digest({"a": 1}) != projection._claims_digest({"a": 2})


def test_claims_digest_non_serializable_raises():
    with pytest.raises(TypeError):
        projection._claims_digest({"x": object()})


def test_frozen_claims_hash_and_eq_key_on_digest_only():
    w1 = projection._FrozenClaims({"a": 1}, "dig")
    w2 = projection._FrozenClaims({"totally": "different"}, "dig")
    assert w1 == w2
    assert hash(w1) == hash(w2)
    assert w1 != projection._FrozenClaims({}, "other")
    assert w1 != "not-a-wrapper"


# -- coverage matrix ---------------------------------------------------------


async def test_admin_projection_is_total_without_jq(env: _Env):
    env.pg.add_policy("admin1", scopes=["*"])  # condition-free wildcard = admin
    env.pg.add_route("/api/tools", "tools-read")
    env.pg.add_route("/api/run-tool", "tools-run")
    env.routes(("/api/tools", ["GET"]), ("/api/run-tool", ["POST"]), ("/api/auth/me", ["GET"]))
    env.tools(["alpha", "beta"])
    result = await build_projection("admin1", ["*"], {})
    assert result.admin is True
    assert result.owner_user_id is None
    assert {r.path for r in result.routes} == {"/api/tools", "/api/run-tool", "/api/auth/me"}
    # A global tool-run door is projected → every registry tool.
    assert result.tools == ["alpha", "beta"]
    assert result.mintable is True


async def test_scoped_caller_projects_only_covered_routes(env: _Env):
    env.pg.add_policy("u1", scopes=["tools-read"])
    env.pg.add_route("/api/tools", "tools-read")
    env.pg.add_route("/api/run-tool", "tools-run")
    env.routes(("/api/tools", ["GET"]), ("/api/run-tool", ["POST"]))
    result = await build_projection("u1", ["tools-read"], {})
    assert result.admin is False
    assert {r.path for r in result.routes} == {"/api/tools"}
    # No global door projected → tools falls back to the (empty) sub-MCP union.
    assert result.tools == []


async def test_deny_wins_requires_all_resolved_scopes(env: _Env):
    # A path resolving (through tiers) to two protected ids needs BOTH covered.
    env.pg.add_route("/api/mixed", "s-a")
    env.pg.add_route("/api/mixed-template", "s-b", pattern=r"^/api/mixed$")
    env.routes(("/api/mixed", ["GET"]))
    env.pg.add_policy("partial", scopes=["s-a"])
    partial = await build_projection("partial", ["s-a"], {})
    assert {r.path for r in partial.routes} == set()
    env.pg.add_policy("full", scopes=["s-a", "s-b"])
    full = await build_projection("full", ["s-a", "s-b"], {})
    assert {r.path for r in full.routes} == {"/api/mixed"}


async def test_editor_projects_me_and_non_auth_but_not_admin_area(env: _Env):
    env.pg.add_policy("editor1", scopes=["*"], condition=EDITOR_JQ)
    env.pg.add_route("/api/auth/scopes", "auth-api")
    env.pg.add_route("/api/tools", "tools")
    env.routes(
        ("/api/auth/me", ["GET"]),
        ("/api/auth/scopes", ["GET", "POST"]),
        ("/api/tools", ["GET", "POST"]),
    )
    result = await build_projection("editor1", ["*"], {})
    routes = {r.path: r.methods for r in result.routes}
    assert routes["/api/auth/me"] == ["GET"]
    assert routes["/api/auth/scopes"] == ["GET"]  # POST fenced (scope administration)
    assert set(routes["/api/tools"]) == {"GET", "POST"}


async def test_owned_key_projects_intersection_and_respects_owner_condition(env: _Env):
    # A ["*"] key owned by a scoped owner that fences to a single path: the projection
    # shows the attenuated intersection AND the owner second-pass denial.
    env.pg.add_policy("key1", scopes=["*"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    env.pg.add_policy("owner1", scopes=["tools", "other"], condition='.request.path == "/api/tools"')
    env.pg.add_route("/api/tools", "tools")
    env.pg.add_route("/api/other", "other")
    env.routes(("/api/tools", ["GET"]), ("/api/other", ["GET"]))
    result = await build_projection("key1", ["tools", "other"], {OWNER_USER_ID_CLAIM: "owner1"})
    assert result.admin is False  # an owned key is never admin
    assert result.owner_user_id == "owner1"
    # /api/other is scope-covered but the owner condition denies it.
    assert {r.path for r in result.routes} == {"/api/tools"}


# -- sub-MCP / tools / agents ------------------------------------------------


async def test_sub_mcp_filtered_by_mount_coverage(env: _Env):
    env.pg.add_policy("u1", scopes=["mcp-app"])
    env.pg.add_route(f"{projection.ROOT_PREFIX}/shop", "mcp-app")
    env.pg.add_route(f"{projection.ROOT_PREFIX}/secret", "secret-scope")
    env.sub_mcp(
        {
            "shop": SimpleNamespace(tools=["t1"], transport="http"),
            "secret": SimpleNamespace(tools=["t2"], transport="http"),
        }
    )
    result = await build_projection("u1", ["mcp-app"], {})
    assert {e.slug for e in result.sub_mcp} == {"shop"}  # secret mount uncovered
    # No global tool door → tools is the union of the ALLOWED sub-MCP mounts' tools.
    assert result.tools == ["t1"]


async def test_global_tool_door_projects_every_registry_tool(env: _Env):
    env.pg.add_policy("u1", scopes=["tools-run"])
    env.pg.add_route("/api/run-tool", "tools-run")
    env.routes(("/api/run-tool", ["POST"]))
    env.tools(["a", "b", "c"])
    result = await build_projection("u1", ["tools-run"], {})
    assert result.tools == ["a", "b", "c"]


async def test_agents_projected_per_run_path_for_editor(env: _Env):
    env.pg.add_policy("editor1", scopes=["*"], condition=EDITOR_JQ)
    env.pg.add_route("/api/agents/alpha/runs", "agents")
    env.pg.add_route("/api/agents/beta/runs", "agents")
    env.agents(["alpha", "beta"])
    result = await build_projection("editor1", ["*"], {})
    assert result.agents == ["alpha", "beta"]


async def test_agents_excluded_when_run_door_jq_denies(env: _Env):
    # A viewer cannot POST an agent run (only the read-only leg admits its methods), so no
    # agent projects even though the run door resolves and its scope is covered.
    env.pg.add_policy("viewer1", scopes=["*"], condition=VIEWER_JQ)
    env.pg.add_route("/api/agents/alpha/runs", "agents")
    env.agents(["alpha"])
    result = await build_projection("viewer1", ["*"], {})
    assert result.agents == []


# -- route patterns ----------------------------------------------------------


async def test_route_patterns_scope_and_jq_filtered_no_topology_leak(env: _Env):
    env.pg.add_policy("u1", scopes=["pub-dyn"])
    env.pg.add_route("/api/dyn-open", "pub-dyn", pattern=r"^/api/dyn/\d+$")
    env.pg.add_route("/api/dyn-secret", "hidden", pattern=r"^/api/secret/\d+$")
    result = await build_projection("u1", ["pub-dyn"], {})
    # The covered pattern projects; the uncovered one is NOT leaked (topology-leak pin).
    assert {p.pattern for p in result.route_patterns} == {r"^/api/dyn/\d+$"}
    assert result.route_patterns[0].scope_id == "pub-dyn"


async def test_templated_route_projects_as_pattern_not_concrete(env: _Env):
    # The real route registry (``load_api_routes``) returns TEMPLATED brace paths like
    # ``/api/agents/{name}/runs``. When such a path is scope-mapped by a dynamic-pattern
    # row (``[^/]+`` fullmatches the ``{name}`` literal), it must project ONLY via
    # route_patterns — never as a concrete RouteEntry carrying literal braces (which would
    # also double-list the surface the pattern already carries).
    env.pg.add_policy("u1", scopes=["agents-run"])
    env.pg.add_route("/api/agents/dyn", "agents-run", pattern=r"^/api/agents/[^/]+/runs$")
    env.pg.add_route("/api/agents/list", "agents-run")
    env.routes(("/api/agents/{name}/runs", ["POST"]), ("/api/agents/list", ["GET"]))
    result = await build_projection("u1", ["agents-run"], {})
    # The concrete sibling route projects as itself; the templated path is emitted ONLY as a
    # pattern, never as a concrete RouteEntry carrying literal braces.
    assert {r.path for r in result.routes} == {"/api/agents/list"}
    assert all("{" not in r.path for r in result.routes)
    assert {p.pattern for p in result.route_patterns} == {r"^/api/agents/[^/]+/runs$"}


async def test_non_sampleable_pattern_is_excluded_and_logged(env: _Env, caplog):
    env.pg.add_policy("u1", scopes=["s"])
    env.pg.add_route("/api/br", "s", pattern=r"^(a)\1$")  # backreference — no representative
    with caplog.at_level("INFO", logger="tai42_skeleton.access_control.projection"):
        result = await build_projection("u1", ["s"], {})
    assert result.route_patterns == []
    assert "non-sampleable" in caplog.text


# -- cache + failure doctrine ------------------------------------------------


async def test_projection_cache_is_version_keyed(env: _Env):
    env.pg.add_policy("u1", scopes=["s"])
    env.pg.add_route("/api/x", "s")
    env.routes(("/api/x", ["GET"]), ("/api/y", ["GET"]), ("/api/z", ["GET"]))
    r1 = await build_projection("u1", ["s"], {})
    assert {r.path for r in r1.routes} == {"/api/x"}
    # A route added + a version bump is visible on the next read (cross-worker miss).
    env.pg.add_route("/api/y", "s")
    await management_module.bump_policy_version()
    r2 = await build_projection("u1", ["s"], {})
    assert {r.path for r in r2.routes} == {"/api/x", "/api/y"}
    # A store change WITHOUT a bump is NOT reflected — proof the projection is cached.
    env.pg.add_route("/api/z", "s")
    r3 = await build_projection("u1", ["s"], {})
    assert {r.path for r in r3.routes} == {"/api/x", "/api/y"}


async def test_store_error_raises_never_a_partial_projection(env: _Env):
    env.pg.add_policy("u1", scopes=["s"])
    env.pg.add_route("/api/x", "s")
    env.routes(("/api/x", ["GET"]))
    env.pg.fault = ("SELECT scope_id FROM access_control_routes WHERE url", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        await build_projection("u1", ["s"], {})


def test_synthetic_full_projection_is_total_with_derived_mintable(env: _Env):
    result = projection.synthetic_full_projection()
    assert result.user_id == projection.NO_AUTH_USER_ID
    assert result.admin is True
    assert result.scopes == ["*"]
    assert result.routes == []
    assert result.tools == []
    assert result.agents == []
    assert result.mintable is True  # derived from the (mint-capable) redis provider


# -- the core property pin: projection ⊆ gate, both directions ---------------


async def _gate_oracle(
    settings, user_id: str, effective_scopes: list[str], claims: dict, path: str, method: str
) -> bool:
    """An INDEPENDENT re-implementation of the gate decision (resolution + coverage +
    carve-out + the backend's two-pass jq), used to cross-check the projection."""
    verifier = AccessControlVerifier(settings, providers=[])
    enforcer = PolicyEnforcer(settings)
    scope_set = set(effective_scopes)
    # The carve-out is checked BEFORE resolution, mirroring the middleware/product order, so
    # a carve-out path that ALSO carries a route row bound to an uncovered scope is still
    # admitted (the carve-out wins) rather than falling through to a coverage test.
    if path in set(settings.authenticated_always_allowed_paths):
        reachable = True
    else:
        ids = await verifier.resolve_resource_ids(path)
        if ids:
            public = settings.public_resource_id
            if set(ids) == {public}:
                reachable = True
            else:
                protected = [rid for rid in ids if rid != public]
                reachable = "*" in scope_set or all(rid in scope_set for rid in protected)
        else:
            reachable = False
    if not reachable:
        return False

    policy = await enforcer.get_policy(user_id)
    owner = claims.get(OWNER_USER_ID_CLAIM)
    owner_policy = await enforcer.get_policy(owner) if owner is not None else None
    live_ctx = await enforcer.get_live_context(user_id)
    request = {"method": method, "path": path}
    try:
        await enforcer.enforce(
            JqAuthContext(
                sub=user_id,
                scopes=list(effective_scopes),
                identity=claims,
                policy=policy.policy_data,
                context=live_ctx,
                request=request,
                system={"time": 0},
            ).model_dump(),
            policy.condition,
            condition_configured=policy.condition is not None or policy.condition_id is not None,
        )
        if owner_policy is not None and (owner_policy.condition is not None or owner_policy.condition_id is not None):
            await enforcer.enforce(
                JqAuthContext(
                    sub=user_id,
                    scopes=list(owner_policy.scopes),
                    identity=claims,
                    policy=owner_policy.policy_data,
                    context=live_ctx,
                    request=request,
                    system={"time": 0},
                ).model_dump(),
                owner_policy.condition,
                condition_configured=True,
            )
    except AuthenticationError:
        return False
    return True


async def test_projection_equals_gate_across_identity_matrix(env: _Env):
    # A matrix of synthetic policies exercising EVERY jq branch cross-checked against the
    # independent oracle: a single-pass condition-bearing caller (editor), a condition-free
    # admin (the total branch), and an owned key whose OWNER carries a path-fencing
    # condition (the two-pass owner branch).
    settings = access_control_settings()
    env.pg.add_policy("editor1", scopes=["*"], condition=EDITOR_JQ)
    env.pg.add_policy("admin1", scopes=["*"])
    env.pg.add_policy("key1", scopes=["*"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    env.pg.add_policy("owner1", scopes=["tools", "other"], condition='.request.path == "/api/tools"')
    env.pg.add_route("/api/auth/scopes", "auth-api")
    env.pg.add_route("/api/tools", "tools")
    env.pg.add_route("/api/other", "other")
    env.routes(
        ("/api/auth/me", ["GET"]),
        ("/api/auth/scopes", ["GET", "POST"]),
        ("/api/tools", ["GET", "POST"]),
        ("/api/other", ["GET"]),
        ("/api/nope", ["GET"]),  # unmapped, not carved — denied for everyone
    )
    candidates = [
        ("/api/auth/me", "GET"),
        ("/api/auth/scopes", "GET"),
        ("/api/auth/scopes", "POST"),
        ("/api/tools", "GET"),
        ("/api/tools", "POST"),
        ("/api/other", "GET"),
        ("/api/nope", "GET"),
    ]
    identities = [
        ("editor1", ["*"], {}),
        ("admin1", ["*"], {}),
        ("key1", ["tools", "other"], {OWNER_USER_ID_CLAIM: "owner1"}),
    ]
    for user_id, scopes, claims in identities:
        result = await build_projection(user_id, scopes, claims)
        projected = {(m, r.path) for r in result.routes for m in r.methods}
        for path, method in candidates:
            admitted = await _gate_oracle(settings, user_id, scopes, claims, path, method)
            assert ((method, path) in projected) is admitted, (user_id, method, path)


async def test_carve_out_wins_over_route_row_bound_to_uncovered_scope(env: _Env):
    # ``/api/auth/me`` is authenticated-always-allowed. Even when it ALSO carries a route
    # row bound to a scope the caller does NOT cover, the carve-out is checked BEFORE
    # resolution (exactly as the middleware does), so the path is STILL projected — the
    # carve-out wins over the uncovered coverage test it would otherwise fall through to.
    settings = access_control_settings()
    env.pg.add_policy("u1", scopes=["unrelated"])
    env.pg.add_route("/api/auth/me", "admin-only-scope")  # a scope u1 does NOT cover
    env.routes(("/api/auth/me", ["GET"]))
    result = await build_projection("u1", ["unrelated"], {})
    projected = {(m, r.path) for r in result.routes for m in r.methods}
    assert ("GET", "/api/auth/me") in projected
    # The independent oracle agrees: carve-out wins over the uncovered route row.
    assert await _gate_oracle(settings, "u1", ["unrelated"], {}, "/api/auth/me", "GET") is True


async def test_projection_uses_original_scope_order_like_the_gate(env: _Env):
    # An order-sensitive condition: the backend evaluates jq against the ORIGINAL scope
    # order, so the projection must too — the sorted cache key must not leak into jq or an
    # order-sensitive fence would diverge (an OVER/under-grant). ``.scopes[0]`` reads the
    # first scope: original order has "tools" first (admit), a sorted order would put
    # "other" first (a wrong deny).
    settings = access_control_settings()
    env.pg.add_policy("u1", scopes=["*"], condition='.scopes[0] == "tools"')
    env.pg.add_route("/api/tools", "tools")
    env.routes(("/api/tools", ["GET"]))
    scopes = ["tools", "other"]
    result = await build_projection("u1", scopes, {})
    projected = {(m, r.path) for r in result.routes for m in r.methods}
    admitted = await _gate_oracle(settings, "u1", scopes, {}, "/api/tools", "GET")
    assert admitted is True  # the gate admits under the original order
    assert ("GET", "/api/tools") in projected  # the projection matches the gate, not sorted


async def test_jq_infra_fault_during_build_propagates_and_gate_fails_closed(env: _Env):
    # A condition that raises at EVALUATION (a path string cannot be a number) is an
    # INFRASTRUCTURE fault, NOT a policy deny.
    settings = access_control_settings()
    env.pg.add_policy("u1", scopes=["*"], condition=".request.path | tonumber")
    env.pg.add_route("/api/tools", "tools")
    env.routes(("/api/tools", ["GET"]))
    # (a) The build PROPAGATES it loudly — never a silently-shrunk 200 projection with the
    # route dropped as if the policy had denied it.
    with pytest.raises(policy_module.PolicyEvaluationError):
        await build_projection("u1", ["*"], {})
    # (b) The runtime gate keeps failing closed on the SAME fault: enforce raises a plain
    # ``Exception`` subclass (not an allow), which backend/authz catch via ``except
    # Exception`` and turn into a deny — unchanged, no new 500 and no security regression.
    enforcer = PolicyEnforcer(settings)
    ctx = JqAuthContext(
        sub="u1",
        scopes=["*"],
        identity={},
        policy={},
        context={},
        request={"method": "GET", "path": "/api/tools"},
        system={"time": 0},
    ).model_dump()
    with pytest.raises(policy_module.PolicyEvaluationError):
        await enforcer.enforce(ctx, ".request.path | tonumber", condition_configured=True)
    assert issubclass(policy_module.PolicyEvaluationError, Exception)


async def test_projected_pattern_is_admitted_by_the_gate(env: _Env):
    # Patterns are IN the property: a projected pattern's representative path is gate-
    # admitted; an excluded (uncovered) pattern's representative path is gate-denied.
    settings = access_control_settings()
    env.pg.add_policy("u1", scopes=["pub-dyn"])
    env.pg.add_route("/api/dyn-open", "pub-dyn", pattern=r"^/api/dyn/\d+$")
    env.pg.add_route("/api/dyn-secret", "hidden", pattern=r"^/api/secret/\d+$")
    result = await build_projection("u1", ["pub-dyn"], {})
    projected = {p.pattern for p in result.route_patterns}
    assert projected == {r"^/api/dyn/\d+$"}
    assert await _gate_oracle(settings, "u1", ["pub-dyn"], {}, "/api/dyn/1", "GET") is True
    assert await _gate_oracle(settings, "u1", ["pub-dyn"], {}, "/api/secret/1", "GET") is False
