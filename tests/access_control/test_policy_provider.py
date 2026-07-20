"""Behavior of ``PolicyEnforcer`` (policy/context fetch + jq enforce).

The enforcer reads the POLICY body from the PG store (the ``pg`` fake), while the
live context and the policy-version counter stay on the AC Redis (the ``FakeRedis``).
The RedisApiKeyProvider identity-lookup coverage lives in the ``tai-identity-redis``
plugin repo along with the provider itself.
"""

from __future__ import annotations

import json

import pytest
from starlette.authentication import AuthenticationError

from tai_skeleton.access_control import policy as policy_module
from tai_skeleton.access_control import store as store_module
from tai_skeleton.access_control.policy import PolicyEnforcer, PolicyEvaluationError
from tai_skeleton.access_control.settings import AccessControlSettings

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

# -- PolicyEnforcer ----------------------------------------------------------


async def test_fetch_policy_builds_from_store(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_policy("u1", scopes=["admin"], policy_data={"plan_limit": 100})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    enforcer = PolicyEnforcer(settings)
    policy = await enforcer.get_policy("u1")
    assert policy.scopes == ["admin"]
    assert policy.policy_data == {"plan_limit": 100}


async def test_fetch_policy_empty_when_no_data(monkeypatch):
    settings = AccessControlSettings()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    enforcer = PolicyEnforcer(settings)
    policy = await enforcer.get_policy("missing")
    assert policy.scopes == []


async def test_fetch_policy_raises_on_error_is_fail_closed(monkeypatch):
    """A policy-fetch error fails closed by RAISING, so alru never caches a
    degraded empty policy; the error propagates out of ``authenticate`` and
    becomes a clean deny rather than a silent no-scopes allow-through."""
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.fault = ("SELECT scopes, policy_data", RuntimeError("pg down"))
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    enforcer = PolicyEnforcer(settings)
    with pytest.raises(RuntimeError, match="pg down"):
        await enforcer.get_policy("u1")


async def test_live_context_returns_redis_value(monkeypatch):
    """The context hash stores each field JSON-encoded; the read per-field-decodes,
    so an integer counter stored as ``"3"`` comes back as a real int ``3``."""
    settings = AccessControlSettings()
    fake = FakeRedis(hashes={f"{settings.context_prefix}u1": {"used": "3"}})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)
    assert await enforcer.get_live_context("u1") == {"used": 3}


async def test_live_context_decodes_mixed_field_types(monkeypatch):
    """Each field value is decoded independently, reassembling ints, strings, and
    nested objects into a plain typed dict before it reaches jq."""
    settings = AccessControlSettings()
    fake = FakeRedis(hashes={f"{settings.context_prefix}u1": {"used": "3", "tier": '"gold"', "usage": '{"a": 1}'}})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)
    assert await enforcer.get_live_context("u1") == {"used": 3, "tier": "gold", "usage": {"a": 1}}


async def test_live_context_empty_when_missing(monkeypatch):
    settings = AccessControlSettings()
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    enforcer = PolicyEnforcer(settings)
    assert await enforcer.get_live_context("u1") == {}


async def test_live_context_malformed_field_raises_fail_closed(monkeypatch):
    """A field value that is not valid JSON makes the per-field decode RAISE; it is
    never defaulted to null/``{}`` (which could flip a deny into an allow). The
    error propagates so the auth decision fails closed."""
    settings = AccessControlSettings()
    fake = FakeRedis(hashes={f"{settings.context_prefix}u1": {"used": "not-json"}})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)
    with pytest.raises(json.JSONDecodeError):
        await enforcer.get_live_context("u1")


async def test_live_context_propagates_error_fail_closed(monkeypatch):
    """A live-context fetch error must NOT be masked as an empty context:
    substituting {} could satisfy a quota/limit allow-condition and flip a deny
    into an allow. The error propagates so the auth decision fails closed."""
    settings = AccessControlSettings()
    fake = FakeRedis(raise_hgetall=RuntimeError("redis down"))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)
    with pytest.raises(RuntimeError, match="redis down"):
        await enforcer.get_live_context("u1")


async def test_live_context_numeric_condition_passes_through_jq(monkeypatch):
    """The decoded int flows into jq as a real JSON number, so a numeric
    allow-condition ``.context.used < .policy.limit`` still compares numerically
    (a ``"3"`` stored as a string would break every ``<`` comparison)."""
    settings = AccessControlSettings()
    fake = FakeRedis(hashes={f"{settings.context_prefix}u1": {"used": "3"}})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)

    context = await enforcer.get_live_context("u1")
    jq_input = {"context": context, "policy": {"limit": 5}}
    # 3 < 5 → allowed (no raise).
    await enforcer.enforce(jq_input, ".context.used < .policy.limit")
    # 3 < 2 → denied.
    jq_input["policy"]["limit"] = 2
    with pytest.raises(AuthenticationError, match="Policy violation"):
        await enforcer.enforce(jq_input, ".context.used < .policy.limit")


async def test_get_auth_data_combines_policy_and_context(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_policy("u1", scopes=["s"])
    fake = FakeRedis(hashes={f"{settings.context_prefix}u1": {"used": "9"}})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)
    policy, context = await enforcer.get_auth_data("u1")
    assert policy.scopes == ["s"]
    assert context == {"used": 9}


async def test_enforce_raises_evaluation_error_on_runtime_jq_error():
    enforcer = PolicyEnforcer(AccessControlSettings())
    # An INFRASTRUCTURE/evaluation fault (a string cannot be a number) is raised as a
    # DISTINCT ``PolicyEvaluationError``, NOT an ``AuthenticationError`` — so a build-time
    # caller that narrowly catches the deny type lets this propagate loudly instead of
    # swallowing it as a deny. It is deliberately not an ``AuthenticationError`` subclass.
    with pytest.raises(PolicyEvaluationError, match="Policy error"):
        await enforcer.enforce({"x": "abc"}, ".x | tonumber")
    assert not issubclass(PolicyEvaluationError, AuthenticationError)


# -- policy version (cross-worker cache coherence) ---------------------------


async def test_policy_version_bump_busts_warm_cache(monkeypatch):
    """A bumped version participates in the cache key, so a warm entry is bypassed
    and the edited policy is re-read — without waiting out the ttl. The version
    counter is plain-Redis; the policy body is the PG store."""
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_policy("u1", scopes=["s1"])
    fake = FakeRedis(strings={})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    enforcer = PolicyEnforcer(settings)

    # Warm the cache at version 0 (no version key set yet).
    assert (await enforcer.get_policy("u1")).scopes == ["s1"]

    # Edit the stored policy WITHOUT bumping: the warm cache still serves it.
    pg.policy("u1")["scopes"] = ["s1", "s2"]
    assert (await enforcer.get_policy("u1")).scopes == ["s1"]

    # Bump the version → different cache key → fresh read.
    fake._strings[settings.policy_version_key] = "1"
    assert (await enforcer.get_policy("u1")).scopes == ["s1", "s2"]


async def test_policy_version_read_error_fails_closed(monkeypatch):
    """A failure reading the version key must fail closed by RAISING, never a
    silent default that would pin the cache to one slot."""
    settings = AccessControlSettings()
    fake = FakeRedis(raise_get=RuntimeError("redis down"))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    with pytest.raises(RuntimeError, match="redis down"):
        await PolicyEnforcer(settings).get_policy("u1")
