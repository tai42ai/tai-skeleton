"""Conformance + behavior tests for the copied access_control feature.

Conformance asserts the local implementations satisfy the
``tai42_contract.access_control`` protocols/ABCs they are meant to implement.
Behavior exercises the policy enforcer's jq evaluation on simple cases (no redis).
"""

import pytest
from starlette.authentication import AuthenticationError
from tai42_contract.access_control.identity import (
    AuthIdentity,
)
from tai42_contract.access_control.identity import (
    IdentityProvider as ContractIdentityProvider,
)
from tai42_contract.access_control.policy import PolicyEnforcer as ContractPolicyEnforcer
from tai42_contract.access_control.verifier import Verifier as ContractVerifier

from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.verifier import AccessControlVerifier


class _StubProvider(ContractIdentityProvider):
    async def validate_token(self, token: str):
        return AuthIdentity(user_id="u", claims={})


def _settings() -> AccessControlSettings:
    return AccessControlSettings()


def test_verifier_satisfies_contract_protocol():
    verifier = AccessControlVerifier(_settings(), providers=[_StubProvider()])
    assert isinstance(verifier, ContractVerifier)


def test_policy_enforcer_satisfies_contract_protocol():
    enforcer = PolicyEnforcer(_settings())
    assert isinstance(enforcer, ContractPolicyEnforcer)


async def test_enforce_empty_expression_is_noop():
    assert await PolicyEnforcer(_settings()).enforce({"anything": 1}, None) is None


async def test_enforce_passes_when_condition_true():
    assert await PolicyEnforcer(_settings()).enforce({"scopes": ["admin"]}, '.scopes | index("admin") != null') is None


async def test_enforce_raises_on_violation():
    enforcer = PolicyEnforcer(_settings())
    # ``match`` pins the generic message: a regression that re-wraps this as
    # "Policy error: ..." (leaking internals) would fail this assertion.
    with pytest.raises(AuthenticationError, match="Policy violation"):
        await enforcer.enforce({"plan": "free"}, '.plan == "pro"')


@pytest.mark.parametrize(
    ("context", "expression"),
    [
        ({"scopes": ["admin"]}, ".scopes"),  # truthy list, not literal ``true``
        ({"n": 5}, ".n"),  # truthy number
        ({"s": "yes"}, ".s"),  # truthy string
    ],
)
async def test_enforce_denies_truthy_non_true_results(context, expression):
    # The jq result must be literally ``true`` to allow. A truthy-but-not-``true``
    # value (a list, a number, a string) must DENY. This guards against a future
    # loosening to ``if not result`` (Python-truthiness), which would flip these
    # deny cases into allow.
    enforcer = PolicyEnforcer(_settings())
    with pytest.raises(AuthenticationError, match="Policy violation"):
        await enforcer.enforce(context, expression)


async def test_enforce_denies_configured_condition_that_renders_empty():
    # A condition WAS configured but rendered to an empty string (a false jinja
    # branch, an undefined var). This must DENY (fail closed), not be mistaken for
    # "no condition configured" (which would fail open and allow the caller).
    enforcer = PolicyEnforcer(_settings())
    with pytest.raises(AuthenticationError, match="Policy violation"):
        await enforcer.enforce({"anything": 1}, "", condition_configured=True)


async def test_enforce_allows_when_no_condition_configured():
    # Genuinely no condition configured -> nothing to enforce -> allow.
    enforcer = PolicyEnforcer(_settings())
    assert await enforcer.enforce({"anything": 1}, "", condition_configured=False) is None
    assert await enforcer.enforce({"anything": 1}, None, condition_configured=False) is None


def test_settings_compose_redis_connection_not_inherit():
    settings = _settings()
    # The redis connection is a composed field, not mixed into the feature config.
    assert not hasattr(settings, "client_kwargs")
    assert settings.redis.client_kwargs()["url"] == "redis://localhost:6379/0"
    assert settings.redis.socket_timeout == 5
    assert settings.redis.retry_on_timeout is True


def test_redis_url_uses_access_control_prefix(monkeypatch):
    # The auth-gate redis is configured by ACCESS_CONTROL_REDIS_URL (its own
    # prefix, matching the interactions/hooks/connector-store siblings).
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("ACCESS_CONTROL_REDIS_URL", "redis://ac-host:6390/2")
    assert AccessControlSettings().redis.redis_url == "redis://ac-host:6390/2"


def test_bare_redis_url_does_not_configure_access_control(monkeypatch):
    # Regression guard: the auth gate must NOT silently read the shared REDIS_URL
    # (it has its own ACCESS_CONTROL_ prefix).
    monkeypatch.delenv("ACCESS_CONTROL_REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://should-not-apply:1/9")
    assert AccessControlSettings().redis.redis_url == "redis://localhost:6379/0"
