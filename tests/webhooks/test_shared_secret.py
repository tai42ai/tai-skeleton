"""The builtin ``shared_secret`` webhook verifier.

Verifies a named header equals an env-resolved secret via a constant-time
compare; the secret is NEVER a stored config value, only a ``secret_env`` name.
A missing env var raises loudly (the door maps that to a fail-closed 500).
"""

from __future__ import annotations

import pytest
from tai42_contract.webhooks import WebhookVerificationError

from tai42_skeleton.webhooks.builtin.shared_secret import SharedSecretVerifier

_CONFIG = {"header": "X-Webhook-Token", "secret_env": "WH_SECRET"}


async def test_happy_path_matching_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WH_SECRET", "s3cr3t")
    v = SharedSecretVerifier()
    # Returns None (no raise) on a matching header; case-insensitive header name.
    assert await v.verify(b"", {"x-webhook-token": "s3cr3t"}, _CONFIG) is None


async def test_wrong_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WH_SECRET", "s3cr3t")
    v = SharedSecretVerifier()
    with pytest.raises(WebhookVerificationError):
        await v.verify(b"", {"X-Webhook-Token": "nope"}, _CONFIG)


async def test_missing_header_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WH_SECRET", "s3cr3t")
    v = SharedSecretVerifier()
    with pytest.raises(WebhookVerificationError):
        await v.verify(b"", {"Other": "x"}, _CONFIG)


async def test_missing_secret_env_raises_not_verification_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WH_SECRET", raising=False)
    v = SharedSecretVerifier()
    # A missing env var is an operator misconfiguration: it raises loudly (KeyError,
    # which the door maps to a fail-closed 500), NOT a WebhookVerificationError.
    with pytest.raises(KeyError):
        await v.verify(b"", {"X-Webhook-Token": "s3cr3t"}, _CONFIG)


@pytest.mark.parametrize("config", [{"secret_env": "WH_SECRET"}, {"header": "X-Token"}, {}])
async def test_malformed_config_raises(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setenv("WH_SECRET", "s3cr3t")
    v = SharedSecretVerifier()
    # A missing/malformed config key is an operator misconfiguration of the binding:
    # it raises a plain exception (mapped to a fail-closed 500), distinct from the
    # WebhookVerificationError (401) a bad request signature gets.
    with pytest.raises(ValueError, match="requires a non-empty"):
        await v.verify(b"", {"X-Token": "s3cr3t"}, config)


def test_shared_secret_is_not_post_only() -> None:
    # Header-based: works over any delivery method.
    assert SharedSecretVerifier().post_only is False
