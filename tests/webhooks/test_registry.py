"""The webhook-verifier registry + the ``app.webhook_verifiers`` facet."""

from __future__ import annotations

import pytest

from tai42_skeleton.webhooks.registry import WebhookVerifierRegistry


class _V:
    async def verify(self, body, headers, config) -> None:
        return None


def test_register_get_round_trip() -> None:
    reg = WebhookVerifierRegistry()
    v = _V()
    reg.register("prov", v)
    assert reg.get("prov") is v
    assert reg.names() == ["prov"]


def test_duplicate_name_raises() -> None:
    reg = WebhookVerifierRegistry()
    reg.register("prov", _V())
    with pytest.raises(ValueError, match="already registered"):
        reg.register("prov", _V())


def test_unknown_name_raises_loudly() -> None:
    reg = WebhookVerifierRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_reset_clears() -> None:
    reg = WebhookVerifierRegistry()
    reg.register("prov", _V())
    reg.reset()
    assert reg.names() == []


def test_facet_registers_and_resolves_through_app() -> None:
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    app._webhook_verifier_registry.reset()
    try:
        v = _V()
        tai42_app.webhook_verifiers.register("facet-prov", v)
        assert tai42_app.webhook_verifiers.get("facet-prov") is v
    finally:
        app._webhook_verifier_registry.reset()


def test_shared_secret_lifecycle_module_registers() -> None:
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    app._webhook_verifier_registry.reset()
    try:
        # The manifest loads the lifecycle module via ``import_or_reload_package``
        # (import-only key); running it registers ``shared_secret``.
        from tai42_skeleton.app.importer import import_or_reload_package

        import_or_reload_package("tai42_skeleton.webhooks.builtin.shared_secret")
        assert "shared_secret" in app._webhook_verifier_registry.names()
    finally:
        app._webhook_verifier_registry.reset()
