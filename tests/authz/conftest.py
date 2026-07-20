"""Shared fakes for the authz suite — reuses the access_control fakes and wires
the redis/pg client seams the verifier + policy enforcer read through."""

from __future__ import annotations

import pytest

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx


class _FakeResourceManager:
    async def render_by_id_or_content(self, *, content, template_id, kwargs):
        return content


class _FakeStorage:
    def __init__(self) -> None:
        self.resource_manager = _FakeResourceManager()


class _FakeApp:
    def __init__(self) -> None:
        self.storage = _FakeStorage()


@pytest.fixture
def bound_app():
    """Bind a minimal fake app onto ``tai42_app`` (the condition renderer the authz
    check reaches through), then restore the unbound state."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    tai42_app.bind(app)
    try:
        yield app
    finally:
        tai42_app.bind(None)


@pytest.fixture
def ac_env(monkeypatch):
    """A fake access-control backend: an empty PG (routes + policies) and a shared
    Redis, wired over the store/verifier/policy client seams. Returns the PG so a
    test seeds routes/policies on it."""
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    return pg
