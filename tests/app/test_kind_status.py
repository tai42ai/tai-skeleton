"""``collect_kind_status`` reports exactly the nine pluggable kinds, each with the
live ``active``/``default``/``off`` state of its registry, and the NoOp-monitoring
warning fires once per process only when NoOp is the active recorder."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from tai42_contract.app import tai42_app

from tai42_skeleton.app import kind_status as ks
from tai42_skeleton.app.instance import build_app
from tai42_skeleton.app.kind_status import KindStatus, collect_kind_status
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.monitoring.noop import NoOpMonitoring
from tai42_skeleton.monitoring.registry import init_monitoring, reset_monitoring
from tai42_skeleton.plugins.registry import StudioPluginError
from tests._helpers import DeliverOnlyChannel

_EXPECTED_KINDS = [
    "identity",
    "accounts",
    "monitoring",
    "storage",
    "backend",
    "channels",
    "webhook_verifiers",
    "config",
    "studio_plugins",
]


class _RealMonitoring:
    """A non-NoOp monitoring backend: composes the NoOp writer/reader so it
    satisfies the ``Monitoring`` protocol without subclassing ``NoOpMonitoring``,
    so the collector reports it ``active``."""

    def __init__(self) -> None:
        self._inner = NoOpMonitoring()

    @property
    def writer(self):
        return self._inner.writer

    @property
    def reader(self):
        return self._inner.reader

    def add_project(self, project) -> None:
        return None


class _FakeStorage:
    """Stand-in for a registered storage provider — the collector reads only its
    type for the row's plugin/detail."""


class _FakeBackend:
    """Stand-in for a registered backend provider."""


class _Channel(DeliverOnlyChannel):
    async def deliver(self, delivery) -> None:
        return None


@pytest.fixture
def bound_app(monkeypatch: pytest.MonkeyPatch):
    """The process app singleton, bound to ``tai42_app`` with an empty live manifest
    and its app-bound registries cleared, so each row is deterministic before a
    test permutes exactly one kind. Cleared again on teardown."""
    app = build_app()
    tai42_app.bind(app)
    monkeypatch.setattr(app, "_manifest", Manifest.model_validate({}), raising=False)
    monkeypatch.setattr(app._storage_registry, "_provider", None)
    monkeypatch.setattr(app._backend_holder, "_backend", None)
    app._channel_registry.reset()
    reset_monitoring()
    try:
        yield app
    finally:
        app._channel_registry.reset()
        reset_monitoring()


def _row(kind: str) -> KindStatus:
    return next(row for row in collect_kind_status() if row.kind == kind)


def test_nine_rows_exact_kind_set(bound_app) -> None:
    assert [row.kind for row in collect_kind_status()] == _EXPECTED_KINDS


# -- identity ------------------------------------------------------------------


def test_identity_active_single_registered(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    # ``redis`` is registered by the autouse identity fixture.
    monkeypatch.setattr(ks, "access_control_settings", lambda: SimpleNamespace(enable=True, auth_providers=["redis"]))
    row = _row("identity")
    assert row.state == "active"
    assert row.plugin == "redis"
    assert row.detail == "providers: redis"


def test_identity_active_multi_flags_unregistered(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ks, "access_control_settings", lambda: SimpleNamespace(enable=True, auth_providers=["redis", "ghost"])
    )
    row = _row("identity")
    assert row.state == "active"
    assert row.plugin == "redis, ghost"
    assert row.detail == "providers: redis, ghost (not registered)"


def test_identity_off_when_access_control_disabled(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "access_control_settings", lambda: SimpleNamespace(enable=False, auth_providers=["redis"]))
    row = _row("identity")
    assert row.state == "off"
    assert row.plugin is None
    assert row.detail == "access control disabled"


# -- accounts ------------------------------------------------------------------


def test_accounts_off_when_registry_empty(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "iter_accounts_provider_factories", list)
    row = _row("accounts")
    assert row.state == "off"
    assert row.plugin is None
    assert row.detail == "no accounts provider registered"


def test_accounts_active_when_registered(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "iter_accounts_provider_factories", lambda: [("email", object), ("phone", object)])
    row = _row("accounts")
    assert row.state == "active"
    assert row.plugin == "email, phone"
    assert row.detail == "providers: email, phone"


# -- monitoring ----------------------------------------------------------------


def test_monitoring_default_when_noop(bound_app) -> None:
    reset_monitoring()
    row = _row("monitoring")
    assert row.state == "default"
    assert row.plugin is None
    assert row.detail == "NoOpMonitoring — no recorder plugin installed"


def test_monitoring_active_when_backend_registered(bound_app) -> None:
    init_monitoring(_RealMonitoring())
    row = _row("monitoring")
    assert row.state == "active"
    assert row.plugin == _RealMonitoring.__module__
    assert row.detail == "_RealMonitoring"


# -- storage -------------------------------------------------------------------


def test_storage_off_when_no_provider(bound_app) -> None:
    row = _row("storage")
    assert row.state == "off"
    assert row.plugin is None
    assert row.detail == "dead by default — no storage provider installed"


def test_storage_active_when_provider_registered(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bound_app._storage_registry, "_provider", _FakeStorage())
    row = _row("storage")
    assert row.state == "active"
    assert row.plugin == _FakeStorage.__module__
    assert row.detail == "_FakeStorage"


# -- backend -------------------------------------------------------------------


def test_backend_off_when_empty(bound_app) -> None:
    row = _row("backend")
    assert row.state == "off"
    assert row.detail == "no backend provider installed"


def test_backend_active_when_registered(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bound_app._backend_holder, "_backend", _FakeBackend())
    row = _row("backend")
    assert row.state == "active"
    assert row.plugin == _FakeBackend.__module__
    assert row.detail == "_FakeBackend"


# -- channels ------------------------------------------------------------------


def test_channels_off_when_none_registered(bound_app) -> None:
    row = _row("channels")
    assert row.state == "off"
    assert row.detail == "no channels registered"


def test_channels_active_lists_sorted_names(bound_app) -> None:
    tai42_app.channels.register("zeta", _Channel())
    tai42_app.channels.register("alpha", _Channel())
    row = _row("channels")
    assert row.state == "active"
    assert row.plugin is None
    assert row.detail == "channels: alpha, zeta"


# -- webhook verifiers ---------------------------------------------------------


def test_webhook_verifiers_off_when_none_configured(bound_app) -> None:
    row = _row("webhook_verifiers")
    assert row.state == "off"
    assert row.detail == "no webhook verifiers configured"


def test_webhook_verifiers_active_from_live_manifest(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = Manifest.model_validate({"webhook_verifier_modules": ["pkg.a", "pkg.b"]})
    monkeypatch.setattr(bound_app, "_manifest", manifest)
    row = _row("webhook_verifiers")
    assert row.state == "active"
    assert row.detail == "modules: pkg.a, pkg.b"


# -- config --------------------------------------------------------------------


def test_config_default_for_file_mode(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "config_mode", lambda: "file")
    row = _row("config")
    assert row.state == "default"
    assert row.detail == "file — built-in default config provider"


def test_config_active_for_non_file_mode(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "config_mode", lambda: "k8s")
    row = _row("config")
    assert row.state == "active"
    assert row.detail == "mode: k8s"


# -- studio plugins ------------------------------------------------------------


def test_studio_plugins_off_when_registry_not_built(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise StudioPluginError("not built")

    monkeypatch.setattr(ks, "current_registry", _raise)
    row = _row("studio_plugins")
    assert row.state == "off"
    assert row.detail == "studio plugin registry not built"


def test_studio_plugins_off_when_built_but_empty(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "current_registry", lambda: SimpleNamespace(plugins={}))
    row = _row("studio_plugins")
    assert row.state == "off"
    assert row.detail == "0 plugins"


def test_studio_plugins_active_lists_sorted_names(bound_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ks, "current_registry", lambda: SimpleNamespace(plugins={"zeta": 1, "alpha": 2}))
    row = _row("studio_plugins")
    assert row.state == "active"
    assert row.detail == "2 plugin(s): alpha, zeta"


# -- once-per-process NoOp warning ---------------------------------------------


def test_noop_warning_fires_once_across_calls(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(ks, "_NOOP_WARNED", False)
    rows = [KindStatus(kind="monitoring", state="default", plugin=None, detail="noop")]
    log = logging.getLogger("test.kind_status")
    with caplog.at_level(logging.WARNING, logger="test.kind_status"):
        ks.warn_if_noop_monitoring(rows, log)
        ks.warn_if_noop_monitoring(rows, log)
    warnings = [r for r in caplog.records if "monitoring: OFF" in r.getMessage()]
    assert len(warnings) == 1


def test_noop_warning_silent_when_backend_active(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(ks, "_NOOP_WARNED", False)
    rows = [KindStatus(kind="monitoring", state="active", plugin="pkg", detail="Real")]
    log = logging.getLogger("test.kind_status")
    with caplog.at_level(logging.WARNING, logger="test.kind_status"):
        ks.warn_if_noop_monitoring(rows, log)
    warnings = [r for r in caplog.records if "monitoring: OFF" in r.getMessage()]
    assert warnings == []
