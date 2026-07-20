"""The process app singleton: ``instance.app`` is a built ``TaiMCP`` with the
catalog refresh wired as a startup+reload handler, and its ``lifespan`` context
opens (and, on exit, closes) the sub-app router lifespan. Process-wide resource
teardown lives on ``app_context`` (see ``test_lifecycle``), not here.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from tai42_contract.connectors.models import ConnectorRef
from tai42_contract.manifest import MCPConfig, TaiMCPConfig
from tai42_kit.settings import reset_all_settings

import tai42_skeleton.app.instance as instance
from tai42_skeleton.app.server import TaiMCP
from tai42_skeleton.manifest import Manifest


def test_app_singleton_is_taimcp():
    assert isinstance(instance.app, TaiMCP)
    assert instance.app.fastmcp.name == "Tai"


def test_build_app_is_idempotent():
    assert instance.build_app() is instance.build_app()
    assert instance.build_app() is instance.app


def test_refresh_catalog_wired_as_startup_and_reload():
    assert instance.refresh_catalog_if_connectors_in_use in instance.app._startup_handlers.values()
    assert instance.refresh_catalog_if_connectors_in_use in instance.app._reload_handlers.values()


def test_rehydrate_presets_wired_as_startup_and_reload():
    # Versioned presets rehydrate at boot AND on every in-place reload, so a
    # persisted preset survives a restart and a reload_config().
    assert instance.rehydrate_versioned_presets_if_store_in_use in instance.app._startup_handlers.values()
    assert instance.rehydrate_versioned_presets_if_store_in_use in instance.app._reload_handlers.values()


def test_access_control_probe_wired_as_startup_when_enabled():
    # The process app singleton is built with access control enabled (the default), so
    # the active identity provider's healthcheck must be registered as a startup
    # handler: it is THE boot probe. Dropping it silently re-opens the boot trap (a
    # broken backend boots clean and dies on the first authenticated request), so this
    # pins the wiring, not just the probe function.
    from tai42_skeleton.access_control.startup import probe_identity_provider

    startup_handlers = instance.app._startup_handlers.values()
    assert probe_identity_provider in startup_handlers


# --- catalog-refresh gate ---------------------------------------------------


def _clear_connector_env(monkeypatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith(("CONNECTORS_", "CONNECTOR_STORE_")):
            monkeypatch.delenv(key, raising=False)


def _record_refresh(monkeypatch) -> list[bool]:
    calls: list[bool] = []

    async def fake_refresh() -> None:
        calls.append(True)

    monkeypatch.setattr(instance, "refresh_catalog", fake_refresh)
    return calls


async def test_catalog_refresh_skipped_when_connectors_unused(monkeypatch):
    # No managed manifest entries, no registered providers, no connector env:
    # the handler skips the catalog load (no Postgres at boot).
    _clear_connector_env(monkeypatch)
    monkeypatch.setattr(instance, "list_providers", list)
    monkeypatch.setattr(instance.build_app(), "_manifest", Manifest())
    calls = _record_refresh(monkeypatch)

    await instance.refresh_catalog_if_connectors_in_use()

    assert calls == []


async def test_catalog_refresh_runs_when_connector_env_present(monkeypatch):
    _clear_connector_env(monkeypatch)
    monkeypatch.setattr(instance, "list_providers", list)
    monkeypatch.setattr(instance.build_app(), "_manifest", Manifest())
    monkeypatch.setenv("CONNECTORS_KEK", "some-key")
    calls = _record_refresh(monkeypatch)

    await instance.refresh_catalog_if_connectors_in_use()

    assert calls == [True]


async def test_catalog_refresh_runs_when_manifest_declares_managed_entry(monkeypatch):
    _clear_connector_env(monkeypatch)
    monkeypatch.setattr(instance, "list_providers", list)
    manifest = Manifest(
        mcp=[
            TaiMCPConfig(
                title="acme_mail_work",
                include=[],
                exclude=[],
                config=MCPConfig(type="http", url="https://mcp.acme.test/"),
                managed=ConnectorRef(
                    connection_id="11111111-1111-4111-8111-111111111111",
                    provider_id="acme",
                    sub_service="mail",
                ),
            )
        ]
    )
    monkeypatch.setattr(instance.build_app(), "_manifest", manifest)
    calls = _record_refresh(monkeypatch)

    await instance.refresh_catalog_if_connectors_in_use()

    assert calls == [True]


async def test_catalog_refresh_runs_when_a_provider_is_registered(monkeypatch):
    _clear_connector_env(monkeypatch)
    monkeypatch.setattr(instance, "list_providers", lambda: [object()])
    monkeypatch.setattr(instance.build_app(), "_manifest", Manifest())
    calls = _record_refresh(monkeypatch)

    await instance.refresh_catalog_if_connectors_in_use()

    assert calls == [True]


# --- versioned-preset rehydration gate --------------------------------------


def _clear_versioning_env(monkeypatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("VERSIONING_STORE_"):
            monkeypatch.delenv(key, raising=False)


def _record_rehydrate(monkeypatch) -> list[bool]:
    calls: list[bool] = []

    async def fake_rehydrate() -> None:
        calls.append(True)

    monkeypatch.setattr(instance.build_app().preset_manager, "rehydrate", fake_rehydrate)
    return calls


async def test_rehydrate_skipped_when_versioning_store_unused(monkeypatch):
    # No VERSIONING_STORE_* env: the handler skips the load (no Postgres at boot).
    _clear_versioning_env(monkeypatch)
    calls = _record_rehydrate(monkeypatch)

    await instance.rehydrate_versioned_presets_if_store_in_use()

    assert calls == []


async def test_rehydrate_runs_when_versioning_store_env_present(monkeypatch):
    _clear_versioning_env(monkeypatch)
    monkeypatch.setenv("VERSIONING_STORE_PG_PASSWORD", "secret")
    calls = _record_rehydrate(monkeypatch)

    await instance.rehydrate_versioned_presets_if_store_in_use()

    assert calls == [True]


# --- logging reload handler -------------------------------------------------


def test_apply_logging_wired_only_by_cli_seam_registration(monkeypatch):
    # A bare ``build_app()`` does NOT register the root-logger reload handler — an
    # embedded app never reconfigures the host's logging. The CLI seams register it
    # explicitly via ``register_cli_logging_reload``, and only as a reload handler
    # (never a startup handler; process start is covered by the CLI's own
    # ``setup_logging`` call). The singleton is reset first so a prior CLI-seam
    # registration on the process singleton (e.g. from the backend beat test) cannot
    # spuriously FAIL the absence assertion by collection-order luck.
    monkeypatch.setattr(instance, "_app", None)

    app = instance.build_app()
    assert instance.apply_logging_settings not in app._reload_handlers.values()
    assert instance.apply_logging_settings not in app._startup_handlers.values()

    instance.register_cli_logging_reload()
    assert instance.apply_logging_settings in app._reload_handlers.values()
    assert instance.apply_logging_settings not in app._startup_handlers.values()


def test_build_app_installs_redactor_at_tai_scope(monkeypatch):
    # The REAL ``build_app()`` wire: it installs the connector-secret redactor at
    # its default ``tai`` scope — a tai-family record is scrubbed, a host-app
    # record passes through untouched. Factory and scope are snapshotted and
    # restored so this test neither depends on nor leaks redactor state; the
    # singleton is reset so the first-build branch (where the install lives)
    # actually runs.
    from tai42_skeleton.connectors import meta_log_redactor

    saved_factory = logging.getLogRecordFactory()
    saved_scope = meta_log_redactor._SCOPE
    logging.setLogRecordFactory(logging.LogRecord)
    meta_log_redactor._SCOPE = "tai"
    monkeypatch.setattr(instance, "_app", None)
    try:
        instance.build_app()

        factory = logging.getLogRecordFactory()
        secret = '{"_meta": {"tai42_hub.access_token": "WIRE-SECRET"}}'
        tai42_rec = factory("tai42_skeleton.connectors", logging.INFO, __file__, 1, secret, None, None)
        host_rec = factory("myhost.app", logging.INFO, __file__, 1, secret, None, None)
        assert "WIRE-SECRET" not in tai42_rec.getMessage()
        assert "WIRE-SECRET" in host_rec.getMessage()
    finally:
        logging.setLogRecordFactory(saved_factory)
        meta_log_redactor._SCOPE = saved_scope


def test_apply_logging_settings_applies_configured_level(monkeypatch, root_logger_restored):
    monkeypatch.setenv("TAI_LOG_LEVEL", "debug")
    reset_all_settings()

    instance.apply_logging_settings()

    assert root_logger_restored.level == logging.DEBUG
    # A root handler with the kit format (which names the logger) is installed.
    formatter = root_logger_restored.handlers[0].formatter
    assert formatter is not None
    assert "%(name)s" in formatter._fmt  # type: ignore[union-attr]


def test_apply_logging_settings_reapplies_after_level_change(monkeypatch, root_logger_restored):
    # The reload path runs reset_all_settings() before the handler, so a changed
    # TAI_LOG_LEVEL is re-read and re-applied without a process restart.
    monkeypatch.setenv("TAI_LOG_LEVEL", "warning")
    reset_all_settings()
    instance.apply_logging_settings()
    assert root_logger_restored.level == logging.WARNING

    monkeypatch.setenv("TAI_LOG_LEVEL", "error")
    reset_all_settings()
    instance.apply_logging_settings()
    assert root_logger_restored.level == logging.ERROR


async def test_lifespan_enters_and_exits_router_lifespan(monkeypatch):
    # The lifespan opens the sub-app router lifespan and closes it on exit; it
    # does not itself tear down resources (that is app_context's job).
    events: list[str] = []

    @asynccontextmanager
    async def fake_router_lifespan(_app):
        events.append("router-open")
        try:
            yield
        finally:
            events.append("router-close")

    monkeypatch.setattr(instance.app.sub_app.mcp_sub_app_router, "lifespan", fake_router_lifespan)

    async with instance.lifespan(instance.app):
        assert events == ["router-open"]

    assert events == ["router-open", "router-close"]
