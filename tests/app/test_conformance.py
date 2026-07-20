"""The concrete ``TaiMCP`` satisfies the ``tai_contract.app`` facade + every facet.

Builds a bare ``TaiMCP`` and asserts it is a structural instance of the assembled
``TaiApp`` protocol and of each of the 18 per-feature sub-protocols, so the flat
impl surface stays correctly partitioned onto the contract namespaces.
"""

from __future__ import annotations

import pytest
from tai_contract.app import (
    AppAdmin,
    AppAgents,
    AppBackends,
    AppBackup,
    AppChannels,
    AppClients,
    AppConfig,
    AppConnectors,
    AppExtensions,
    AppHttp,
    AppLifecycle,
    AppMonitoring,
    AppPresets,
    AppStorage,
    AppSubApp,
    AppVersioning,
    AppWebhookVerifiers,
    TaiApp,
)
from tai_contract.tools import AppTools

from tai_skeleton.app.server import TaiMCP


@pytest.fixture(scope="module")
def app() -> TaiMCP:
    # Constructing a TaiMCP does not touch the global ``tai_app`` handle (only
    # start()/app_context binds), so this throwaway app for the structural
    # checks is safe to build without saving/restoring the handle.
    return TaiMCP(name="conformance")


def test_app_satisfies_tai_app(app: TaiMCP) -> None:
    assert isinstance(app, TaiApp)


@pytest.mark.parametrize(
    ("namespace", "protocol"),
    [
        ("tools", AppTools),
        ("agents", AppAgents),
        ("backends", AppBackends),
        ("storage", AppStorage),
        ("connectors", AppConnectors),
        ("monitoring", AppMonitoring),
        ("extensions", AppExtensions),
        ("http", AppHttp),
        ("clients", AppClients),
        ("lifecycle", AppLifecycle),
        ("admin", AppAdmin),
        ("config", AppConfig),
        ("sub_app", AppSubApp),
        ("backup", AppBackup),
        ("versioning", AppVersioning),
        ("presets", AppPresets),
        ("webhook_verifiers", AppWebhookVerifiers),
        ("channels", AppChannels),
    ],
)
def test_facet_satisfies_protocol(app: TaiMCP, namespace: str, protocol: type) -> None:
    facet = getattr(app, namespace)
    assert isinstance(facet, protocol), f"app.{namespace} does not satisfy {protocol.__name__}"
