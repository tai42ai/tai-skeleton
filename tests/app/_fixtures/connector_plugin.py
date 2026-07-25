"""Fixture connector-plugin module.

Mirrors a real connector plugin: at import time it calls
``tai42_app.connectors.register_connector(descriptor)``. Listed in a manifest
module list so every ``start()`` / reload re-imports it and re-runs the
registration — the case that crashes a reload unless the provider registry is
reset first.
"""

from tai42_contract.app import tai42_app
from tai42_contract.connectors.providers import McpServerDescriptor, ProviderDescriptor, SubServiceDescriptor

PROVIDER_ID = "fixture_conn"

tai42_app.connectors.register_connector(
    ProviderDescriptor(
        id=PROVIDER_ID,
        display_name="Fixture Connector",
        icon_url="https://fixture.test/icon.png",
        kind="none",
        origin="system",
        category="data",
        sub_services={
            "main": SubServiceDescriptor(
                id="main",
                display_name="Main",
                mcp_server=McpServerDescriptor(type="http", url="https://fixture.test/mcp"),
            ),
        },
    )
)
