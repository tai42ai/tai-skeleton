"""Fixture registering a channel ON IMPORT.

Loaded via a manifest ``channel_modules`` entry so each ``start()`` re-imports
it and re-runs the ``tai42_app.channels.register(...)`` side-effect — exactly as a
real channel plugin module does, including binding the plugin's own public
inbound route at import. The registry is reset each ``start()``, so the repeated
registration is clean, never a duplicate-name crash.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app
from tai42_contract.channels import ChannelDelivery

from tests._helpers import DeliverOnlyChannel


class _FixtureChannel(DeliverOnlyChannel):
    async def deliver(self, delivery: ChannelDelivery) -> None:
        return None


@tai42_app.http.custom_route(
    # Deliberately OFF the ``/api/`` prefix: the route registry is process-global,
    # and an ``/api/*`` fixture route would leak into the CLI-parity and OpenAPI
    # coverage gates, which enumerate every recorded ``/api/*`` route.
    "/channels-fixture/inbound",
    methods=["POST"],
    summary="Fixture channel inbound door",
    tags=["channels"],
    response_model=None,
    authed=False,
)
async def fixture_inbound(request: Request) -> Response:
    return JSONResponse({"data": {"ok": True}})


tai42_app.channels.register("fixture_channel", _FixtureChannel())
