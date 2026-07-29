"""The in-app Prometheus ``GET /metrics`` scrape route.

Serves THIS process's exposition — the body :func:`render_metrics` produces,
which auto-selects multiproc-merge vs in-process rendering from the frozen
``prometheus_client`` value backend (never by sniffing the environment). A
``tai serve`` worker renders the merged multiproc db (the same bytes the dedicated
metrics server in :mod:`tai42_skeleton.cli.metrics` serves off its own port); an
embedded in-process app (no ``PROMETHEUS_MULTIPROC_DIR``) renders its own default
registry. The registration lives in its own router module, not in
:mod:`tai42_skeleton.routers.prometheus`, because that module is imported early by
a worker (for :func:`assert_multiproc_value_class`) before the app handle is bound;
a router module is imported at ``start()`` with the app bound, like every other
route.

Auth posture: ``authed=True`` — the scrape is governed by the operator's
route→resource table rather than served unconditionally public. An
operator maps ``/metrics`` to the resource its scrapers hold (or to the public
marker for an internal-only network); an unmapped ``/metrics`` fails closed. The
route is a concrete non-``/api`` GET, so it joins the derived reserved set and the
SPA-shell fallback skips it.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from tai42_contract.app import tai42_app

from tai42_skeleton.routers.prometheus import render_metrics


@tai42_app.http.custom_route(
    "/metrics",
    methods=["GET"],
    summary="Prometheus metrics scrape",
    tags=["metrics"],
    response_model=None,
    authed=True,
    action="read",
)
async def metrics_scrape(request: Request) -> Response:
    return Response(render_metrics(), media_type="text/plain")
