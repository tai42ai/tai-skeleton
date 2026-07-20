import asyncio
import os

from prometheus_client import CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response
from tai42_contract.app import tai42_app

from tai42_skeleton.routers.prometheus import init_prometheus_multiproc_dir, render_metrics

# Ensure the multiproc dir exists only when a multiproc environment is claimed —
# every CLI process (master/worker/sidecar) sets ``PROMETHEUS_MULTIPROC_DIR``
# before this module imports. An embedded process serves the in-process registry
# and must not create an unused dir on the host filesystem.
if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
    init_prometheus_multiproc_dir()


@tai42_app.http.custom_route(
    "/metrics",
    methods=["GET"],
    summary="Prometheus metrics exposition endpoint",
    tags=["metrics"],
    response_model=None,
    authed=False,
)
async def metrics_endpoint(request: Request) -> Response:
    # ``render_metrics`` may do a blocking mmap/merge of every worker's db file (in
    # multiproc mode); run it on a worker thread so a scrape never stalls the
    # serving loop (which also carries /mcp traffic). Serve the canonical Prometheus
    # content type so scrapers get the format version, not a bare ``text/plain``.
    body = await asyncio.to_thread(render_metrics)
    return Response(body, media_type=CONTENT_TYPE_LATEST)
