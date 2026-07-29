"""Prometheus metrics exporter.

A plain Prometheus exporter over the shared multiprocess directory. Binds
localhost by default; operators govern exposure via the bind host and cluster
network policy, never via in-process auth.
"""

import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from tai42_kit.logging import logging_settings, setup_logging

from tai42_skeleton.config.config_mode import config_mode
from tai42_skeleton.routers.metrics_settings import activate_multiproc_env, metrics_settings


async def get_metrics() -> Response:
    # Imported lazily: ``tai42_skeleton.routers.prometheus`` imports
    # ``prometheus_client``, which freezes its value backend (mmap vs in-process
    # mutex) at first import based on ``PROMETHEUS_MULTIPROC_DIR``. Registering
    # this command must not pull that import in — otherwise a `tai serve` master,
    # which loads every command module before its own entrypoint stamps the env,
    # would freeze the mutex backend and lose every tool counter.
    from tai42_skeleton.routers.prometheus import render_multiproc_metrics

    return Response(render_multiproc_metrics(), media_type="text/plain")


def create_app() -> FastAPI:
    app = FastAPI()

    # Lazily imported (see ``get_metrics``): keep ``prometheus_client`` out of
    # the CLI-registration import path so a `tai serve` master freezes the mmap
    # backend, not the mutex one.
    from tai42_skeleton.routers.prometheus import init_prometheus_multiproc_dir

    init_prometheus_multiproc_dir()
    app.add_api_route("/metrics", get_metrics, methods=["GET"])
    return app


@click.command()
@click.option("--host", default=None, help="Host to bind the server to")
@click.option("--port", default=None, type=int, help="Port to run the server on")
def main(host: str | None, port: int | None):
    """Serve the Prometheus metrics endpoint."""
    if config_mode() != "k8s":
        load_dotenv()

    # Configure the root logger at process start, right after the env bootstrap, so
    # ``TAI_LOG_LEVEL`` takes effect; the metrics server runs in-process here, so
    # ``main`` alone covers it.
    setup_logging(logging_settings())

    # Publish the multiproc dir so the collector reads the shared run-family dir at
    # scrape time, BEFORE the lazy prometheus import in ``create_app`` freezes the
    # value backend. This is a pure READER: it never writes counters, so its own
    # value class does not matter — only the collector's read target must point at
    # the right dir.
    activate_multiproc_env()

    # Read settings after the bootstrap so a local ``.env`` is in effect.
    settings = metrics_settings()
    app = create_app()
    uvicorn.run(
        app,
        host=host if host is not None else settings.backend_metrics_host,
        port=port if port is not None else settings.backend_metrics_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
