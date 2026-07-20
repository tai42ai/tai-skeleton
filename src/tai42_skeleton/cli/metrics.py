import importlib
import logging
from typing import Any, cast

import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from tai42_contract.access_control.registry import get_identity_provider_factory
from tai42_kit.logging import logging_settings, setup_logging

from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai42_skeleton.config.config_mode import config_mode
from tai42_skeleton.config.factory import ConfigManagerFactory
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.routers.metrics_settings import activate_multiproc_env, metrics_settings

logger = logging.getLogger(__name__)


async def get_metrics() -> Response:
    # Imported lazily: ``tai42_skeleton.routers.prometheus`` imports
    # ``prometheus_client``, which freezes its value backend (mmap vs in-process
    # mutex) at first import based on ``PROMETHEUS_MULTIPROC_DIR``. Registering
    # this command must not pull that import in — otherwise a `tai serve` master,
    # which loads every command module before its own entrypoint stamps the env,
    # would freeze the mutex backend and lose every tool counter.
    from tai42_skeleton.routers.prometheus import render_multiproc_metrics

    return Response(render_multiproc_metrics(), media_type="text/plain")


def _identity_provider_registered(name: str) -> bool:
    try:
        get_identity_provider_factory(name)
        return True
    except KeyError:
        return False


def _register_manifest_identity_provider(settings: AccessControlSettings) -> None:
    """Populate the identity-provider registry for the metrics process.

    The metrics entrypoint builds its OWN ``AuthAdapter`` to guard ``/metrics`` but
    never runs ``start()``, so nothing has imported the manifest's identity plugin
    and the module-level registry is empty — token verification could not resolve the
    configured provider. Mirror the served path: read the manifest and import its
    ``lifecycle_modules`` (the same import-only home the identity plugin registers
    through), stopping as soon as the configured provider registers.

    A lifecycle module that cannot import in this app-less process (e.g. one that
    touches the unbound ``tai42_app`` handle, like a webhook verifier) is not an
    identity provider — it is logged and skipped. If, after importing them all, the
    configured provider is STILL unregistered, RAISE loudly: ``/metrics`` must never
    come up un-authenticatable.
    """
    if not settings.enable:
        # Access control off → the adapter adds no auth middleware, so no token is
        # ever verified and no identity provider is needed.
        return

    def _missing() -> list[str]:
        return [name for name in settings.auth_providers if not _identity_provider_registered(name)]

    if not _missing():
        return

    manifest = Manifest.model_validate(ConfigManagerFactory.create().read_manifest())
    for module in manifest.lifecycle_modules or []:
        try:
            importlib.import_module(module)
        except Exception:
            # The metrics process needs ONLY the identity plugins, which register via
            # a plain module import with no app handle. Any other lifecycle module
            # (e.g. one registering through the unbound tai42_app) legitimately cannot
            # import here — logged (never silent) and skipped.
            logger.warning(
                "metrics: skipped manifest lifecycle module %r (not importable in the app-less metrics process)",
                module,
                exc_info=True,
            )
        if not _missing():
            return

    raise RuntimeError(
        f"identity providers {_missing()!r} are not registered after importing the manifest's "
        "lifecycle_modules — /metrics cannot authenticate. Name the identity plugin(s) (e.g. "
        "tai42_identity_redis.redis_api_key_provider) in the manifest lifecycle_modules."
    )


def create_app() -> FastAPI:
    """Build the metrics app: access-control middleware plus the ``/metrics``
    route.

    Reads access-control settings, so it must run after the env bootstrap
    (``load_dotenv`` in ``main``) for a local ``.env`` to take effect.
    """
    app = FastAPI()

    settings = access_control_settings()
    # The metrics process builds its OWN AuthAdapter and never runs start(), so it
    # imports the manifest's identity plugin here — otherwise the registry is empty
    # and /metrics token verification cannot resolve the configured provider.
    _register_manifest_identity_provider(settings)
    auth_adapter = AuthAdapter(settings)
    for middleware in reversed(auth_adapter.get_middleware()):
        # ``add_middleware`` is ParamSpec-typed against the middleware class; a
        # starlette ``Middleware`` carries its class + kwargs erased, so the
        # ParamSpec can't be bound here.
        app.add_middleware(cast(Any, middleware.cls), **middleware.kwargs)

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
