import asyncio
import logging
import os
import signal

import click
from dotenv import load_dotenv
from tai_kit.logging import logging_settings, setup_logging

from tai_skeleton.app import instance
from tai_skeleton.app.boot_rules import require_bus_for_backend, require_bus_for_k8s
from tai_skeleton.app.bus import OriginKind
from tai_skeleton.backend.settings import base_backend_settings
from tai_skeleton.config.config_mode import config_mode
from tai_skeleton.connectors.meta_log_redactor import install_meta_log_redactor
from tai_skeleton.manifest import Manifest

logger = logging.getLogger(__name__)


async def run_backend(extra_args):
    # Handle SIGTERM (the deployment-standard stop signal) by cancelling this main
    # task: the cancellation unwinds through ``app_context``'s ``finally`` (shutdown
    # handlers + ``_teardown_resources``) so teardown is guaranteed, where an
    # unhandled SIGTERM would kill the process outright and skip it. SIGINT is
    # already safe (KeyboardInterrupt → the runner cancels the task). Unix/uvloop
    # only, matching the CLI's target.
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    assert main_task is not None
    loop.add_signal_handler(signal.SIGTERM, main_task.cancel)

    # Worker-bus boot rule (fail loud, naming TAI_BUS_REDIS_URL). Run BEFORE the app
    # (and its config manager) is built, so a k8s-mode busless backend refuses on the
    # bus var rather than failing first on a kubeconfig connection.
    require_bus_for_k8s()

    # Obtained after ``main``'s env bootstrap so access-control settings reflect
    # a local ``.env`` rather than a pre-bootstrap default.
    app = instance.build_app()
    # This CLI-configured backend process keeps its root logger in sync across
    # config reloads; the public factory registers nothing, so an embedded host's
    # root logger stays untouched.
    instance.register_cli_logging_reload()
    # This CLI-owned process owns its whole logging surface, so the connector-secret
    # redactor covers every record in the process, not just the tai logger family.
    install_meta_log_redactor(scope="process")
    settings = base_backend_settings()
    logger.info("Configuration mode: %s", config_mode())
    manifest = Manifest.model_validate(app.config.config_manager.read_manifest())

    # A dedicated backend runtime always registers a task backend, so it requires the
    # worker bus (the backend-runtime and server processes must converge on config
    # reloads). Refuse loudly, naming TAI_BUS_REDIS_URL, before entering app_context.
    require_bus_for_backend(manifest)

    # The resolved (``!ENV``-expanded) manifest is placed into this env var for the
    # WORKER runtime only — the one that forks job children. It carries the resolved
    # secrets and is readable by same-user process inspection, so it is not exported
    # into runtimes that fork nothing (a scheduler ``beat``, a dashboard): those need
    # no manifest view, and widening the export would spread the secrets for nothing.
    if "worker" in extra_args:
        os.environ[settings.manifest_key] = manifest.model_dump_json()

    # Every backend invocation launches inside ``app_context``: ``start()`` binds
    # the global ``tai_app`` handle and THEN imports the manifest's
    # ``backend_module``, so a plugin module that registers its Backend at import
    # (``tai_app.backends.register_backend``) lands the registration in the bound
    # app's holder before ``run_backend`` reaches the registered backend's
    # ``launch`` — on the worker path and the non-worker (beat/flower-style) path
    # alike. Importing the plugin module outside the context would hit the
    # unbound handle and crash.
    async with app.app_context(manifest=manifest, origin_kind=OriginKind.backend):
        await app.run_backend(extra_args)


@click.command("backend", context_settings={"ignore_unknown_options": True})
@click.option(
    "--manifest-path",
    help="Path to a YAML manifest file (file mode only; ignored in K8s mode).",
    default=None,
    show_default=True,
)
@click.argument("extra_args", nargs=-1)
@click.pass_context
def main(ctx, manifest_path, extra_args):
    """Run a tai execution-backend runtime (worker / beat / dashboard).

    \b
    A backend runtime always registers a task backend, so it requires the worker
    bus — set TAI_BUS_REDIS_URL. The backend-runtime and server processes must
    converge on config reloads, so a busless backend is refused at boot (and again
    on any reload that would leave a backend without the bus). TAI_CONFIG_MODE=k8s
    is likewise refused without the bus.
    """
    if config_mode() != "k8s":
        load_dotenv()

    # Configure the root logger at process start, right after the env bootstrap, so
    # ``TAI_LOG_LEVEL`` (from a local ``.env`` or the environment) takes effect for
    # the backend worker's logging.
    setup_logging(logging_settings())

    if manifest_path:
        os.environ["TAI_MANIFEST_PATH"] = manifest_path

    # Label this process's tool metrics as the backend runtime so scrapes can tell
    # backend-run increments from the server's. ``setdefault`` respects an explicit
    # operator override.
    os.environ.setdefault("PROMETHEUS_RUNTIME", "backend")

    # Publish the multiproc dir BEFORE the first ``prometheus_client`` import: the
    # library freezes its value backend from this env var at that first import, and
    # importing ``routers.prometheus`` pulls ``prometheus_client`` in. So set the env
    # first, THEN import the assert helper. This process writes tool counters, so
    # verify the mmap backend froze rather than the in-process mutex whose counters
    # no scrape sees.
    from tai_skeleton.routers.metrics_settings import activate_multiproc_env

    activate_multiproc_env()

    from tai_skeleton.routers.prometheus import assert_multiproc_value_class

    assert_multiproc_value_class()

    try:
        import uvloop

        loop = uvloop
    except ImportError:
        loop = asyncio

    all_args = list(ctx.args) + list(extra_args)
    try:
        loop.run(run_backend(all_args))
    except asyncio.CancelledError:
        # Only the SIGTERM handler cancels the main task, so this is the deliberate
        # signal-driven stop — teardown already ran through ``app_context``'s
        # ``finally`` during the unwind. Convert it into a clean exit.
        logger.info("SIGTERM received — backend worker shut down after teardown")


if __name__ == "__main__":
    main()
