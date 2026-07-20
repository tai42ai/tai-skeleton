import asyncio
import logging
import os
import socket
import stat
import sys
import uuid
from typing import Any, cast

import click
import uvicorn
from dotenv import load_dotenv
from tai_kit.logging import logging_settings, setup_logging
from tai_kit.utils.runtime.uvicorn_util import parse_and_validate_uvicorn_args

from tai_skeleton import asgi
from tai_skeleton.app import instance
from tai_skeleton.app.boot_rules import require_bus_for_k8s, require_bus_for_workers
from tai_skeleton.config.config_mode import config_mode
from tai_skeleton.connectors.meta_log_redactor import install_meta_log_redactor
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.manifest import Manifest
from tai_skeleton.settings.cache import app_args_settings
from tai_skeleton.settings.cache import manifest_path as default_manifest_path

logger = logging.getLogger(__name__)

# The streamable-HTTP transports fastmcp can serve statelessly.
_HTTP_TRANSPORTS = frozenset({"http", "streamable-http"})
# Every transport whose MCP sessions are pinned to the worker that created them —
# the streamable-HTTP session manager and the SSE app both keep per-process state,
# so a second worker breaks any session it did not open.
_STATEFUL_TRANSPORTS = frozenset({"http", "streamable-http", "sse"})


def create_app():
    # Configure the root logger in every uvicorn worker process: multi-worker
    # ``uvicorn.run(..., factory=True)`` imports this factory string and never runs
    # ``main()``, so without this the workers (which do all the real logging) stay
    # unconfigured. ``force=True`` inside ``setup_logging`` makes the repeat call in
    # a single-process run harmless.
    setup_logging(logging_settings())

    # This worker is a metrics WRITER (its tool calls increment counters), so the
    # multiproc mmap backend must have frozen — verify before serving. The worker
    # inherits ``PROMETHEUS_MULTIPROC_DIR`` from the master (set in ``run_mcp_app``
    # before spawn), so this holds; if it does not, fail loudly rather than record
    # counters no scrape can see.
    from tai_skeleton.routers.prometheus import assert_multiproc_value_class

    assert_multiproc_value_class()

    # This CLI-configured worker keeps its root logger in sync across config
    # reloads. The public factory registers nothing, so an embedded host's root
    # logger stays untouched.
    instance.register_cli_logging_reload()

    # This CLI-owned worker owns its whole logging surface, so the connector-secret
    # redactor covers every record in the process, not just the tai logger family.
    install_meta_log_redactor(scope="process")

    # ``run_mcp_app`` stamps ``TAI_TRANSPORT``/``TAI_STATELESS_HTTP`` beside the
    # manifest path so the flags travel to this uvicorn factory worker (a factory
    # import string carries no arguments). Read them and delegate to the public
    # factory, which validates the transport at call time and raises on anything
    # outside the accepted literals.
    transport = os.getenv("TAI_TRANSPORT", "http")
    stateless_http = os.getenv("TAI_STATELESS_HTTP") == "1"
    return asgi.create_app(transport=cast("asgi.Transport", transport), stateless_http=stateless_http)


async def run_stdio():
    # The stdio server writes tool counters in this process, so the multiproc mmap
    # backend must have frozen (``run_mcp_app`` set the env before calling this).
    from tai_skeleton.routers.prometheus import assert_multiproc_value_class

    assert_multiproc_value_class()

    app = instance.build_app()
    # This CLI-configured process keeps its root logger in sync across config reloads.
    instance.register_cli_logging_reload()
    # This CLI-owned process owns its whole logging surface, so the connector-secret
    # redactor covers every record in the process, not just the tai logger family.
    install_meta_log_redactor(scope="process")
    manifest = Manifest.model_validate(app.config.config_manager.read_manifest())
    async with app.app_context(manifest):
        await app.run_async(transport="stdio")
    return 0


async def run_debug(transport: str, config_kwargs: dict[str, Any], stateless_http: bool = False):
    # The debug server writes tool counters in this process, so the multiproc mmap
    # backend must have frozen (``run_mcp_app`` set the env before calling this).
    from tai_skeleton.routers.prometheus import assert_multiproc_value_class

    assert_multiproc_value_class()

    app = instance.build_app()
    # This CLI-configured process keeps its root logger in sync across config reloads.
    instance.register_cli_logging_reload()
    # This CLI-owned process owns its whole logging surface, so the connector-secret
    # redactor covers every record in the process, not just the tai logger family.
    install_meta_log_redactor(scope="process")
    manifest = Manifest.model_validate(app.config.config_manager.read_manifest())

    async with app.app_context(manifest):
        if transport == "sse":
            config_kwargs["app"] = app.sse_app()
        elif stateless_http:
            config_kwargs["app"] = app.http_app(stateless_http=True)
        else:
            config_kwargs["app"] = app.http_app()
        await uvicorn.Server(uvicorn.Config(**config_kwargs)).serve()
    return 0


def _prepare_uds_path(uds: str) -> None:
    """Clear a stale UDS socket before binding, or refuse a live/foreign path.

    A missing path binds fresh. An existing socket path is probed with a
    ``connect``: a connectable socket is a live server, so binding is refused and
    the socket is left untouched; a connection-refused socket is stale from an
    unclean shutdown and is unlinked so the fresh bind succeeds. An existing path
    that is NOT a socket is refused and never unlinked — it is not this command's
    file to delete. An unlink failure raises rather than proceeding onto a doomed
    bind.
    """
    try:
        mode = os.stat(uds).st_mode
    except FileNotFoundError:
        return

    if not stat.S_ISSOCK(mode):
        raise click.BadParameter(
            f"path '{uds}' already exists and is not a socket; remove it or choose another path.",
            param_hint="'--uds'",
        )

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        try:
            probe.connect(uds)
        except (ConnectionRefusedError, FileNotFoundError):
            stale = True
        else:
            stale = False

    if not stale:
        raise click.BadParameter(
            f"a server is already running at '{uds}' (its socket answered a connect probe).",
            param_hint="'--uds'",
        )

    try:
        os.unlink(uds)
    except OSError as exc:
        raise click.BadParameter(f"could not remove the stale socket at '{uds}': {exc}", param_hint="'--uds'") from exc
    logger.info("Removed stale Unix Domain Socket before binding: %s", uds)


def run_mcp_app(
    manifest_path: str,
    transport: str,
    host: str,
    port: int,
    workers: int,
    uds: str | None = None,
    stateless_http: bool = False,
    uvicorn_kwargs: dict[str, Any] | None = None,
) -> int:
    # Configure logging for this served process: the multi-worker master and the
    # in-process stdio/debug servers this dispatches to all run through here (the
    # shipped ``tai serve`` reaches this via ``cli``, not ``main``). Uvicorn workers
    # are separate processes that reconfigure via ``create_app``. ``force=True`` inside
    # ``setup_logging`` keeps a repeat call idempotent.
    setup_logging(logging_settings())

    if uvicorn_kwargs is None:
        uvicorn_kwargs = {}

    if workers < 1:
        raise click.BadParameter("Number of workers must be at least 1.", param_hint="'-w'/'--workers'")

    if uds and sys.platform == "win32":
        raise click.BadParameter("Unix Domain Sockets are not supported on Windows.")

    if uds and transport == "stdio":
        raise click.BadParameter("'--uds' cannot be used with '--transport stdio'.")

    defaults = app_args_settings()
    if (transport == "stdio" or uds) and (host != defaults.host or port != defaults.port):
        raise click.BadParameter("Host and port should not be set when using 'stdio' transport or '--uds'.")

    # ``--stateless-http`` only means anything for the streamable-HTTP transports:
    # 'sse' pins each session to one worker (it has no stateless mode) and 'stdio'
    # has no HTTP sessions at all, so pairing it with either is a usage error.
    if stateless_http and transport not in _HTTP_TRANSPORTS:
        raise click.BadParameter(
            f"'--stateless-http' requires an http transport (got '{transport}'); "
            "use '--transport http' or drop '--stateless-http'.",
            param_hint="'--stateless-http'",
        )

    # Every HTTP/SSE transport is stateful per process, so a second worker breaks
    # any session it did not create. The http/streamable-http transports lift this
    # only under '--stateless-http'; 'sse' and 'stdio' never do.
    if workers > 1:
        if transport == "stdio":
            raise click.BadParameter(
                "Multiple workers are not supported with 'stdio' transport; run one worker.",
                param_hint="'-w'/'--workers'",
            )
        if transport in _STATEFUL_TRANSPORTS and not stateless_http:
            fix = (
                "run one worker, or pass '--stateless-http' with an http transport"
                if transport in _HTTP_TRANSPORTS
                else "run one worker (the 'sse' transport has no stateless mode)"
            )
            raise click.BadParameter(
                f"Multiple workers are not supported with the stateful '{transport}' transport "
                f"because each MCP session is pinned to the worker that created it. To fix: {fix}.",
                param_hint="'-w'/'--workers'",
            )

    # Worker-bus boot rules (fail loud, naming TAI_BUS_REDIS_URL). Run BEFORE any
    # config-manager construction below, so a k8s-mode busless boot refuses on the
    # bus var rather than failing first on a kubeconfig connection. The workers rule
    # lives only here in our CLI — an external process manager driving the ASGI
    # factory with its own --workers bypasses it (a documented limitation); the k8s
    # and backend rules also run at the app_context seam.
    require_bus_for_k8s()
    require_bus_for_workers(workers)

    os.environ["TAI_MANIFEST_PATH"] = manifest_path
    os.environ["TAI_TRANSPORT"] = transport
    # The worker uvicorn factory reads this env flag (a factory import string
    # carries no arguments); clear it otherwise so a prior run cannot leak in.
    if stateless_http:
        os.environ["TAI_STATELESS_HTTP"] = "1"
    else:
        os.environ.pop("TAI_STATELESS_HTTP", None)
    # Stamp a per-run id inherited by every forked worker, so the Prometheus
    # multiproc-dir wipe fires once per run and is not skipped when consecutive
    # runs happen to share a parent pid.
    os.environ["TAI_METRICS_RUN_ID"] = uuid.uuid4().hex

    # Publish the multiproc dir to the environment BEFORE importing anything that
    # pulls in ``prometheus_client``: the library freezes its value backend (mmap
    # vs in-process mutex) at first import from this env var. Spawned uvicorn
    # workers and the in-process stdio/debug writers inherit it. Import order is
    # load-bearing — set the env first, then import the wipe (its module imports
    # ``prometheus_client``), then launch.
    from tai_skeleton.routers.metrics_settings import activate_multiproc_env

    activate_multiproc_env()

    # The master owns the once-per-run wipe of stale db files; every worker/reader
    # path only ensures the dir exists. Wipe here, before workers are spawned.
    from tai_skeleton.routers.prometheus import wipe_prometheus_multiproc_dir

    wipe_prometheus_multiproc_dir()

    if transport == "stdio":
        return asyncio.run(run_stdio())

    config_kwargs: dict[str, Any] = {
        "ws": "wsproto",
        "loop": "auto",
        "http": "auto",
        # Settings-backed default bounding uvicorn's wait for in-flight requests on
        # SIGTERM so teardown always runs. Inserted BEFORE the extra-arg update so a
        # shipped ``--timeout-graceful-shutdown`` CLI arg still wins. Feeds both the
        # served (``uvicorn.run``) and debug (``uvicorn.Config``) paths.
        "timeout_graceful_shutdown": app_args_settings().timeout_graceful_shutdown,
    }
    config_kwargs.update(uvicorn_kwargs)

    if uds:
        _prepare_uds_path(uds)
        logger.info(f"Binding to Unix Domain Socket: {uds}")
        config_kwargs["uds"] = uds

    else:
        logger.info(f"Binding to TCP: {host}:{port}")
        config_kwargs["host"] = host
        config_kwargs["port"] = port

    run_mode = (os.environ.get("TAI_RUN_MODE") or "").strip()
    if run_mode.lower() == "debug":
        if workers > 1:
            logger.info(
                "TAI_RUN_MODE=debug: running a single in-process server; the requested --workers=%d is ignored.",
                workers,
            )
        else:
            logger.info("TAI_RUN_MODE=debug: running a single in-process server; --workers is ignored.")
        return asyncio.run(run_debug(transport, config_kwargs, stateless_http))

    # An unrecognized run mode fails loudly rather than silently falling through to
    # the normal path — a bad value never silently selects a mode.
    if run_mode:
        raise click.ClickException(
            f"TAI_RUN_MODE={run_mode!r} is not a recognized run mode; accepted values are "
            "unset/empty (multi-worker server) or 'debug' (single in-process server)."
        )

    logger.info("Starting Tai MCP Server with %d worker(s).", workers)
    uvicorn.run("tai_skeleton.cli.mcp_app:create_app", workers=workers, factory=True, **config_kwargs)

    logger.info("Tai MCP Server shutdown.")
    return 0


@click.command("tai-mcp-server", context_settings={"ignore_unknown_options": True})
@click.option(
    "--manifest-path",
    help="Path to a YAML manifest file.",
    default=None,
)
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "http", "sse", "streamable-http"], case_sensitive=False),
    default=None,
    help="Transport mechanism.",
)
@click.option(
    "--host",
    default=None,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port number to bind the server to.",
)
@click.option(
    "--uds",
    type=str,
    default=None,
    help="Unix Domain Socket path.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=1,
    help="Number of worker processes. More than one requires the worker bus (set TAI_BUS_REDIS_URL).",
    show_default=True,
)
@click.option(
    "--stateless-http/--no-stateless-http",
    default=False,
    show_default=True,
    help=(
        "Run an http/streamable-http transport in fastmcp's stateless mode, which lifts the "
        "single-worker restriction for those transports. Refused with 'sse'/'stdio'."
    ),
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def cli(
    manifest_path: str | None,
    transport: str | None,
    host: str | None,
    port: int | None,
    uds: str | None,
    workers: int,
    stateless_http: bool,
    extra_args: tuple[str, ...],
) -> None:
    """Run the tai MCP server (FastMCP + Starlette), serving the Studio SPA too.

    \b
    Worker / transport combinations:
      | transport              | workers=1                | workers>1                          |
      |------------------------|--------------------------|------------------------------------|
      | stdio                  | ok (no --stateless-http) | refused                            |
      | sse                    | ok (no --stateless-http) | refused                            |
      | http / streamable-http | ok (stateful default)    | refused unless --stateless-http    |

    \b
    Worker-bus boot rules (set TAI_BUS_REDIS_URL to enable the bus):
      - more than one worker is refused without the bus — sibling workers would
        serve stale config after a reload with no channel to converge on;
      - TAI_CONFIG_MODE=k8s is refused without the bus — a pod cannot see its own
        replica count;
      - a manifest that registers a task backend is refused without the bus — the
        backend-runtime and server processes must converge on reloads.
    The workers rule lives in this CLI only: an external process manager driving
    the ASGI factory with its own --workers bypasses it, so set TAI_BUS_REDIS_URL
    in any multi-process deployment.
    """
    # Resolve the launch arguments from settings here (not at option-decoration
    # time, which runs at import): ``main`` has already bootstrapped the env, so
    # a local ``.env`` is in effect when these settings are read.
    defaults = app_args_settings()
    manifest_path = manifest_path if manifest_path is not None else default_manifest_path()
    if manifest_path is None:
        raise click.BadParameter("A manifest path is required.", param_hint="'--manifest-path'")
    transport = transport if transport is not None else defaults.transport
    host = host if host is not None else defaults.host
    port = port if port is not None else defaults.port
    uds = uds if uds is not None else defaults.uds

    uvicorn_kwargs = parse_and_validate_uvicorn_args(extra_args)

    try:
        run_mcp_app(
            manifest_path=manifest_path,
            transport=transport.lower(),
            host=host,
            port=port,
            workers=workers,
            uds=uds,
            stateless_http=stateless_http,
            uvicorn_kwargs=uvicorn_kwargs,
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
        sys.exit(130)
    except (TaiValidationError, FileNotFoundError, RuntimeError, TimeoutError, OSError, ImportError) as e:
        logger.error(str(e))
        raise click.ClickException(str(e)) from e


def main() -> None:
    if config_mode() != "k8s":
        load_dotenv()

    # Configure the root logger at process start, right after the env bootstrap, so
    # ``TAI_LOG_LEVEL`` from a local ``.env`` takes effect. Covers the master process,
    # the ``stdio`` path (``run_stdio``), and the debug path (``run_debug``);
    # ``basicConfig``'s default handler writes to stderr, never the stdout protocol
    # stream. In k8s config mode ``load_dotenv`` is skipped but this still runs — the
    # settings read the environment directly.
    setup_logging(logging_settings())

    cli()


if __name__ == "__main__":
    main()
