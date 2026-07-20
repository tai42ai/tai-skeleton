"""Public ASGI application factory for a user-owned tai server process.

``create_app`` is the programmatic entry point for embedding the tai MCP
application inside a host-owned ASGI process — a plain ``uvicorn main:app`` run or
a mount inside an existing FastAPI/Starlette host:

```python
from tai_skeleton.asgi import create_app

app = create_app(manifest_path="manifest.yml")
```

The returned Starlette app carries the full worker lifespan (manifest load,
``app_context``, transport selection, and the inner FastMCP lifespan), so the host
serves it exactly as the runtime CLI serves its own workers.

One app per process. The ``tai_app`` contract handle and the built app are
process-global singletons, so a second ``create_app`` lifespan entered while one
is already active raises loudly rather than silently rebinding the handle;
sequential lifespans in one process (enter -> exit -> enter) stay legal.

Deliberately CLI-owned and absent here: root-logger configuration (an embedded
app never touches the host's logging), the Prometheus multiprocess metrics
environment (an embedded process serves the in-process registry), and
process-fleet orchestration. Fleet config-reload fan-out is the app's internal
worker bus: an embedded worker joins it like any other process when
``TAI_BUS_REDIS_URL`` is set (the rules cannot count sibling processes in an
embedded host, so set it in any multi-process embed).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, get_args

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount

from tai_skeleton.app import instance
from tai_skeleton.config.config_mode import config_mode
from tai_skeleton.manifest import Manifest

__all__ = ["Transport", "create_app", "lifespan"]

logger = logging.getLogger(__name__)

Transport = Literal["http", "streamable-http", "sse"]

_VALID_TRANSPORTS = get_args(Transport)

# Marker naming the active app's manifest in the one-app guard message when that
# app was created without an explicit ``manifest_path`` (its manifest resolves
# from the environment / config dir at lifespan start).
_ENV_RESOLVED_MARKER = "<env-resolved manifest>"

# One-app-per-process guard. ``_guard_lock`` makes the claim and the release
# atomic so two lifespans entering from different threads cannot both pass the
# check and rebind the process-global handle. ``_app_active`` is the token;
# ``_active_manifest_marker`` names the holder's manifest for the raise message.
_guard_lock = threading.Lock()
_app_active = False
_active_manifest_marker = ""


def create_app(
    manifest_path: str | None = None,
    *,
    transport: Transport = "http",
    stateless_http: bool = False,
) -> Starlette:
    """Build the public ASGI app for a user-owned process.

    One app per process: the returned app's lifespan claims a process-global
    one-app token, and a second lifespan entered while one is active raises. The
    ``transport`` and ``stateless_http`` arguments are validated at call time.

    Args:
        manifest_path: The manifest this app loads. Omit it to let the existing
            environment / config-dir resolution apply.
        transport: The MCP transport to serve.
        stateless_http: Run an http/streamable-http transport in fastmcp's
            stateless mode. Has no ``sse`` equivalent, so pairing it with the
            ``sse`` transport is rejected at call time.

    Returns:
        A Starlette app whose lifespan claims the one-app token, stamps the
        manifest env, builds the app singleton, enters ``app_context`` and the
        inner FastMCP lifespan, and releases the token (restoring the env) on exit.
    """
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"transport must be one of {', '.join(repr(t) for t in _VALID_TRANSPORTS)}; got {transport!r}."
        )
    if stateless_http and transport == "sse":
        raise ValueError(
            "stateless_http requires an http transport ('http' or 'streamable-http'); "
            "the 'sse' transport has no stateless mode."
        )

    app_state: dict = {}

    @asynccontextmanager
    async def worker_lifespan(_app):
        global _app_active, _active_manifest_marker

        saved_manifest_env: str | None = None

        # Claim the one-app token and stamp the manifest env under the same lock,
        # BEFORE the try whose finally releases the token — a failed claim must
        # raise without touching the active holder's token or environment.
        with _guard_lock:
            if _app_active:
                raise RuntimeError(
                    "a tai app lifespan is already active in this process; the tai_app handle "
                    "and the built app are process-global singletons (one app per process). "
                    f"The active app's manifest is {_active_manifest_marker}."
                )
            _app_active = True
            _active_manifest_marker = manifest_path if manifest_path is not None else _ENV_RESOLVED_MARKER
            if manifest_path is not None:
                saved_manifest_env = os.environ.get("TAI_MANIFEST_PATH")
                try:
                    os.environ["TAI_MANIFEST_PATH"] = manifest_path
                except Exception:
                    # The stamp raised after the token was claimed (e.g. a manifest
                    # path with an embedded NUL byte). Roll the claim back under the
                    # same lock so a stamp failure never wedges the one-app guard,
                    # then re-raise loudly. The failed assignment left the env
                    # untouched, so there is nothing to restore.
                    _app_active = False
                    _active_manifest_marker = ""
                    raise

        try:
            app = instance.build_app()
            logger.info("Configuration mode: %s", config_mode())
            manifest = Manifest.model_validate(app.config.config_manager.read_manifest())

            # Initialize the core app context
            async with app.app_context(manifest):
                # Select the appropriate transport mode
                if transport == "sse":
                    logger.info("Initializing Legacy SSE App")
                    inner_app = app.sse_app()
                elif stateless_http:
                    logger.info("Initializing stateless Streamable HTTP App")
                    inner_app = app.http_app(stateless_http=True)
                else:
                    logger.info("Initializing Streamable HTTP App")
                    inner_app = app.http_app()

                app_state["app"] = inner_app

                # FastMCP requires its own lifespan to run to initialize
                # TaskGroups. The mounted dispatch below swallows the lifespan
                # scope, so we enter it by hand. ``finalize`` records the
                # lifespan-bearing FastMCP app as ``mcp_lifespan_app`` so the
                # lifespan is entered even when middleware wraps ``inner_app``
                # (a middleware wrapper exposes no router/lifespan of its own).
                lifespan_app = getattr(inner_app, "mcp_lifespan_app", inner_app)
                lifespan = getattr(lifespan_app, "lifespan", None)
                if lifespan is not None:
                    async with lifespan(lifespan_app):
                        yield
                else:
                    yield

        except Exception:
            logger.exception("Worker application lifespan failed")
            raise
        finally:
            # Release the token and restore the manifest env under the same lock
            # so a failed boot never wedges the process and a later no-param app
            # resolves the config-dir default rather than this app's path.
            with _guard_lock:
                if manifest_path is not None:
                    if saved_manifest_env is not None:
                        os.environ["TAI_MANIFEST_PATH"] = saved_manifest_env
                    else:
                        os.environ.pop("TAI_MANIFEST_PATH", None)
                _app_active = False
                _active_manifest_marker = ""

    async def dispatch(scope, receive, send):
        """Forward requests to the inner app."""
        # The outer Starlette app owns the lifespan (worker_lifespan). If a
        # server delivers a lifespan scope to this mounted sub-app anyway,
        # swallow it here: the request paths below speak HTTP and would emit an
        # invalid response against a lifespan scope.
        if scope["type"] == "lifespan":
            return

        if "app" in app_state:
            try:
                await app_state["app"](scope, receive, send)
            except Exception:
                logger.exception("Error processing request in mcp app")
                try:
                    # Fixed generic body: internal exception text (hosts,
                    # paths) must never reach the client.
                    response = JSONResponse({"error": "Internal Server Error"}, status_code=500)
                    await response(scope, receive, send)
                except RuntimeError:
                    # The response already started before the failure — nothing
                    # more can be sent on this connection. The original error is
                    # logged above; record the double-fault too.
                    logger.warning(
                        "could not send the 500 response; response already started",
                        exc_info=True,
                    )
        else:
            response = JSONResponse({"error": "Service Unavailable", "detail": "Initializing..."}, status_code=503)
            await response(scope, receive, send)

    return Starlette(lifespan=worker_lifespan, routes=[Mount("/", app=dispatch)])


def lifespan(app: Starlette):
    """Return a ``create_app`` app's lifespan context manager, for composing into a
    host lifespan when the app is mounted.

    Mounting the factory app is not enough on its own: Starlette does not run a
    mounted sub-app's lifespan, so the host must run it. Enter the returned context
    manager inside the host's own lifespan so the tai app's worker startup (manifest
    load, ``app_context``, transport selection, the inner FastMCP lifespan) runs for
    the host process's lifetime.

    Args:
        app: A ``create_app`` result whose lifespan the host runs.

    Returns:
        The app's lifespan context manager (``app.router.lifespan_context(app)``),
        to be entered inside the host lifespan.
    """
    return app.router.lifespan_context(app)
