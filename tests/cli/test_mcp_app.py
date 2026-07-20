"""Main MCP app launcher.

These tests cover the launcher wiring up to (but not through) the blocking
server start. ``uvicorn.run`` / ``uvicorn.Server.serve`` / ``asyncio.run`` are
mocked so the arg/transport validation, env publishing, config-mode logging,
app construction, the ``app_context`` worker lifespan, and the request-dispatch
forwarder all execute without a live server.

The ``app`` and ``Manifest`` seams are replaced with fakes: the unit under test
is the CLI wiring, not the real MCP app or manifest validation.
"""

from __future__ import annotations

import logging
import os
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from starlette.routing import Mount
from starlette.testclient import TestClient

import tai42_skeleton.cli.mcp_app as mcp_app
from tai42_skeleton.connectors import meta_log_redactor

_LOGGING_RELOAD_KEY = "tai42_skeleton.app.instance.apply_logging_settings"


@pytest.fixture(autouse=True)
def _restore_log_record_factory():
    """Save/restore the process-global record factory and its monotonic redaction
    scope around every test: the CLI seams install the connector-secret redactor at
    process scope, so this keeps one test's install from leaking onward."""
    saved_factory = logging.getLogRecordFactory()
    saved_scope = meta_log_redactor._SCOPE
    try:
        yield
    finally:
        logging.setLogRecordFactory(saved_factory)
        meta_log_redactor._SCOPE = saved_scope


@pytest.fixture(autouse=True)
def _bus_configured(monkeypatch: pytest.MonkeyPatch):
    """These serve-path tests exercise multi-worker runs, which the boot rules now
    require the worker bus for. Report it configured so the launch wiring under test
    runs (nothing here opens a real bus — uvicorn.run and the in-process servers are
    faked); the busless refusals have their own suite."""
    from tai42_kit.settings import reset_all_settings

    monkeypatch.setenv("TAI_BUS_REDIS_URL", "redis://localhost:6379/0")
    reset_all_settings()
    try:
        yield
    finally:
        reset_all_settings()


_HTTP_SCOPE = {
    "type": "http",
    "http_version": "1.1",
    "method": "GET",
    "path": "/",
    "raw_path": b"/",
    "query_string": b"",
    "headers": [],
    "scheme": "http",
    "server": ("testserver", 80),
    "client": ("testclient", 1234),
}


# --- fakes for the app / manifest seams -----------------------------------


class _FakeConfigManager:
    def __init__(self, raise_on_read: bool = False) -> None:
        self.raise_on_read = raise_on_read

    def read_manifest(self) -> dict:
        if self.raise_on_read:
            raise RuntimeError("manifest read failed")
        return {}


class _FakeInnerApp:
    """ASGI stand-in for ``app.http_app()`` / ``app.sse_app()``."""

    def __init__(self, *, raise_on_call: bool = False, with_lifespan: bool = True) -> None:
        self.raise_on_call = raise_on_call
        self.lifespan_entered = False
        self.calls = 0
        if with_lifespan:
            # Mirrors ``StarletteWithLifespan.lifespan`` (a callable taking the app
            # and returning the lifespan context manager), which the worker enters.
            self.lifespan = self._lifespan_context

    @asynccontextmanager
    async def _lifespan_context(self, _app):
        self.lifespan_entered = True
        yield

    async def __call__(self, scope, receive, send) -> None:
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("inner boom")
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


class _FakeLifecycle:
    """Records reload-handler registrations by their ``module.qualname`` key, so a
    test can assert ``register_cli_logging_reload`` wired ``apply_logging_settings``."""

    def __init__(self) -> None:
        self.reload_handlers: dict[str, object] = {}

    def on_reload(self, func):
        self.reload_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func


class _FakeApp:
    def __init__(self, inner: _FakeInnerApp, *, raise_on_read: bool = False) -> None:
        self.inner = inner
        # The config manager is reached through the ``config`` facet namespace.
        self.config = SimpleNamespace(config_manager=_FakeConfigManager(raise_on_read=raise_on_read))
        # The CLI seams register the root-logger reload handler through the app's
        # lifecycle; the recorder captures it without a real lifecycle.
        self.lifecycle = _FakeLifecycle()
        self.http_called = False
        self.http_stateless: bool | None = None
        self.sse_called = False
        self.run_async_transport: str | None = None

    @asynccontextmanager
    async def app_context(self, manifest):
        yield

    def http_app(self, stateless_http: bool | None = None) -> _FakeInnerApp:
        self.http_called = True
        self.http_stateless = stateless_http
        return self.inner

    def sse_app(self) -> _FakeInnerApp:
        self.sse_called = True
        return self.inner

    async def run_async(self, transport: str) -> None:
        self.run_async_transport = transport


@pytest.fixture
def patch_app_seam(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``app`` + bypass real ``Manifest`` validation."""

    def _install(app: _FakeApp) -> _FakeApp:
        # The launcher obtains the app via the deferred factory ``instance.build_app``.
        monkeypatch.setattr(mcp_app.instance, "build_app", lambda: app)
        monkeypatch.setattr(mcp_app.Manifest, "model_validate", staticmethod(lambda data: SimpleNamespace()))
        return app

    return _install


# --- create_app: worker lifespan + dispatch -------------------------------


def test_create_app_http_lifespan_and_dispatch(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    inner = _FakeInnerApp()
    app = patch_app_seam(_FakeApp(inner))

    star = mcp_app.create_app()
    with TestClient(star) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == "ok"
    assert app.http_called is True
    assert app.sse_called is False
    assert inner.lifespan_entered is True


def test_create_app_sse_transport_selects_sse(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "sse")
    inner = _FakeInnerApp()
    app = patch_app_seam(_FakeApp(inner))

    star = mcp_app.create_app()
    with TestClient(star) as client:
        client.get("/")

    assert app.sse_called is True
    assert app.http_called is False


def test_create_app_inner_without_lifespan_context(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    inner = _FakeInnerApp(with_lifespan=False)
    patch_app_seam(_FakeApp(inner))

    star = mcp_app.create_app()
    with TestClient(star) as client:
        response = client.get("/")

    assert response.status_code == 200


def test_create_app_enters_lifespan_via_mcp_lifespan_app_when_wrapped(
    patch_app_seam, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``@app.http.middleware`` wraps the served app, ``http_app()`` returns
    a plain-ASGI wrapper with no lifespan of its own. ``finalize`` records the
    lifespan-bearing FastMCP app as ``mcp_lifespan_app``; the worker must enter
    THAT lifespan, then still handle the request through the wrapper — otherwise
    the streamable-http session-manager task group never starts."""
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    inner = _FakeInnerApp()  # the lifespan-bearing FastMCP app

    class _PassThroughMiddleware:
        """Records that it ran, exposes no lifespan — like a finalized wrapper."""

        def __init__(self, app) -> None:
            self._app = app
            self.mcp_lifespan_app = app
            self.saw_request = False

        async def __call__(self, scope, receive, send) -> None:
            self.saw_request = True
            await self._app(scope, receive, send)

    wrapped = _PassThroughMiddleware(inner)
    app = _FakeApp(inner)
    monkeypatch.setattr(app, "http_app", lambda: wrapped)
    patch_app_seam(app)

    star = mcp_app.create_app()
    with TestClient(star) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == "ok"
    # The FastMCP lifespan was entered via ``mcp_lifespan_app`` despite the wrapper.
    assert inner.lifespan_entered is True
    # ...and the request still flowed through the middleware wrapper.
    assert wrapped.saw_request is True


def test_dispatch_forwards_inner_error_as_500(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    inner = _FakeInnerApp(raise_on_call=True)
    patch_app_seam(_FakeApp(inner))

    star = mcp_app.create_app()
    with TestClient(star, raise_server_exceptions=False) as client:
        response = client.get("/")

    assert response.status_code == 500
    assert response.json()["error"] == "Internal Server Error"


def test_worker_lifespan_reraises_init_failure(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    inner = _FakeInnerApp()
    patch_app_seam(_FakeApp(inner, raise_on_read=True))

    star = mcp_app.create_app()
    with pytest.raises(RuntimeError, match="manifest read failed"), TestClient(star):
        pass


async def test_dispatch_swallows_double_fault_when_response_started(
    patch_app_seam, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The inner app starts a response, then raises. The dispatch error handler
    # tries to send its own 500, but the ASGI stream already began → the second
    # ``http.response.start`` raises RuntimeError, which the handler swallows.
    monkeypatch.setenv("TAI_TRANSPORT", "http")

    class _StartThenRaiseInner(_FakeInnerApp):
        async def __call__(self, scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("inner boom after start")

    patch_app_seam(_FakeApp(_StartThenRaiseInner()))
    star = mcp_app.create_app()
    # Running the worker lifespan once populates ``app_state["app"]`` (the dict
    # in the dispatch closure persists after shutdown), so the dispatch call
    # below forwards to the inner app.
    with TestClient(star, raise_server_exceptions=False):
        pass
    mount = star.routes[0]
    assert isinstance(mount, Mount)  # create_app builds a single Mount("/", app=dispatch)
    dispatch = mount.app

    started = {"flag": False}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            if started["flag"]:
                raise RuntimeError("Response already started")
            started["flag"] = True

    # The error handler's send of the 500 hits the already-started stream and is
    # swallowed; the call returns without propagating.
    await dispatch(dict(_HTTP_SCOPE), receive, send)
    assert started["flag"] is True


async def test_dispatch_service_unavailable_before_init(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    # Reach the dispatch closure directly, without running the lifespan, so the
    # inner app is not yet wired → 503. Also drives the lifespan-scope early
    # return, which Starlette never forwards to a mounted sub-app.
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    patch_app_seam(_FakeApp(_FakeInnerApp()))
    mount = mcp_app.create_app().routes[0]
    assert isinstance(mount, Mount)  # create_app builds a single Mount("/", app=dispatch)
    dispatch = mount.app

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    lifespan_sent: list = []

    async def send_lifespan(message):
        lifespan_sent.append(message)

    await dispatch({"type": "lifespan"}, receive, send_lifespan)
    assert lifespan_sent == []  # early return, nothing emitted

    http_sent: list = []

    async def send_http(message):
        http_sent.append(message)

    await dispatch(dict(_HTTP_SCOPE), receive, send_http)
    start = next(m for m in http_sent if m["type"] == "http.response.start")
    assert start["status"] == 503


# --- run_stdio / run_debug ------------------------------------------------


async def test_run_stdio_enters_context_and_runs(patch_app_seam) -> None:
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))

    rc = await mcp_app.run_stdio()

    assert rc == 0
    assert app.run_async_transport == "stdio"


class _FakeUvicorn:
    def __init__(self) -> None:
        self.config_kwargs: dict | None = None
        self.served = False

    def Config(self, **kwargs):
        self.config_kwargs = kwargs
        return ("config", kwargs)

    def Server(self, config):
        outer = self

        class _Server:
            async def serve(self) -> None:
                outer.served = True

        return _Server()


async def test_run_debug_http_builds_http_app(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))
    fake_uvicorn = _FakeUvicorn()
    monkeypatch.setattr(mcp_app, "uvicorn", fake_uvicorn)

    config_kwargs: dict = {"host": "127.0.0.1", "port": 8000}
    rc = await mcp_app.run_debug("http", config_kwargs)

    assert rc == 0
    assert app.http_called is True
    assert config_kwargs["app"] is app.inner
    assert fake_uvicorn.served is True


async def test_run_debug_sse_builds_sse_app(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))
    fake_uvicorn = _FakeUvicorn()
    monkeypatch.setattr(mcp_app, "uvicorn", fake_uvicorn)

    config_kwargs: dict = {"host": "127.0.0.1", "port": 8000}
    await mcp_app.run_debug("sse", config_kwargs)

    assert app.sse_called is True
    assert config_kwargs["app"] is app.inner


# --- CLI-seam logging-reload registration ---------------------------------


def test_wrapper_registers_cli_logging_reload(
    patch_app_seam, root_logger_restored, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    monkeypatch.delenv("TAI_STATELESS_HTTP", raising=False)
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))

    mcp_app.create_app()

    assert _LOGGING_RELOAD_KEY in app.lifecycle.reload_handlers


async def test_run_stdio_registers_cli_logging_reload(patch_app_seam) -> None:
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))

    await mcp_app.run_stdio()

    assert _LOGGING_RELOAD_KEY in app.lifecycle.reload_handlers


async def test_run_debug_registers_cli_logging_reload(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))
    monkeypatch.setattr(mcp_app, "uvicorn", _FakeUvicorn())

    await mcp_app.run_debug("http", {"host": "127.0.0.1", "port": 8000})

    assert _LOGGING_RELOAD_KEY in app.lifecycle.reload_handlers


def test_wrapper_installs_process_scope_redactor(
    patch_app_seam, root_logger_restored, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This CLI-owned worker widens the connector-secret redactor to the whole process.
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    monkeypatch.delenv("TAI_STATELESS_HTTP", raising=False)
    patch_app_seam(_FakeApp(_FakeInnerApp()))
    scopes: list[object] = []
    monkeypatch.setattr(mcp_app, "install_meta_log_redactor", lambda **kwargs: scopes.append(kwargs.get("scope")))

    mcp_app.create_app()

    assert scopes == ["process"]


async def test_run_stdio_installs_process_scope_redactor(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_app_seam(_FakeApp(_FakeInnerApp()))
    scopes: list[object] = []
    monkeypatch.setattr(mcp_app, "install_meta_log_redactor", lambda **kwargs: scopes.append(kwargs.get("scope")))

    await mcp_app.run_stdio()

    assert scopes == ["process"]


async def test_run_debug_installs_process_scope_redactor(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_app_seam(_FakeApp(_FakeInnerApp()))
    monkeypatch.setattr(mcp_app, "uvicorn", _FakeUvicorn())
    scopes: list[object] = []
    monkeypatch.setattr(mcp_app, "install_meta_log_redactor", lambda **kwargs: scopes.append(kwargs.get("scope")))

    await mcp_app.run_debug("http", {"host": "127.0.0.1", "port": 8000})

    assert scopes == ["process"]


# --- run_mcp_app: validation branches -------------------------------------


def _defaults() -> tuple[str, int]:
    settings = mcp_app.app_args_settings()
    return settings.host, settings.port


def test_uds_on_windows_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "win32")
    host, port = _defaults()
    with pytest.raises(click.BadParameter, match="Unix Domain Sockets"):
        mcp_app.run_mcp_app("m.yaml", "sse", host, port, workers=1, uds="/tmp/s.sock")


def test_uds_with_stdio_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    host, port = _defaults()
    with pytest.raises(click.BadParameter, match="cannot be used with '--transport stdio'"):
        mcp_app.run_mcp_app("m.yaml", "stdio", host, port, workers=1, uds="/tmp/s.sock")


def test_stdio_with_nondefault_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    _, port = _defaults()
    with pytest.raises(click.BadParameter, match="should not be set"):
        mcp_app.run_mcp_app("m.yaml", "stdio", "0.0.0.0", port, workers=1)


def test_stdio_with_multiple_workers_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    host, port = _defaults()
    with pytest.raises(click.BadParameter, match="Multiple workers"):
        mcp_app.run_mcp_app("m.yaml", "stdio", host, port, workers=2)


# --- serve hardening: stateful-transport worker guard ---------------------


@pytest.mark.parametrize("transport", ["http", "streamable-http", "sse"])
def test_stateful_transport_multiple_workers_rejected(transport: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stateful HTTP/SSE transport refuses workers>1, naming the fix."""
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    host, port = _defaults()
    with pytest.raises(click.BadParameter) as excinfo:
        mcp_app.run_mcp_app("m.yaml", transport, host, port, workers=2)
    message = str(excinfo.value)
    assert "run one worker" in message
    if transport in {"http", "streamable-http"}:
        assert "--stateless-http" in message
    else:
        assert "no stateless mode" in message


@pytest.mark.parametrize("transport", ["http", "streamable-http"])
def test_stateless_http_lifts_multi_worker_refusal(transport: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("m.yaml", transport, host, port, workers=4, stateless_http=True)

    assert rc == 0
    assert recorded[0][1]["workers"] == 4
    assert mcp_app.os.environ["TAI_STATELESS_HTTP"] == "1"


@pytest.mark.parametrize("transport", ["sse", "stdio"])
def test_stateless_http_with_non_http_transport_rejected(transport: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    host, port = _defaults()
    # ``stdio``/``uds`` also refuse a non-default host, so keep them at defaults.
    args = (host, port) if transport == "sse" else _defaults()
    with pytest.raises(click.BadParameter, match="requires an http transport"):
        mcp_app.run_mcp_app("m.yaml", transport, *args, workers=1, stateless_http=True)


def test_stateless_http_clears_env_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run WITHOUT --stateless-http clears any stale env flag from a prior run."""
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.setenv("TAI_STATELESS_HTTP", "1")
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: None)
    host, port = _defaults()

    mcp_app.run_mcp_app("m.yaml", "http", host, port, workers=1)

    assert "TAI_STATELESS_HTTP" not in mcp_app.os.environ


def test_create_app_stateless_reaches_http_app_factory(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env flag set, the worker factory builds ``http_app(stateless_http=True)``."""
    monkeypatch.setenv("TAI_TRANSPORT", "http")
    monkeypatch.setenv("TAI_STATELESS_HTTP", "1")
    inner = _FakeInnerApp()
    app = patch_app_seam(_FakeApp(inner))

    star = mcp_app.create_app()
    with TestClient(star) as client:
        client.get("/")

    assert app.http_called is True
    assert app.http_stateless is True


# --- serve hardening: stale UDS socket cleanup ----------------------------


def _stale_uds_socket(path: str) -> None:
    """Create a socket-typed path with no listener — a connect refuses it."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.close()  # the filesystem entry persists; nothing is listening on it


def test_prepare_uds_unlinks_stale_socket(tmp_path) -> None:
    path = str(tmp_path / "stale.sock")
    _stale_uds_socket(path)
    assert os.path.exists(path)

    mcp_app._prepare_uds_path(path)

    assert not os.path.exists(path)  # stale socket removed, ready to rebind


def test_prepare_uds_refuses_live_socket(tmp_path) -> None:
    path = str(tmp_path / "live.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    try:
        with pytest.raises(click.BadParameter, match="already running"):
            mcp_app._prepare_uds_path(path)
        assert os.path.exists(path)  # a live server's socket is never unlinked
    finally:
        server.close()
        os.unlink(path)


def test_prepare_uds_refuses_non_socket_without_unlink(tmp_path) -> None:
    path = tmp_path / "regular.file"
    path.write_text("not a socket")
    with pytest.raises(click.BadParameter, match="not a socket"):
        mcp_app._prepare_uds_path(str(path))
    assert path.exists()  # a non-socket path is refused and left in place


def test_prepare_uds_missing_path_is_noop(tmp_path) -> None:
    # A path that does not exist binds fresh — no error, nothing created.
    mcp_app._prepare_uds_path(str(tmp_path / "absent.sock"))


def test_run_mcp_app_uds_cleans_stale_before_bind(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    path = str(tmp_path / "run.sock")
    _stale_uds_socket(path)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("m.yaml", "http", host, port, workers=1, uds=path)

    assert rc == 0
    assert recorded[0][1]["uds"] == path
    assert not os.path.exists(path)  # unlinked before the (faked) bind


# --- run_mcp_app: launch paths --------------------------------------------


def test_run_mcp_app_stdio_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    calls: list = []

    def fake_asyncio_run(coro):
        coro.close()
        calls.append(coro)
        return 0

    monkeypatch.setattr(mcp_app.asyncio, "run", fake_asyncio_run)
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "stdio", host, port, workers=1, uvicorn_kwargs=None)

    assert rc == 0
    assert len(calls) == 1
    assert os.environ["TAI_MANIFEST_PATH"] == "manifest.yaml"
    assert os.environ["TAI_TRANSPORT"] == "stdio"


def test_run_mcp_app_activates_multiproc_env_before_wipe(monkeypatch: pytest.MonkeyPatch) -> None:
    # The multiproc dir env must be published BEFORE the wipe is imported/called —
    # importing the wipe's module is the first thing that pulls in prometheus_client
    # (which freezes its value backend from the env). This pins that ordering.
    import tai42_skeleton.routers.metrics_settings as ms
    import tai42_skeleton.routers.prometheus as prom

    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.setattr(mcp_app.asyncio, "run", lambda coro: coro.close() or 0)

    settings_dir = ms.metrics_settings().prometheus_multiproc_dir
    order: list = []

    def fake_activate() -> str:
        order.append("activate")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings_dir
        return settings_dir

    def fake_wipe() -> str:
        # Record the env visible at wipe time — it must already be the settings dir.
        order.append(("wipe", os.environ.get("PROMETHEUS_MULTIPROC_DIR")))
        return settings_dir

    monkeypatch.setattr(ms, "activate_multiproc_env", fake_activate)
    monkeypatch.setattr(prom, "wipe_prometheus_multiproc_dir", fake_wipe)
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "stdio", host, port, workers=1)

    assert rc == 0
    assert os.path.isabs(settings_dir)
    assert order == ["activate", ("wipe", settings_dir)]


def test_run_mcp_app_tcp_path_calls_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    # >1 worker on an http transport is only allowed under --stateless-http.
    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=3, stateless_http=True)

    assert rc == 0
    target, kwargs = recorded[0]
    assert target == "tai42_skeleton.cli.mcp_app:create_app"
    assert kwargs["factory"] is True
    assert kwargs["workers"] == 3
    assert kwargs["host"] == host
    assert kwargs["port"] == port
    # The flag travels to the factory worker by env.
    assert mcp_app.os.environ["TAI_STATELESS_HTTP"] == "1"


def test_run_mcp_app_uds_path_binds_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "sse", host, port, workers=1, uds="/tmp/s.sock")

    assert rc == 0
    _, kwargs = recorded[0]
    assert kwargs["uds"] == "/tmp/s.sock"
    assert "host" not in kwargs
    assert "port" not in kwargs


def test_run_mcp_app_debug_environment_runs_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.setenv("TAI_RUN_MODE", "debug")
    calls: list = []

    def fake_asyncio_run(coro):
        coro.close()
        calls.append(coro)
        return 0

    monkeypatch.setattr(mcp_app.asyncio, "run", fake_asyncio_run)
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)

    assert rc == 0
    assert len(calls) == 1


def test_run_mcp_app_debug_run_mode_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``debug`` matches regardless of case.
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.setenv("TAI_RUN_MODE", "DEBUG")
    calls: list = []

    def fake_asyncio_run(coro):
        coro.close()
        calls.append(coro)
        return 0

    monkeypatch.setattr(mcp_app.asyncio, "run", fake_asyncio_run)
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)

    assert rc == 0
    assert len(calls) == 1


def test_run_mcp_app_unknown_run_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any other non-empty value fails loudly rather than silently falling through
    # to the normal multi-worker path.
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.setenv("TAI_RUN_MODE", "production")
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda *a, **kw: pytest.fail("uvicorn.run must not be reached"))
    monkeypatch.setattr(mcp_app.asyncio, "run", lambda *a, **kw: pytest.fail("asyncio.run must not be reached"))
    host, port = _defaults()

    with pytest.raises(click.ClickException) as excinfo:
        mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)

    message = str(excinfo.value)
    assert "TAI_RUN_MODE" in message
    assert "production" in message
    assert "debug" in message


def test_run_mcp_app_normal_path_logs_worker_count(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: None)
    # run_mcp_app configures logging (basicConfig force=True, which would drop the
    # caplog handler); no-op it here so this test can capture the run-mode message.
    monkeypatch.setattr(mcp_app, "setup_logging", lambda *a, **k: None)
    host, port = _defaults()

    with caplog.at_level("INFO", logger=mcp_app.logger.name):
        # stateless-http lifts the single-worker restriction for the http transport.
        rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=3, stateless_http=True)

    assert rc == 0
    assert any("worker" in rec.getMessage() and "3" in rec.getMessage() for rec in caplog.records)


def test_run_mcp_app_configures_logging_on_serve_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shipped ``tai serve`` reaches ``run_mcp_app`` via ``cli`` (not ``main``),
    # so ``run_mcp_app`` must configure logging itself at its top — otherwise the
    # master/stdio/debug servers it dispatches to stay unconfigured. This is the
    # positive guard: ``setup_logging`` IS invoked with the resolved
    # ``logging_settings()`` before the (faked) uvicorn launch.
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: None)
    recorded: list = []
    monkeypatch.setattr(mcp_app, "setup_logging", lambda cfg: recorded.append(cfg))
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=3, stateless_http=True)

    assert rc == 0
    assert recorded == [mcp_app.logging_settings()]


def test_run_mcp_app_merges_uvicorn_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1, uvicorn_kwargs={"timeout_keep_alive": 5})

    _, kwargs = recorded[0]
    assert kwargs["timeout_keep_alive"] == 5
    assert kwargs["ws"] == "wsproto"


# --- graceful-shutdown timeout --------------------------------------------


def test_run_mcp_app_sets_graceful_shutdown_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The normal (uvicorn.run) path carries the settings-backed default.
    from tai42_kit.settings import reset_all_settings

    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.delenv("APP_ARGS_TIMEOUT_GRACEFUL_SHUTDOWN", raising=False)
    reset_all_settings()
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)

    assert rc == 0
    assert recorded[0][1]["timeout_graceful_shutdown"] == 10
    reset_all_settings()


def test_run_mcp_app_debug_path_carries_graceful_shutdown(patch_app_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    # The debug path builds a uvicorn.Config from the same config_kwargs, so the
    # setting reaches it too.
    app = patch_app_seam(_FakeApp(_FakeInnerApp()))
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.setenv("TAI_RUN_MODE", "debug")
    fake_uvicorn = _FakeUvicorn()
    monkeypatch.setattr(mcp_app, "uvicorn", fake_uvicorn)
    host, port = _defaults()

    rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)

    assert rc == 0
    assert app.http_called is True
    assert fake_uvicorn.config_kwargs is not None
    assert (
        fake_uvicorn.config_kwargs["timeout_graceful_shutdown"] == mcp_app.app_args_settings().timeout_graceful_shutdown
    )


def test_cli_graceful_shutdown_extra_arg_overrides_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # A shipped ``--timeout-graceful-shutdown`` CLI extra-arg wins over the default.
    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1, uvicorn_kwargs={"timeout_graceful_shutdown": 3})

    assert recorded[0][1]["timeout_graceful_shutdown"] == 3


def test_run_mcp_app_graceful_shutdown_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai42_kit.settings import reset_all_settings

    monkeypatch.setattr(mcp_app.sys, "platform", "linux")
    monkeypatch.delenv("TAI_RUN_MODE", raising=False)
    monkeypatch.setenv("APP_ARGS_TIMEOUT_GRACEFUL_SHUTDOWN", "7")
    reset_all_settings()
    recorded: list = []
    monkeypatch.setattr(mcp_app.uvicorn, "run", lambda target, **kw: recorded.append((target, kw)))
    host, port = _defaults()

    try:
        rc = mcp_app.run_mcp_app("manifest.yaml", "http", host, port, workers=1)
        assert rc == 0
        assert recorded[0][1]["timeout_graceful_shutdown"] == 7
    finally:
        reset_all_settings()


# --- cli (Click entry) ----------------------------------------------------


def test_cli_forwards_parsed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(mcp_app, "run_mcp_app", lambda **kw: captured.update(kw) or 0)

    result = CliRunner().invoke(
        mcp_app.cli,
        [
            "--manifest-path",
            "m.yaml",
            "--transport",
            "HTTP",
            "--host",
            "1.2.3.4",
            "--port",
            "9001",
            "--workers",
            "2",
            "--timeout-keep-alive",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["manifest_path"] == "m.yaml"
    assert captured["transport"] == "http"  # lowercased before forwarding
    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 9001
    assert captured["workers"] == 2
    assert captured["uvicorn_kwargs"]["timeout_keep_alive"] == 5


def test_cli_manifest_default_comes_from_tai_manifest_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """TAI_MANIFEST_PATH is the single manifest env var: with no --manifest-path
    flag, the CLI resolves its default from it (via CoreSettings) end-to-end."""
    from tai42_skeleton.settings import cache

    monkeypatch.setenv("TAI_MANIFEST_PATH", "/etc/tai/from-env.yaml")
    cache.manifest_path.cache_clear()
    captured: dict = {}
    monkeypatch.setattr(mcp_app, "run_mcp_app", lambda **kw: captured.update(kw) or 0)
    try:
        result = CliRunner().invoke(mcp_app.cli, [])
    finally:
        cache.manifest_path.cache_clear()

    assert result.exit_code == 0, result.output
    assert captured["manifest_path"] == "/etc/tai/from-env.yaml"


def test_cli_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mcp_app, "run_mcp_app", boom)

    result = CliRunner().invoke(mcp_app.cli, ["--manifest-path", "m.yaml"])

    assert result.exit_code == 130


def test_cli_known_error_becomes_click_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(mcp_app, "run_mcp_app", boom)

    result = CliRunner().invoke(mcp_app.cli, ["--manifest-path", "m.yaml"])

    assert result.exit_code != 0
    assert "launch failed" in result.output


def test_main_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list = []
    monkeypatch.setattr(mcp_app, "cli", lambda: called.append(True))
    monkeypatch.setattr(mcp_app, "config_mode", lambda: "k8s")

    mcp_app.main()

    assert called == [True]


def test_main_bootstraps_env_when_not_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_app, "cli", lambda: None)
    monkeypatch.setattr(mcp_app, "config_mode", lambda: "file")
    monkeypatch.setattr(mcp_app, "load_dotenv", lambda: called.append(True))

    mcp_app.main()

    assert called == [True]


def test_main_skips_env_bootstrap_in_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_app, "cli", lambda: None)
    monkeypatch.setattr(mcp_app, "config_mode", lambda: "k8s")
    monkeypatch.setattr(mcp_app, "load_dotenv", lambda: called.append(True))

    mcp_app.main()

    assert called == []
