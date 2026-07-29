"""Public ASGI factory (`tai42_skeleton.asgi.create_app`).

Covers argument validation, the manifest-env lifespan stamp (and its restore,
including a stamp failure rolling the one-app claim back), transport selection,
the one-app-per-process guard, host mounting with a composed lifespan, and that a
factory boot leaves the host's root logger untouched.

The ``app`` and ``Manifest`` seams are replaced with fakes: the unit under test is
the factory wiring, not the real MCP app or manifest validation. The factory's
``build_app`` / ``Manifest`` are the same module objects the CLI wrapper shares, so
patching ``asgi.instance.build_app`` / ``asgi.Manifest.model_validate`` reaches both.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient

import tai42_skeleton.asgi as asgi
import tai42_skeleton.cli.mcp_app as mcp_app

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
    def read_manifest(self) -> dict:
        return {}


class _FakeInnerApp:
    """ASGI stand-in for ``app.http_app()`` / ``app.sse_app()``."""

    def __init__(self) -> None:
        self.lifespan_entered = False
        self.lifespan = self._lifespan_context

    @asynccontextmanager
    async def _lifespan_context(self, _app):
        self.lifespan_entered = True
        yield

    async def __call__(self, scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


class _FakeLifecycle:
    """Records reload-handler registrations by their ``module.qualname`` key."""

    def __init__(self) -> None:
        self.reload_handlers: dict[str, object] = {}

    def on_reload(self, func):
        self.reload_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func


class _FakeApp:
    def __init__(self, *, inner: _FakeInnerApp | None = None, raise_in_context: bool = False) -> None:
        self.inner = inner or _FakeInnerApp()
        # The config manager is reached through the ``config`` facet namespace.
        self.config = SimpleNamespace(config_manager=_FakeConfigManager())
        self.lifecycle = _FakeLifecycle()
        self.raise_in_context = raise_in_context
        self.http_called = False
        self.http_stateless: bool | None = None
        self.sse_called = False
        # ``TAI_MANIFEST_PATH`` as seen when the app context is entered — proves the
        # lifespan stamp is live by then.
        self.manifest_env_at_context: list[str | None] = []

    @asynccontextmanager
    async def app_context(self, manifest):
        self.manifest_env_at_context.append(os.environ.get("TAI_MANIFEST_PATH"))
        if self.raise_in_context:
            raise RuntimeError("boot boom")
        yield

    def http_app(self, stateless_http: bool | None = None) -> _FakeInnerApp:
        self.http_called = True
        self.http_stateless = stateless_http
        return self.inner

    def sse_app(self) -> _FakeInnerApp:
        self.sse_called = True
        return self.inner


@pytest.fixture
def patch_factory_seam(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``app`` + bypass real ``Manifest`` validation for the factory."""

    def _install(app: _FakeApp) -> _FakeApp:
        monkeypatch.setattr(asgi.instance, "build_app", lambda: app)
        monkeypatch.setattr(asgi.Manifest, "model_validate", staticmethod(lambda data: SimpleNamespace()))
        return app

    return _install


@pytest.fixture(autouse=True)
def _reset_one_app_guard():
    """Reset the process-global one-app token around every test so a failed enter in
    one test can never wedge another."""
    asgi._app_active = False
    asgi._active_manifest_marker = ""
    try:
        yield
    finally:
        asgi._app_active = False
        asgi._active_manifest_marker = ""


# --- argument validation --------------------------------------------------


def test_bad_transport_raises_at_call_time() -> None:
    with pytest.raises(ValueError, match="transport must be one of"):
        asgi.create_app(transport="bogus")  # type: ignore[arg-type]


def test_stateless_sse_raises_at_call_time() -> None:
    with pytest.raises(ValueError, match="no stateless mode"):
        asgi.create_app(transport="sse", stateless_http=True)


def test_wrapper_garbage_transport_raises(root_logger_restored, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI wrapper delegates to the validating factory, so a hand-set garbage
    # TAI_TRANSPORT env raises loudly rather than serving the default http transport.
    monkeypatch.setenv("TAI_TRANSPORT", "garbage")
    monkeypatch.delenv("TAI_STATELESS_HTTP", raising=False)
    monkeypatch.setattr(mcp_app.instance, "build_app", lambda: _FakeApp())
    with pytest.raises(ValueError, match="transport must be one of"):
        mcp_app.create_app()


def test_wrapper_sse_stateless_raises(root_logger_restored, monkeypatch: pytest.MonkeyPatch) -> None:
    # sse + TAI_STATELESS_HTTP=1 raises loudly rather than sse silently winning.
    monkeypatch.setenv("TAI_TRANSPORT", "sse")
    monkeypatch.setenv("TAI_STATELESS_HTTP", "1")
    monkeypatch.setattr(mcp_app.instance, "build_app", lambda: _FakeApp())
    with pytest.raises(ValueError, match="no stateless mode"):
        mcp_app.create_app()


# --- manifest param stamp -------------------------------------------------


def test_manifest_param_not_stamped_at_call_time(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    patch_factory_seam(_FakeApp())

    asgi.create_app(manifest_path="m1")  # constructed, never entered

    assert "TAI_MANIFEST_PATH" not in os.environ


def test_manifest_param_stamped_at_lifespan_entry(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    app = patch_factory_seam(_FakeApp())

    with TestClient(asgi.create_app(manifest_path="m1")):
        pass

    assert app.manifest_env_at_context == ["m1"]


def test_manifest_env_deleted_after_exit_when_absent(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    patch_factory_seam(_FakeApp())

    with TestClient(asgi.create_app(manifest_path="m1")):
        pass

    # A following no-param app must resolve the config-dir default, not m1.
    assert "TAI_MANIFEST_PATH" not in os.environ


def test_manifest_env_prior_value_restored_after_exit(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_MANIFEST_PATH", "prior")
    app = patch_factory_seam(_FakeApp())

    with TestClient(asgi.create_app(manifest_path="m1")):
        pass

    assert app.manifest_env_at_context == ["m1"]
    assert os.environ["TAI_MANIFEST_PATH"] == "prior"


def test_omitted_param_never_writes_env(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    app = patch_factory_seam(_FakeApp())

    with TestClient(asgi.create_app()):
        pass

    assert app.manifest_env_at_context == [None]
    assert "TAI_MANIFEST_PATH" not in os.environ


def test_steal_window_first_app_reads_own_manifest(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build app A (m1), build app B (m2) with no lifespan, then enter A: A must read
    # m1 — the stamp is per-lifespan-entry, not a call-time global a later call steals.
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    app_a = patch_factory_seam(_FakeApp())

    star_a = asgi.create_app(manifest_path="m1")
    asgi.create_app(manifest_path="m2")  # constructed, never entered

    with TestClient(star_a):
        pass

    assert app_a.manifest_env_at_context == ["m1"]


def test_stamp_failure_rolls_back_one_app_claim(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    # A manifest_path with an embedded NUL byte makes the env stamp raise
    # (``os.environ`` rejects null bytes). The stamp runs after the one-app claim, so
    # the claim MUST roll back — otherwise a stamp failure wedges the guard for the
    # process lifetime and no later app can ever enter.
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    patch_factory_seam(_FakeApp())

    with pytest.raises(ValueError, match="null byte"), TestClient(asgi.create_app(manifest_path="bad\x00path")):
        pass

    # The guard was rolled back, not wedged: a following normal app enters cleanly.
    with TestClient(asgi.create_app(manifest_path="m1")):
        pass


# --- transport selection --------------------------------------------------


def test_transport_http_reaches_http_app(patch_factory_seam) -> None:
    app = patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(transport="http")) as client:
        client.get("/")
    assert app.http_called is True
    assert app.http_stateless is None
    assert app.sse_called is False


def test_transport_streamable_stateless_reaches_stateless_http_app(patch_factory_seam) -> None:
    app = patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(transport="streamable-http", stateless_http=True)) as client:
        client.get("/")
    assert app.http_called is True
    assert app.http_stateless is True


def test_transport_sse_reaches_sse_app(patch_factory_seam) -> None:
    app = patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(transport="sse")) as client:
        client.get("/")
    assert app.sse_called is True
    assert app.http_called is False


# --- one-app-per-process guard --------------------------------------------


def test_second_lifespan_raises_naming_active_manifest(patch_factory_seam) -> None:
    patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(manifest_path="mA")):
        with (
            pytest.raises(RuntimeError, match="already active") as excinfo,
            TestClient(asgi.create_app(manifest_path="mB")),
        ):
            pass
        assert "mA" in str(excinfo.value)


def test_second_lifespan_message_env_resolved_marker(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    patch_factory_seam(_FakeApp())
    # The active app (entered first) has no manifest param, so its guard marker is
    # the env-resolved marker; the second app's enter raises naming it.
    with (
        TestClient(asgi.create_app()),
        pytest.raises(RuntimeError, match="<env-resolved manifest>"),
        TestClient(asgi.create_app(manifest_path="mB")),
    ):
        pass


def test_sequential_lifespans_are_legal(patch_factory_seam) -> None:
    patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(manifest_path="m1")):
        pass
    with TestClient(asgi.create_app(manifest_path="m2")):
        pass  # enter -> exit -> enter stays legal


def test_failed_boot_releases_token(patch_factory_seam) -> None:
    patch_factory_seam(_FakeApp(raise_in_context=True))
    with pytest.raises(RuntimeError, match="boot boom"), TestClient(asgi.create_app()):
        pass

    # The token was released, so a fresh healthy app enters normally.
    patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app()):
        pass


def test_failed_claim_keeps_active_token_and_leaves_env(patch_factory_seam, monkeypatch: pytest.MonkeyPatch) -> None:
    # With app A active, app B's enter fails the guard; app C's enter must STILL
    # raise (B's failed claim did not release A's live token), and B's manifest_path
    # must never have been stamped (the stamp is after the claim).
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    patch_factory_seam(_FakeApp())
    with TestClient(asgi.create_app(manifest_path="mA")):
        assert os.environ["TAI_MANIFEST_PATH"] == "mA"

        with pytest.raises(RuntimeError, match="already active"), TestClient(asgi.create_app(manifest_path="mB")):
            pass
        assert os.environ["TAI_MANIFEST_PATH"] == "mA"  # B's failed claim did not stamp mB

        with pytest.raises(RuntimeError, match="already active"), TestClient(asgi.create_app()):
            pass

    assert "TAI_MANIFEST_PATH" not in os.environ


# --- mount in a host process ----------------------------------------------


def test_lifespan_helper_delegates_to_router_context(monkeypatch: pytest.MonkeyPatch) -> None:
    # The public helper returns exactly what the router's ``lifespan_context`` yields
    # for this app, so a host composes it in place of reaching that attribute directly.
    tai = asgi.create_app()
    sentinel = object()
    monkeypatch.setattr(tai.router, "lifespan_context", lambda app: sentinel if app is tai else None)
    assert asgi.lifespan(tai) is sentinel


def test_mount_in_host_reaches_inner_app(patch_factory_seam) -> None:
    tai = asgi.create_app()
    patch_factory_seam(_FakeApp())

    @asynccontextmanager
    async def host_lifespan(_app):
        async with asgi.lifespan(tai):
            yield

    host = Starlette(routes=[Mount("/tai", app=tai)], lifespan=host_lifespan)
    with TestClient(host) as client:
        response = client.get("/tai/")

    assert response.status_code == 200
    assert response.text == "ok"


async def test_dispatch_503_before_lifespan_init(patch_factory_seam) -> None:
    # Reach the dispatch closure without running the lifespan: the inner app is not
    # wired yet, so the fixed ``Initializing...`` 503 answers.
    patch_factory_seam(_FakeApp())
    mount = asgi.create_app().routes[0]
    assert isinstance(mount, Mount)
    dispatch = mount.app

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list = []

    async def send(message):
        sent.append(message)

    await dispatch(dict(_HTTP_SCOPE), receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 503
    # The body is the fixed generic payload, never leaking internal detail.
    body = next(m for m in sent if m["type"] == "http.response.body")
    assert json.loads(body["body"]) == {"error": "Service Unavailable", "detail": "Initializing..."}


# --- root logger untouched by a factory boot ------------------------------


def test_factory_boot_leaves_root_logger_untouched(patch_factory_seam, root_logger_restored) -> None:
    root = root_logger_restored
    level, handlers = root.level, root.handlers[:]
    patch_factory_seam(_FakeApp())

    with TestClient(asgi.create_app()):
        pass

    assert root.level == level
    assert root.handlers == handlers


def test_factory_boot_leaves_host_logger_record_unredacted(patch_factory_seam, root_logger_restored) -> None:
    """Embed scope: with the redactor installed at its default ``tai`` scope, a
    host app's own logger record passes through unredacted, and a factory boot
    never widens that to process scope (only the CLI seams do). The real
    ``build_app`` → default-scope install wire is pinned separately in the
    instance suite — this suite fakes ``build_app``."""
    from tai42_skeleton.connectors import meta_log_redactor

    saved_factory = logging.getLogRecordFactory()
    saved_scope = meta_log_redactor._SCOPE
    logging.setLogRecordFactory(logging.LogRecord)
    meta_log_redactor._SCOPE = "tai"
    try:
        # Mirror the embed install (``build_app`` is faked out in this suite).
        meta_log_redactor.install_meta_log_redactor(scope="tai")
        patch_factory_seam(_FakeApp())
        with TestClient(asgi.create_app()):
            pass

        secret = '{"tai_hub.access_token": "HOST-SECRET"}'
        rec = logging.getLogRecordFactory()("myhost.app", logging.INFO, __file__, 1, secret, None, None)
        assert "HOST-SECRET" in rec.getMessage()
    finally:
        logging.setLogRecordFactory(saved_factory)
        meta_log_redactor._SCOPE = saved_scope
