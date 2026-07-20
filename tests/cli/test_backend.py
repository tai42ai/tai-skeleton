"""Worker-backend launcher.

``run_backend`` launches every invocation inside ``app_context`` — which binds
the ``tai42_app`` handle and imports the manifest's backend module, so a plugin
registering its Backend at import lands in the bound holder. It joins the worker
bus under the ``backend`` origin kind, and publishes the resolved manifest into the
env for the WORKER runtime only — the one whose forked children need it. The
``main`` Click entry just selects an event loop (uvloop, else asyncio) and hands
the coroutine to its blocking ``run`` — that blocking run is mocked so the
launcher wiring (env publish, arg merge) is exercised without a real loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import tai42_skeleton.cli.backend as backend
from tai42_skeleton.connectors import meta_log_redactor

_LOGGING_RELOAD_KEY = "tai42_skeleton.app.instance.apply_logging_settings"


@pytest.fixture(autouse=True)
def _restore_log_record_factory():
    """Save/restore the process-global record factory and its monotonic redaction
    scope around every test: ``run_backend`` installs the connector-secret redactor
    at process scope, so this keeps one test's install from leaking onward."""
    saved_factory = logging.getLogRecordFactory()
    saved_scope = meta_log_redactor._SCOPE
    try:
        yield
    finally:
        logging.setLogRecordFactory(saved_factory)
        meta_log_redactor._SCOPE = saved_scope


class _FakeConfigManager:
    def read_manifest(self) -> dict:
        return {}


class _FakeManifest:
    backend_module = "my.backend.module"

    def model_dump_json(self) -> str:
        return '{"backend_module": "my.backend.module"}'


class _FakeLifecycle:
    """Records reload-handler registrations by their ``module.qualname`` key, so a
    test can assert ``register_cli_logging_reload`` wired ``apply_logging_settings``."""

    def __init__(self) -> None:
        self.reload_handlers: dict[str, object] = {}

    def on_reload(self, func):
        self.reload_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func


class _FakeApp:
    def __init__(self) -> None:
        # The config manager is reached through the ``config`` facet namespace.
        self.config = SimpleNamespace(config_manager=_FakeConfigManager())
        # ``run_backend`` registers the root-logger reload handler through the app's
        # lifecycle; the recorder captures it without a real lifecycle.
        self.lifecycle = _FakeLifecycle()
        self.run_backend_args: list = []
        self.context_entered = False
        # The env as it stands the moment the context opens. A worker forks job
        # children that inherit the resolved manifest, so the launcher must publish it
        # BEFORE the context opens (start() imports the backend module inside it).
        self.manifest_env_at_entry: str | None = None
        # The origin kind the launcher joined the worker bus under: a backend runtime
        # subscribes as ``backend``, not ``serve``.
        self.origin_kind_at_entry = None

    @asynccontextmanager
    async def app_context(self, manifest, *, origin_kind=None):
        self.context_entered = True
        self.origin_kind_at_entry = origin_kind
        settings = backend.base_backend_settings()
        self.manifest_env_at_entry = os.environ.get(settings.manifest_key)
        yield

    async def run_backend(self, args) -> None:
        self.run_backend_args.append(args)


@pytest.fixture(autouse=True)
def _bus_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend runtime always registers a task backend, so ``run_backend``'s boot
    rule requires the worker bus. Report it configured so the launcher-wiring tests
    exercise the launch path without a real Redis (the app itself builds the no-op
    local bus, which never connects)."""
    import tai42_skeleton.app.boot_rules as boot_rules

    monkeypatch.setattr(boot_rules, "_bus_configured", lambda: True)


@pytest.fixture
def fake_app(monkeypatch: pytest.MonkeyPatch) -> _FakeApp:
    app = _FakeApp()
    # The launcher obtains the app via the deferred factory ``instance.build_app``.
    monkeypatch.setattr(backend.instance, "build_app", lambda: app)
    monkeypatch.setattr(backend.Manifest, "model_validate", staticmethod(lambda data: _FakeManifest()))
    return app


async def test_run_backend_non_worker_enters_context_as_backend_origin(
    fake_app: _FakeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tai42_skeleton.app.bus import OriginKind

    settings = backend.base_backend_settings()
    monkeypatch.delenv(settings.manifest_key, raising=False)

    # A non-worker backend runtime (e.g. a scheduler ``beat``): it launches inside
    # app_context and joins the worker bus under the ``backend`` origin kind. It forks
    # no job children, so the secret-bearing manifest is NOT exported into it.
    await backend.run_backend(["beat"])

    assert fake_app.context_entered is True
    assert fake_app.run_backend_args == [["beat"]]
    assert fake_app.origin_kind_at_entry is OriginKind.backend
    assert fake_app.manifest_env_at_entry is None
    assert settings.manifest_key not in os.environ


async def test_run_backend_worker_publishes_manifest_and_enters_context(
    fake_app: _FakeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = backend.base_backend_settings().manifest_key
    monkeypatch.delenv(key, raising=False)

    await backend.run_backend(["worker"])

    assert fake_app.context_entered is True
    assert fake_app.run_backend_args == [["worker"]]
    # The resolved manifest is published before the context opens so the worker's
    # prefork children inherit it.
    assert fake_app.manifest_env_at_entry == '{"backend_module": "my.backend.module"}'
    assert os.environ[key] == '{"backend_module": "my.backend.module"}'


async def test_run_backend_registers_cli_logging_reload(fake_app: _FakeApp) -> None:
    # This CLI-configured backend process keeps its root logger in sync across config
    # reloads, so ``run_backend`` registers the reload handler through the lifecycle.
    await backend.run_backend(["worker"])

    assert _LOGGING_RELOAD_KEY in fake_app.lifecycle.reload_handlers


async def test_run_backend_installs_process_scope_redactor(fake_app: _FakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    # This CLI-owned backend process widens the connector-secret redactor to the
    # whole process.
    scopes: list[object] = []
    monkeypatch.setattr(backend, "install_meta_log_redactor", lambda **kwargs: scopes.append(kwargs.get("scope")))

    await backend.run_backend(["beat"])

    assert scopes == ["process"]


async def test_run_backend_beat_reaches_import_registered_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A beat-style (non-worker) invocation with a REAL app and a plugin module
    that registers its Backend at import: ``run_backend`` must bind the app
    (``app_context`` -> ``start()``) BEFORE the backend module import runs, so
    the import-time ``tai42_app.backends.register_backend`` lands in the bound
    holder and ``launch`` receives the args. Importing the module before the
    bind instead hits the unbound ``tai42_app`` handle (AttributeError) and the
    invocation never reaches ``launch``."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app import instance

    real_app = instance.build_app()
    bound_before = object.__getattribute__(tai42_app, "_impl")
    # Model a fresh process: the handle is unbound until start() binds it, so an
    # import-before-bind registration raises instead of landing anywhere. The
    # plugin module is therefore NOT imported here — start() imports it, after
    # the bind.
    tai42_app.bind(None)
    saved_backend = real_app._backend_holder._backend
    real_app._backend_holder._backend = None
    try:
        monkeypatch.setattr(
            real_app.config.config_manager,
            "read_manifest",
            lambda: {"backend_module": "tests._fakes.beat_backend"},
        )

        await backend.run_backend(["beat"])

        # start() freshly (re)imported the module inside app_context, so its
        # import-time registration landed and launch received the args.
        plugin = sys.modules["tests._fakes.beat_backend"]
        assert plugin.LaunchRecordingBackend.launched == [["beat"]]
        plugin.LaunchRecordingBackend.launched.clear()
    finally:
        real_app._backend_holder._backend = saved_backend
        tai42_app.bind(bound_before)


def test_main_sets_manifest_env_and_runs_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)

    runs: list = []

    def fake_loop_run(coro):
        coro.close()  # do not execute the launcher coroutine; just record the launch
        runs.append(coro)

    captured: list = []

    def fake_run_backend(args):
        captured.append(args)

        async def _coro() -> None:
            return None

        return _coro()

    monkeypatch.setattr(backend, "run_backend", fake_run_backend)
    # uvloop is the selected loop when importable; mock its blocking run.
    import uvloop

    monkeypatch.setattr(uvloop, "run", fake_loop_run)

    result = CliRunner().invoke(backend.main, ["--manifest-path", "/tmp/m.yaml", "worker", "--flag"])

    assert result.exit_code == 0, result.output
    assert os.environ["TAI_MANIFEST_PATH"] == "/tmp/m.yaml"
    assert len(runs) == 1
    # Unknown options + positional extra args are merged and forwarded.
    assert captured == [["--flag", "worker"]] or captured == [["worker", "--flag"]]


def test_main_falls_back_to_asyncio_when_uvloop_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the ``import uvloop`` to fail so the asyncio fallback branch runs.
    monkeypatch.setitem(sys.modules, "uvloop", None)

    runs: list = []

    def fake_loop_run(coro):
        coro.close()
        runs.append(coro)

    monkeypatch.setattr(backend.asyncio, "run", fake_loop_run)
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    result = CliRunner().invoke(backend.main, ["worker"])

    assert result.exit_code == 0, result.output
    assert len(runs) == 1


def test_main_bootstraps_env_when_not_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(backend, "config_mode", lambda: "file")
    monkeypatch.setattr(backend, "load_dotenv", lambda: called.append(True))
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", lambda coro: coro.close())

    result = CliRunner().invoke(backend.main, ["worker"])

    assert result.exit_code == 0, result.output
    assert called == [True]


def test_main_skips_env_bootstrap_in_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(backend, "config_mode", lambda: "k8s")
    monkeypatch.setattr(backend, "load_dotenv", lambda: called.append(True))
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", lambda coro: coro.close())

    result = CliRunner().invoke(backend.main, ["dashboard"])

    assert result.exit_code == 0, result.output
    assert called == []


def test_main_labels_backend_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # The backend process's tool metrics must carry ``runtime="backend"`` so a
    # scrape tells them from the server's; ``main`` publishes the label.
    monkeypatch.delenv("PROMETHEUS_RUNTIME", raising=False)
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", lambda coro: coro.close())

    result = CliRunner().invoke(backend.main, ["worker"])

    assert result.exit_code == 0, result.output
    assert os.environ["PROMETHEUS_RUNTIME"] == "backend"


def test_main_respects_explicit_runtime_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # An operator-set runtime label is honored, not clobbered.
    monkeypatch.setenv("PROMETHEUS_RUNTIME", "custom")
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", lambda coro: coro.close())

    result = CliRunner().invoke(backend.main, ["worker"])

    assert result.exit_code == 0, result.output
    assert os.environ["PROMETHEUS_RUNTIME"] == "custom"


def test_main_publishes_absolute_multiproc_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``main`` sets the multiproc dir env (from the absolute settings value) before
    # ``run_backend`` builds the app and imports prometheus_client. The suite-wide
    # conftest pre-sets ``PROMETHEUS_MULTIPROC_DIR`` (to freeze the mmap value class),
    # so asserting the env ``==`` a fresh ``metrics_settings()`` reading that same env
    # would hold even if ``main`` published nothing. Drop the env and reset the settings
    # cache so ``metrics_settings()`` falls back to its CODED absolute default, then
    # assert ``main`` published exactly that — a value only production's
    # ``activate_multiproc_env()`` can put back on the env.
    import tempfile

    from tai42_kit.settings import reset_all_settings

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    reset_all_settings()
    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", lambda coro: coro.close())

    try:
        result = CliRunner().invoke(backend.main, ["worker"])

        assert result.exit_code == 0, result.output
        # ``main``'s ``activate_multiproc_env()`` must publish the coded default: the
        # host-tempdir absolute path the settings field resolves to with the env unset.
        expected = os.path.join(tempfile.gettempdir(), "tai42_prometheus")
        assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == expected
        assert os.path.isabs(os.environ["PROMETHEUS_MULTIPROC_DIR"])
    finally:
        # ``main`` cached ``metrics_settings()`` at the coded default while the env was
        # unset; clear the cache so it rebuilds against the conftest env (restored by
        # ``cli/conftest.py``) that the rest of the suite relies on.
        reset_all_settings()


# -- SIGTERM handling --------------------------------------------------------


async def test_run_backend_sigterm_cancels_main_and_runs_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``run_backend`` installs a SIGTERM handler that cancels the main task; the
    # cancellation must unwind through ``app_context``'s teardown before surfacing.
    exited = {"value": False}
    entered = asyncio.Event()

    class _SigApp:
        def __init__(self) -> None:
            self.config = SimpleNamespace(config_manager=_FakeConfigManager())
            self.lifecycle = _FakeLifecycle()

        @asynccontextmanager
        async def app_context(self, manifest, *, origin_kind=None):
            try:
                yield
            finally:
                # Guaranteed teardown the SIGTERM handler must let run.
                exited["value"] = True

        async def run_backend(self, args) -> None:
            entered.set()
            await asyncio.Event().wait()  # block until the cancellation lands

    app = _SigApp()
    monkeypatch.setattr(backend.instance, "build_app", lambda: app)
    monkeypatch.setattr(backend.Manifest, "model_validate", staticmethod(lambda data: _FakeManifest()))

    # Capture the handler ``run_backend`` registers instead of installing a real one.
    captured: dict[int, object] = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb, *a: captured.__setitem__(sig, cb))

    task = asyncio.create_task(backend.run_backend(["worker"]))
    await entered.wait()

    # Deliver SIGTERM by invoking the captured handler (== main_task.cancel).
    assert signal.SIGTERM in captured
    captured[signal.SIGTERM]()  # type: ignore[operator]

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert exited["value"] is True


def test_main_catches_sigterm_cancellation_as_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``main`` converts the deliberate SIGTERM-driven cancellation into a clean exit
    # (exit code 0), not a crash.
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)

    def fake_loop_run(coro):
        coro.close()
        raise asyncio.CancelledError

    monkeypatch.setattr(backend, "run_backend", lambda args: _noop_coro())

    import uvloop

    monkeypatch.setattr(uvloop, "run", fake_loop_run)

    result = CliRunner().invoke(backend.main, ["worker"])

    assert result.exit_code == 0, result.output


async def _noop_coro() -> None:
    return None
