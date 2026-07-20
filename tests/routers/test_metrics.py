"""Metrics router coverage: the non-wiping ensure-exists initializer, the
master-only wipe (first wipe + sentinel early-return + re-wipe on a new run id),
the writer-only mmap-value-class assert, the absolute/relative multiproc-dir
settings validation, a real cross-process scrape of a spawned worker's counter,
the /metrics endpoint rendering an empty multiproc registry, the mode-aware
render fork (multiproc vs in-process, logged once on the first scrape), and the
import-time multiproc-dir gate (skipped without the env, run with it).
Filesystem state is under ``tmp_path``; the cross-process scrape, the writer
import-order probes, and the import-time gate tests each spawn a subprocess."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import cast

import pytest
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response

from tai42_skeleton.routers import metrics as metrics_mod
from tai42_skeleton.routers import prometheus as prom_mod
from tai42_skeleton.routers.metrics_settings import MetricsSettings
from tai42_skeleton.routers.prometheus import render_multiproc_metrics


class _FakeRequest:
    pass


def _fake_settings(multiproc_dir: str) -> MetricsSettings:
    return MetricsSettings(prometheus_multiproc_dir=multiproc_dir)


def test_metrics_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # The suite sets ``PROMETHEUS_MULTIPROC_DIR`` (the mmap freeze); drop it so the
    # field resolves to its coded default. The default is a FIXED absolute path
    # (host tempdir), CWD-independent so every run-family process agrees on one dir.
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    settings = MetricsSettings()
    assert settings.backend_metrics_host == "127.0.0.1"
    assert settings.backend_metrics_port == 8012
    expected = os.path.join(tempfile.gettempdir(), "tai42_prometheus")
    assert settings.prometheus_multiproc_dir == expected
    assert os.path.isabs(settings.prometheus_multiproc_dir)  # the default passes its own validator


def test_metrics_settings_rejects_relative_override_argument() -> None:
    # A relative override resolves per-CWD and splits the shared dir — refuse loudly.
    with pytest.raises(ValidationError, match="absolute path"):
        MetricsSettings(prometheus_multiproc_dir="relative/tai42_prometheus")


def test_metrics_settings_rejects_relative_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "relative/tai42_prometheus")
    with pytest.raises(ValidationError, match="absolute path"):
        MetricsSettings()


def test_metrics_settings_accepts_absolute_override(tmp_path) -> None:
    settings = MetricsSettings(prometheus_multiproc_dir=str(tmp_path))
    assert settings.prometheus_multiproc_dir == str(tmp_path)


def test_init_ensures_dir_without_wiping(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``init`` is the reader/worker ensure-exists path: it creates the dir if
    missing but NEVER removes a populated one, so a live worker's mmap files
    survive a metrics-server or backend-worker start against a shared dir."""
    target = tmp_path / "metrics"
    target.mkdir()
    (target / "live.db").write_text("live", encoding="utf-8")
    monkeypatch.setattr(prom_mod, "metrics_settings", lambda: _fake_settings(str(target)))

    returned = prom_mod.init_prometheus_multiproc_dir()

    assert returned == str(target)
    assert target.is_dir()
    assert (target / "live.db").exists()  # ensure-only: a pre-existing file survives


def test_init_creates_missing_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    target = tmp_path / "metrics"
    monkeypatch.setattr(prom_mod, "metrics_settings", lambda: _fake_settings(str(target)))

    assert prom_mod.init_prometheus_multiproc_dir() == str(target)
    assert target.is_dir()


def test_wipe_removes_stale_and_stamps_sentinel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    target = tmp_path / "metrics"
    # Pre-create with a stale file to prove the master wipe clears it.
    target.mkdir()
    (target / "stale.db").write_text("old", encoding="utf-8")
    monkeypatch.setattr(prom_mod, "metrics_settings", lambda: _fake_settings(str(target)))

    returned = prom_mod.wipe_prometheus_multiproc_dir()

    assert returned == str(target)
    assert target.is_dir()
    assert not (target / "stale.db").exists()  # wiped
    assert (target / ".init_done").exists()  # sentinel created


def test_wipe_second_call_same_run_early_returns(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    target = tmp_path / "metrics"
    monkeypatch.setattr(prom_mod, "metrics_settings", lambda: _fake_settings(str(target)))

    prom_mod.wipe_prometheus_multiproc_dir()  # creates sentinel
    marker = target / "keep.db"
    marker.write_text("kept", encoding="utf-8")

    prom_mod.wipe_prometheus_multiproc_dir()  # sentinel present, same run -> no wipe

    assert marker.exists()  # second call did not wipe


def test_wipe_rewipes_on_new_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A new run id re-wipes even though the sentinel exists. A pid-based marker
    would wrongly skip the wipe when consecutive runs share a parent pid."""
    target = tmp_path / "metrics"
    monkeypatch.setattr(prom_mod, "metrics_settings", lambda: _fake_settings(str(target)))

    monkeypatch.setenv("TAI_METRICS_RUN_ID", "run-1")
    prom_mod.wipe_prometheus_multiproc_dir()
    kept = target / "keep.db"
    kept.write_text("x", encoding="utf-8")

    monkeypatch.setenv("TAI_METRICS_RUN_ID", "run-2")
    prom_mod.wipe_prometheus_multiproc_dir()

    assert not kept.exists()  # a new run id forced the wipe to re-run


def test_assert_multiproc_value_class_passes_under_mmap() -> None:
    # The suite freezes the mmap value backend (conftest sets the env before the
    # first ``prometheus_client`` import), so a writer's assert is satisfied.
    prom_mod.assert_multiproc_value_class()  # no raise


def test_assert_multiproc_value_class_raises_under_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MutexValue:
        _multiprocess = False

    monkeypatch.setattr(prom_mod.values, "ValueClass", _MutexValue)
    with pytest.raises(RuntimeError, match="multiprocess mmap backend"):
        prom_mod.assert_multiproc_value_class()


@pytest.mark.parametrize("module", ["tai42_skeleton.cli.mcp_app", "tai42_skeleton.cli.backend"])
def test_writer_cli_import_does_not_preload_prometheus(module: str) -> None:
    """Each WRITER entry point must publish ``PROMETHEUS_MULTIPROC_DIR`` before
    anything imports ``prometheus_client`` (which freezes its value backend). That
    the module import itself pulls in NO ``prometheus_client`` proves the env-set
    inside ``run_mcp_app`` / ``main`` runs first — the import happens only later,
    from inside those functions, after the env is set."""
    probe = (
        f"import sys; import {module}; "
        "assert 'prometheus_client' not in sys.modules, "
        "sorted(m for m in sys.modules if 'prometheus' in m)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_counter_from_spawned_process_is_visible_to_scrape(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A counter incremented in a SEPARATE process whose env sets the shared
    multiproc dir before ``prometheus_client`` imports is visible to a fresh
    collector scrape in this process — the mmap files carry across the process
    boundary."""
    mp_dir = tmp_path / "mp"
    mp_dir.mkdir()

    child = textwrap.dedent(
        """
        from prometheus_client import Counter
        from prometheus_client.values import ValueClass

        # The env (set by the parent below) must have frozen the mmap backend.
        assert ValueClass.__name__ == "MmapedValue", ValueClass.__name__
        counter = Counter("tai_worker_probe_total", "cross-process probe", ["runtime"])
        counter.labels(runtime="worker").inc(3)
        """
    )
    # The env is set BEFORE the child interpreter starts, exactly as a spawned
    # uvicorn worker inherits it from the master — the ordering the mmap backend
    # requires.
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(mp_dir)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Scrape from THIS process against the same dir; the collector reads the child's
    # mmap file left behind after it exited.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(mp_dir))
    exposition = render_multiproc_metrics().decode()

    assert "tai_worker_probe_total" in exposition
    assert 'runtime="worker"' in exposition
    assert 'tai_worker_probe_total{runtime="worker"} 3.0' in exposition


def test_backend_main_activates_multiproc_before_prometheus_import_in_clean_env() -> None:
    """``cli/backend.py::main`` must call ``activate_multiproc_env()`` BEFORE the
    statement that imports ``routers.prometheus`` (which pulls in
    ``prometheus_client`` and freezes its value backend). That is a statement-order
    contract inside one function, invisible to a same-process test because conftest
    already froze the value class for this interpreter.

    Drive ``main`` in a FRESH interpreter with ``PROMETHEUS_MULTIPROC_DIR`` UNSET:
    if activation runs first, the child freezes the mmap value class and the
    in-``main`` ``assert_multiproc_value_class`` passes (exit 0); if the prometheus
    import runs first, it freezes the in-process mutex and that assert raises
    (non-zero exit). ``run_backend`` and the event loop are stubbed so ``main`` runs
    its activate→import→assert prelude without building or serving the app. This is
    the test that catches a writer whose import order would silently lose every tool
    counter to the mutex backend."""
    child = textwrap.dedent(
        """
        import asyncio

        import tai42_skeleton.cli.backend as backend
        from click.testing import CliRunner

        async def _noop():
            return None

        # Stop main before it builds or serves; only its activate+import+assert
        # prelude must run.
        backend.run_backend = lambda args: _noop()
        try:
            import uvloop

            uvloop.run = lambda coro: coro.close()
        except ImportError:
            pass
        asyncio.run = lambda coro: coro.close()

        # catch_exceptions=False so a raised assert propagates to a non-zero exit
        # rather than being swallowed into result.exception.
        result = CliRunner().invoke(backend.main, ["worker"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        from prometheus_client.values import ValueClass

        assert ValueClass.__name__ == "MmapedValue", ValueClass.__name__
        print("MmapedValue")
        """
    )
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)  # a clean env, as production launches into
    result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "MmapedValue" in result.stdout


def test_mcp_app_master_activates_multiproc_before_prometheus_import_in_clean_env(tmp_path) -> None:
    """``run_mcp_app`` (the mcp_app writer master) must call
    ``activate_multiproc_env()`` BEFORE the statement that imports the wipe helper
    from ``routers.prometheus`` (which pulls in ``prometheus_client`` and freezes its
    value backend). Same statement-order contract as the backend, checked in a FRESH
    interpreter with ``PROMETHEUS_MULTIPROC_DIR`` UNSET.

    ``run_stdio`` is stubbed so the master runs its env-stamp→activate→import→wipe
    prelude without building or serving; the child then asserts the mmap value class
    froze, which only holds if activation ran before the prometheus import. ``TMPDIR``
    is redirected into ``tmp_path`` so the master's real wipe touches tmp, not the
    host tempdir, while the env var itself stays unset so the freeze must come from
    activation."""
    child = textwrap.dedent(
        """
        import tai42_skeleton.cli.mcp_app as mcp_app
        from tai42_skeleton.settings.cache import app_args_settings

        async def _fake_run_stdio():
            return 0

        # Skip the real stdio serve (build_app + app_context); the activate→import→wipe
        # prelude ahead of it is what carries the ordering contract.
        mcp_app.run_stdio = _fake_run_stdio

        # stdio refuses a host/port that differs from the resolved defaults, so pass the
        # defaults through unchanged.
        defaults = app_args_settings()
        mcp_app.run_mcp_app(
            manifest_path="unused.yaml",
            transport="stdio",
            host=defaults.host,
            port=defaults.port,
            workers=1,
        )

        from prometheus_client.values import ValueClass

        assert ValueClass.__name__ == "MmapedValue", ValueClass.__name__
        print("MmapedValue")
        """
    )
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)  # a clean env, as production launches into
    env["TMPDIR"] = str(tmp_path)  # redirect the coded-default dir the real wipe touches
    result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "MmapedValue" in result.stdout


def test_lock_helpers_provide_real_mutual_exclusion(tmp_path) -> None:
    """The lock helpers are REAL exclusion, never a no-op: while one fd holds the
    posix exclusive lock, a second fd's non-blocking acquire fails. (The win32
    branch uses ``msvcrt.locking``, unimportable here, so this pins the posix
    path — the one that runs in this environment.)"""
    import fcntl

    lock_path = tmp_path / "x.lock"
    with open(lock_path, "w") as first:
        prom_mod._lock_exclusive(first.fileno())
        try:
            # A held exclusive lock blocks a second fd's non-blocking acquire.
            with open(lock_path, "w") as second, pytest.raises(BlockingIOError):
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            prom_mod._unlock(first.fileno())


async def test_metrics_endpoint_renders_prometheus_exposition(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from prometheus_client import CONTENT_TYPE_LATEST

    target = tmp_path / "mp"
    target.mkdir()
    # The multiproc dir is owned by ``init_prometheus_multiproc_dir`` (env),
    # not stamped per request — the endpoint just renders whatever dir the
    # environment points the collector at.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

    # The endpoint never reads the request; starlette's concrete Request can't be
    # matched structurally, so cast the empty stand-in.
    response = await metrics_mod.metrics_endpoint(cast(Request, _FakeRequest()))

    assert isinstance(response, Response)
    # The canonical Prometheus content type (with format version), not bare text.
    assert response.media_type == CONTENT_TYPE_LATEST
    # An empty multiproc dir yields an empty exposition, not an error.
    assert response.body == b""


async def test_metrics_endpoint_renders_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # A blocking render must not stall the serving loop — the handler offloads it
    # to a thread, so a concurrently awaited ticker keeps advancing.
    import asyncio
    import time

    ticks: list[int] = []

    def _slow_render() -> bytes:
        time.sleep(0.3)
        return b"# rendered"

    monkeypatch.setattr(metrics_mod, "render_metrics", _slow_render)

    async def ticker() -> None:
        for _ in range(20):
            ticks.append(1)
            await asyncio.sleep(0.01)

    tick_task = asyncio.ensure_future(ticker())
    response = await metrics_mod.metrics_endpoint(cast(Request, _FakeRequest()))
    await tick_task
    assert response.body == b"# rendered"
    assert len(ticks) == 20


# --- mode-aware render fork -------------------------------------------------


def test_render_metrics_multiproc_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list = []
    monkeypatch.setattr(prom_mod, "multiproc_active", lambda: True)
    monkeypatch.setattr(prom_mod, "render_multiproc_metrics", lambda: called.append(True) or b"MP")

    assert prom_mod.render_metrics() == b"MP"
    assert called == [True]


def test_render_metrics_in_process_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prom_mod, "multiproc_active", lambda: False)
    monkeypatch.setattr(
        prom_mod, "generate_latest", lambda registry: b"INPROC" if registry is prom_mod.REGISTRY else b"WRONG"
    )

    # The in-process branch renders the module-global default registry.
    assert prom_mod.render_metrics() == b"INPROC"


def test_render_metrics_logs_mode_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # The served mode is announced exactly once — on the first scrape, naming the
    # mode and the frozen value backend that decided it — never per scrape.
    monkeypatch.setattr(prom_mod, "multiproc_active", lambda: True)
    monkeypatch.setattr(prom_mod, "render_multiproc_metrics", lambda: b"MP")
    monkeypatch.setattr(prom_mod, "_mode_logged", False)

    with caplog.at_level(logging.INFO, logger="tai42_skeleton.routers.prometheus"):
        prom_mod.render_metrics()
        first = [r for r in caplog.records if "/metrics in" in r.getMessage()]
        assert len(first) == 1
        assert "multiproc" in first[0].getMessage()

        caplog.clear()
        prom_mod.render_metrics()
        assert not [r for r in caplog.records if "/metrics in" in r.getMessage()]


# --- import-time multiproc-dir gate -----------------------------------------


def test_metrics_dir_init_skipped_without_multiproc_env(tmp_path) -> None:
    # Importing the metrics router in an EMBEDDED process (no PROMETHEUS_MULTIPROC_DIR)
    # must NOT create the multiproc dir on the host filesystem.
    child = textwrap.dedent(
        """
        import os

        from tai42_contract.app import tai42_app
        from tai42_skeleton.app.route_registry import _SpecApp

        tai42_app.bind(_SpecApp())
        import tai42_skeleton.routers.metrics  # noqa: F401

        from tai42_skeleton.routers.metrics_settings import metrics_settings

        d = metrics_settings().prometheus_multiproc_dir
        assert not os.path.exists(d), d
        print("NOT_CREATED")
        """
    )
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)  # embedded: no multiproc env
    env["TMPDIR"] = str(tmp_path)  # redirect the coded-default dir off the shared host tempdir
    result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "NOT_CREATED" in result.stdout


def test_metrics_dir_init_runs_with_multiproc_env(tmp_path) -> None:
    # With the multiproc env claimed, importing the router DOES ensure the dir —
    # otherwise an accidentally-always-false gate would pass the skip test too.
    mp_dir = tmp_path / "mp_gate"
    child = textwrap.dedent(
        """
        import os

        from tai42_contract.app import tai42_app
        from tai42_skeleton.app.route_registry import _SpecApp

        tai42_app.bind(_SpecApp())
        import tai42_skeleton.routers.metrics  # noqa: F401

        d = os.environ["PROMETHEUS_MULTIPROC_DIR"]
        assert os.path.isdir(d), d
        print("CREATED")
        """
    )
    env = os.environ.copy()
    env["PROMETHEUS_MULTIPROC_DIR"] = str(mp_dir)  # claimed, dir does not pre-exist
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CREATED" in result.stdout
