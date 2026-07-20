"""Metrics server launcher.

``main`` is a launch-only CLI: its only logic is binding the resolved host/port
and handing the FastAPI app to ``uvicorn.run`` (the blocking call, mocked). The
``/metrics`` handler is exercised directly against an empty multiproc dir so the
prometheus scrape wiring runs without a live server.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from fastapi.responses import Response

import tai42_skeleton.cli.metrics as metrics


def test_main_launches_uvicorn_with_default_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    sentinel_app = object()
    monkeypatch.setattr(metrics, "create_app", lambda: sentinel_app)
    monkeypatch.setattr(metrics.uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))

    result = CliRunner().invoke(metrics.main, [])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["app"] is sentinel_app
    settings = metrics.metrics_settings()
    assert call["host"] == settings.backend_metrics_host
    assert call["port"] == settings.backend_metrics_port
    assert call["reload"] is False


def test_main_publishes_multiproc_dir_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    # The metrics server is a pure reader: it publishes the shared multiproc dir so
    # the collector reads it at scrape time (its own value class already froze at
    # import and is never written). The env must be set by the time uvicorn serves.
    # The suite-wide conftest pre-sets ``PROMETHEUS_MULTIPROC_DIR`` (to freeze the mmap
    # value class), so asserting the published env ``==`` a fresh ``metrics_settings()``
    # reading that same env would hold even if ``main`` published nothing. Drop the env
    # and reset the settings cache so ``metrics_settings()`` falls back to its CODED
    # absolute default, then assert ``main`` published exactly that.
    import os
    import tempfile

    from tai42_kit.settings import reset_all_settings

    captured: dict[str, str | None] = {}

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    reset_all_settings()
    monkeypatch.setattr(metrics, "create_app", lambda: object())
    monkeypatch.setattr(
        metrics.uvicorn,
        "run",
        lambda app, **kw: captured.update(dir=os.environ.get("PROMETHEUS_MULTIPROC_DIR")),
    )

    try:
        result = CliRunner().invoke(metrics.main, [])

        assert result.exit_code == 0, result.output
        published = captured["dir"]
        # ``main``'s ``activate_multiproc_env()`` must publish the coded default: the
        # host-tempdir absolute path the settings field resolves to with the env unset.
        assert published == os.path.join(tempfile.gettempdir(), "tai42_prometheus")
        assert published is not None
        assert os.path.isabs(published)
    finally:
        # ``main`` cached ``metrics_settings()`` at the coded default while the env was
        # unset; clear the cache so it rebuilds against the conftest env (restored by
        # ``cli/conftest.py``) that the rest of the suite relies on.
        reset_all_settings()


def test_main_launches_uvicorn_with_overridden_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(metrics, "create_app", lambda: object())
    monkeypatch.setattr(metrics.uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))

    result = CliRunner().invoke(metrics.main, ["--host", "0.0.0.0", "--port", "9999"])

    assert result.exit_code == 0
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9999


def test_main_bootstraps_env_when_not_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(metrics, "config_mode", lambda: "file")
    monkeypatch.setattr(metrics, "load_dotenv", lambda: called.append(True))
    monkeypatch.setattr(metrics, "create_app", lambda: object())
    monkeypatch.setattr(metrics.uvicorn, "run", lambda app, **kw: None)

    result = CliRunner().invoke(metrics.main, [])

    assert result.exit_code == 0, result.output
    assert called == [True]


def test_main_skips_env_bootstrap_in_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(metrics, "config_mode", lambda: "k8s")
    monkeypatch.setattr(metrics, "load_dotenv", lambda: called.append(True))
    monkeypatch.setattr(metrics, "create_app", lambda: object())
    monkeypatch.setattr(metrics.uvicorn, "run", lambda app, **kw: None)

    result = CliRunner().invoke(metrics.main, [])

    assert result.exit_code == 0, result.output
    assert called == []


def test_create_app_registers_metrics_route() -> None:
    app = metrics.create_app()

    routes = {getattr(route, "path", None) for route in app.routes}
    assert "/metrics" in routes


async def test_get_metrics_returns_prometheus_text(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Point the scrape at an empty multiproc dir so the collector reads no
    # samples and the endpoint returns an empty (but valid) exposition body.
    # The dir is owned by the init-time environment, not stamped per request.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    response = await metrics.get_metrics()

    assert isinstance(response, Response)
    assert response.media_type == "text/plain"
    # An empty multiproc dir yields an empty (but valid) exposition body.
    assert bytes(response.body) == b""
