"""Worker-bus boot rules: three shapes refuse to start without the bus.

* more than one server worker (siblings would serve stale config after a reload),
* a task backend registered in the manifest (server + backend-runtime must
  converge), and
* ``TAI_CONFIG_MODE=k8s`` (multi-pod shared config).

Every refusal names ``TAI_BUS_REDIS_URL`` so the operator knows the fix, and the
k8s check runs BEFORE any config-manager construction so a busless k8s boot fails
on the bus var rather than first on a kubeconfig connection. The single-worker,
file-mode, no-backend, no-bus shape is supported and runs on ``WorkerBus.local``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest
from tai_kit.settings import reset_all_settings

from tai_skeleton.app import boot_rules
from tai_skeleton.cli import mcp_app

if TYPE_CHECKING:
    from tai_skeleton.manifest import Manifest


@pytest.fixture(autouse=True)
def _reset_settings_after() -> Iterator[None]:
    yield
    reset_all_settings()


def _busless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()


def _with_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_BUS_REDIS_URL", "redis://localhost:6379/0")
    reset_all_settings()


class _Manifest:
    def __init__(self, backend_module: str) -> None:
        self.backend_module = backend_module


# -- workers rule -------------------------------------------------------------


def test_multi_worker_busless_refuses_naming_the_bus_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _busless(monkeypatch)
    with pytest.raises(RuntimeError, match="TAI_BUS_REDIS_URL") as exc:
        boot_rules.require_bus_for_workers(4)
    assert "4 workers" in str(exc.value)


def test_multi_worker_with_bus_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    boot_rules.require_bus_for_workers(4)


def test_single_worker_busless_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _busless(monkeypatch)
    boot_rules.require_bus_for_workers(1)


def test_run_mcp_app_serve_dash_w4_busless_raises_naming_the_bus_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _busless(monkeypatch)
    with pytest.raises(RuntimeError, match="TAI_BUS_REDIS_URL"):
        mcp_app.run_mcp_app(
            manifest_path="unused.yml",
            transport="http",
            host="127.0.0.1",
            port=8000,
            workers=4,
            stateless_http=True,
        )


# -- backend rule -------------------------------------------------------------


def test_registered_backend_busless_refuses_naming_the_bus_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _busless(monkeypatch)
    with pytest.raises(RuntimeError, match="TAI_BUS_REDIS_URL"):
        boot_rules.require_bus_for_backend(cast("Manifest", _Manifest("myapp.backend")))


def test_registered_backend_with_bus_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_bus(monkeypatch)
    boot_rules.require_bus_for_backend(cast("Manifest", _Manifest("myapp.backend")))


def test_no_backend_busless_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _busless(monkeypatch)
    boot_rules.require_bus_for_backend(cast("Manifest", _Manifest("")))


# -- k8s rule (fails on the bus var, BEFORE any kubeconfig connection) --------


def test_k8s_mode_busless_refuses_on_the_bus_var_not_kubeconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_CONFIG_MODE", "k8s")
    _busless(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        boot_rules.require_bus_for_k8s()
    message = str(exc.value)
    # The refusal names the bus var and never mentions kubeconfig — the check runs
    # before the config manager is ever constructed.
    assert "TAI_BUS_REDIS_URL" in message
    assert "kubeconfig" not in message.lower()
    assert "configmap" not in message.lower()


def test_k8s_mode_with_bus_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_CONFIG_MODE", "k8s")
    _with_bus(monkeypatch)
    boot_rules.require_bus_for_k8s()


def test_file_mode_busless_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_CONFIG_MODE", "file")
    _busless(monkeypatch)
    boot_rules.require_bus_for_k8s()


def test_run_mcp_app_k8s_busless_raises_on_the_bus_var_before_kubeconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    # A busless k8s-mode `tai serve` boot refuses naming TAI_BUS_REDIS_URL, never a
    # kubeconfig/connection error — the check precedes config-manager construction.
    monkeypatch.setenv("TAI_CONFIG_MODE", "k8s")
    _busless(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        mcp_app.run_mcp_app(
            manifest_path="unused.yml",
            transport="http",
            host="127.0.0.1",
            port=8000,
            workers=1,
        )
    assert "TAI_BUS_REDIS_URL" in str(exc.value)
