"""``cli/metrics.py`` identity-provider wiring.

The metrics entrypoint builds its OWN ``AuthAdapter`` to guard ``/metrics`` but
never runs ``start()``, so it must import the manifest's identity plugin itself or
token verification cannot resolve the configured provider. These pin: the helper is
a no-op when access control is off, short-circuits when the provider is already
registered, raises loudly when the manifest names no identity plugin, and — in a
FRESH interpreter, as the metrics process launches — imports the plugin so the
provider resolves.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from tai42_contract.access_control import registry

from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.cli import metrics


def test_disabled_access_control_is_a_noop() -> None:
    # AC off → the adapter adds no auth middleware, so no token is verified and no
    # provider is needed: the helper must not read the manifest or touch the registry.
    metrics._register_manifest_identity_provider(AccessControlSettings(enable=False))


def test_already_registered_short_circuits() -> None:
    # "redis" is registered by the suite's default-provider fixture; the helper
    # returns before reading any manifest (there is none in the test CWD).
    metrics._register_manifest_identity_provider(AccessControlSettings(enable=True, auth_providers=["redis"]))
    assert "redis" in registry._REGISTRY


def test_missing_identity_plugin_raises_loudly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # AC enabled but the manifest names NO identity plugin → /metrics cannot
    # authenticate → raise loudly (never silently un-authenticatable).
    manifest = tmp_path / "manifest.yml"
    manifest.write_text("lifecycle_modules: []\n", encoding="utf-8")
    monkeypatch.setenv("TAI_MANIFEST_PATH", str(manifest))
    registry.reset_registry()  # simulate a metrics process that has imported nothing

    with pytest.raises(RuntimeError, match="not registered"):
        metrics._register_manifest_identity_provider(AccessControlSettings(enable=True, auth_providers=["redis"]))


def test_metrics_process_resolves_redis_after_importing_plugin(tmp_path) -> None:
    # In a FRESH interpreter (as the metrics process launches, with the plugin not
    # yet imported), the helper reads the manifest, imports its identity plugin, and
    # the configured provider resolves — /metrics can verify tokens.
    manifest = tmp_path / "manifest.yml"
    manifest.write_text(
        "lifecycle_modules:\n  - tai42_identity_redis.redis_api_key_provider\n",
        encoding="utf-8",
    )
    child = textwrap.dedent(
        """
        from tai42_contract.access_control import registry
        from tai42_contract.access_control.registry import get_identity_provider_factory
        from tai42_skeleton.access_control.settings import AccessControlSettings
        from tai42_skeleton.cli import metrics

        # A fresh process has imported no identity plugin.
        assert "redis" not in registry._REGISTRY

        metrics._register_manifest_identity_provider(
            AccessControlSettings(enable=True, auth_providers=["redis"])
        )
        # The configured provider now resolves (would raise KeyError otherwise).
        get_identity_provider_factory("redis")
        print("resolved")
        """
    )
    env = {**os.environ, "TAI_MANIFEST_PATH": str(manifest)}
    result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "resolved" in result.stdout
