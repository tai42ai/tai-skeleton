"""Settings for the Studio-plugin SPA host: where the built Studio app lives.

Follows the ``metrics_settings`` precedent — a ``TaiBaseSettings`` subclass plus a
``@settings_cache`` accessor, so a reload re-reads it and tests monkeypatch the
accessor. The dist path is a per-deployment/env concern (like the metrics ports),
NOT manifest content: it is deliberately NOT the contract Manifest's ``static_dir``
field, so the two static-dir concepts stay separate.
"""

from tai42_kit.settings import TaiBaseSettings, settings_cache


class PluginsSettings(TaiBaseSettings):
    # Absolute (or CWD-relative) path to the built Studio SPA dist directory —
    # the ``index.html``, the hashed asset bundles, the stable ``vendor/`` ESM
    # assets, and the static OAuth pages. Unset (None) means this deployment does
    # not host the Studio UI: the SPA catch-all route is always registered (this
    # setting is reload-mutable, so registration cannot depend on it) and answers
    # every request with 404 while the path is unset.
    studio_dist_path: str | None = None


@settings_cache
def plugins_settings() -> PluginsSettings:
    return PluginsSettings()
