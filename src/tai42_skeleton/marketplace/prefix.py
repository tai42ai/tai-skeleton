"""The plugin prefix: a persistent install root for marketplace plugin
distributions, kept apart from the platform's own environment packages (the
wp-content principle — plugin files live separate from the platform's packages).

``TAI_PLUGINS_PREFIX`` unset -> installs go into the running interpreter
environment, which a container image discards on restart (today's behavior).
Set -> the deployment OPTS IN to persistence: the installer installs each plugin's
OWN distribution into the prefix while pip resolves shared dependencies against the
running environment, so a package already importable in the environment is never
copied into the prefix (only genuinely-new dependencies land there), and boot adds
the prefix's site directories to the END of ``sys.path`` so the environment shadows
the prefix for any package present in both.

The dependency rule is pip's own: ``pip install --prefix`` resolves against the
running environment and installs only the distributions that environment does not
already satisfy, so the image's ``tai42-contract`` / ``tai42-kit`` / … always win
and are never duplicated in the prefix. Uninstall does NOT shell pip — pip refuses
to remove a distribution installed outside its own environment — so it removes the
plugin's own files from the prefix directly, by the ``RECORD`` its ``.dist-info``
ships, leaving every dependency in place (the same "only the named distribution is
removed" boundary the environment path has).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings, settings_cache

from tai42_skeleton.marketplace.errors import PluginPrefixError

logger = logging.getLogger(__name__)


class PluginPrefixSettings(TaiBaseSettings):
    """``TAI_PLUGINS_*`` — where marketplace plugin distributions install.

    The manifest location is already env-configurable (``TAI_MANIFEST_PATH`` on
    ``CoreSettings``), so it is not restated here.
    """

    model_config = SettingsConfigDict(env_prefix="TAI_PLUGINS_")

    # Absolute path to the persistent plugin-install prefix. Unset (None) keeps
    # today's behavior: installs land in the running environment. Set opts into
    # restart-survival — plugins install here and boot puts it on ``sys.path``.
    prefix: str | None = None


@settings_cache
def plugin_prefix_settings() -> PluginPrefixSettings:
    return PluginPrefixSettings()


def configured_prefix() -> str | None:
    """The configured plugin prefix, or ``None`` when persistence is not opted in."""
    return plugin_prefix_settings().prefix


def prefix_site_dirs(prefix: str) -> list[str]:
    """The site-package directories a ``pip install --prefix <prefix>`` populates —
    the prefix scheme's ``purelib`` and ``platlib`` (identical on posix). These are
    the exact directories boot adds to ``sys.path`` and uninstall scans."""
    paths = sysconfig.get_paths(vars={"base": prefix, "platbase": prefix})
    dirs: list[str] = []
    for key in ("purelib", "platlib"):
        directory = paths[key]
        if directory not in dirs:
            dirs.append(directory)
    return dirs


def ensure_prefix_writable(prefix: str) -> None:
    """Create the prefix's site directories if missing and confirm the prefix is
    writable, before an install touches it.

    A set-but-uncreatable/unwritable prefix is a loud deployment fault
    (:class:`PluginPrefixError`), never a silent fall back to installing into the
    environment.
    """
    root = Path(prefix)
    try:
        for site in prefix_site_dirs(prefix):
            Path(site).mkdir(parents=True, exist_ok=True)
        probe = root / ".tai42-prefix-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PluginPrefixError(f"plugin prefix {prefix!r} is set but not writable: {exc}") from exc


def activate_prefix() -> None:
    """Put the configured prefix on ``sys.path`` so its plugin distributions import.

    No-op when the prefix is unset. The site directories go at the END of
    ``sys.path`` so the environment shadows the prefix for any package present in
    both. Idempotent — safe on every reload. Runs before the manifest modules
    import at boot.

    An existing directory (even read-only, a prefix of pre-installed plugins on a
    read-only mount) is a no-op to ``mkdir``, so it activates fine. A MISSING prefix
    whose parent is not writable raises loudly here — a set-but-broken prefix is a
    misconfiguration, not a silent empty import path.
    """
    prefix = configured_prefix()
    if prefix is None:
        return
    changed = False
    for site in prefix_site_dirs(prefix):
        Path(site).mkdir(parents=True, exist_ok=True)
        if site not in sys.path:
            sys.path.append(site)
            changed = True
    if changed:
        importlib.invalidate_caches()
        logger.info("plugin prefix activated on sys.path: %s", prefix)


def prefix_has_distribution(package: str, prefix: str) -> bool:
    """Whether ``package``'s own distribution is installed under ``prefix`` — its
    prefix site dirs ONLY, never an environment copy that also sits on ``sys.path``
    at runtime."""
    site_dirs = [str(Path(site)) for site in prefix_site_dirs(prefix)]
    target = canonicalize_name(package)
    return any(canonicalize_name(dist.name) == target for dist in importlib.metadata.distributions(path=site_dirs))


def environment_distribution_version(package: str, prefix: str) -> str | None:
    """``package``'s version as installed in the running ENVIRONMENT — every
    ``sys.path`` entry that is NOT a prefix site dir — or ``None`` when the
    environment does not provide it.

    Both the environment and the prefix sit on ``sys.path`` at runtime (the prefix
    appended at the END), so the prefix site dirs are excluded here to read the
    environment copy alone. The first match wins, mirroring import shadowing order.
    """
    prefix_dirs = {str(Path(site)) for site in prefix_site_dirs(prefix)}
    env_paths = [p for p in sys.path if p not in prefix_dirs]
    target = canonicalize_name(package)
    for dist in importlib.metadata.distributions(path=env_paths):
        if canonicalize_name(dist.name) == target:
            return dist.version
    return None


def uninstall_from_prefix(package: str, prefix: str) -> None:
    """Remove ``package``'s own distribution from ``prefix``, by its ``RECORD``.

    pip cannot do this — it refuses to uninstall a distribution installed outside
    its own environment — so the plugin's files are removed directly: every path
    the distribution's ``RECORD`` lists (its package tree, its ``.dist-info``, any
    console script under the prefix's ``bin``) is unlinked and the emptied
    directories are pruned. Only the NAMED distribution is removed; its
    dependencies stay, exactly as ``pip uninstall`` leaves them on the environment
    path.

    A package absent from the prefix raises :class:`PluginPrefixError` rather than
    reaching for the environment copy — with a prefix configured, uninstall must
    never mutate the image. A ``RECORD`` path that escapes the prefix, or a
    distribution shipping no ``RECORD``, is likewise a loud refusal, never a
    best-effort partial delete.
    """
    root = Path(prefix).resolve()
    site_dirs = [str(Path(site)) for site in prefix_site_dirs(prefix)]
    target = canonicalize_name(package)
    match = [
        dist for dist in importlib.metadata.distributions(path=site_dirs) if canonicalize_name(dist.name) == target
    ]
    if not match:
        raise PluginPrefixError(
            f"{package!r} is not installed in the plugin prefix {prefix} — refusing to touch the environment"
        )
    dist = match[0]
    files = dist.files
    if files is None:
        raise PluginPrefixError(f"{package!r} in the plugin prefix has no RECORD; refusing a best-effort delete")
    parents: set[Path] = set()
    for record_file in files:
        path = Path(str(dist.locate_file(record_file))).resolve()
        if root != path and root not in path.parents:
            raise PluginPrefixError(
                f"{package!r} RECORD path {path} escapes the plugin prefix {root}; refusing to delete outside it"
            )
        if path.is_symlink() or path.exists():
            path.unlink()
        parents.add(path.parent)
    for directory in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        _prune_empty(directory, root)


def _prune_empty(directory: Path, root: Path) -> None:
    """Remove ``directory`` and its now-empty ancestors up to (not including)
    ``root``. Stops at the first non-empty directory."""
    current = directory
    while current != root and root in current.parents:
        if current.is_dir() and not any(current.iterdir()):
            current.rmdir()
            current = current.parent
        else:
            break
