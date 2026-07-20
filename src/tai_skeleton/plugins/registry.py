"""Studio-plugin registry: discover installed plugins, validate their manifests,
and build the served-URL → sha384 integrity map the import map injects.

The registry is rebuilt by a pass registered on BOTH ``lifecycle.on_startup`` and
``lifecycle.on_reload`` (wired in ``app/instance.py`` alongside the connector
catalog refresh), so a reload that changes ``manifest.studio_plugins`` reflects
without a process restart. Data is read at call time from the live manifest and
the ``plugins_settings`` dist path — never captured at import time.

Every failure here is LOUD: a listed package missing its ``studio/`` dist or
``studio-manifest.json``, a manifest that violates the charset allow-lists, an
integrity filename that escapes the plugin's ``studio/`` root, or a required
shared-vendor asset missing from the deployed SPA — each raises, never a silent
skip. A silent skip would drop a plugin's UI or ship an unpinned asset while the
startup reported success.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.resources
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

logger = logging.getLogger(__name__)

# -- Validation charsets (injection-safe: every manifest-derived string that
#    reaches the PUBLIC index.html import map is constrained here at load) ------

# Python distribution/package name (the ``studio_plugins`` entries name importable
# packages). Deliberately strict — anything outside this rejects loudly.
_PKG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A single content-hashed asset filename or a ``/``-joined relative path of them.
# No ``..``, no leading ``/``, no backslashes — the traversal check re-verifies on
# the realpath, but rejecting here keeps a hostile string out of the served HTML.
_ASSET_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

# base64-encoded sha384 (48 bytes -> 64 base64 chars, optional ``=`` padding),
# prefixed ``sha384-`` as the Subresource-Integrity / import-map ``integrity``
# grammar requires.
_SHA384_RE = re.compile(r"^sha384-[A-Za-z0-9+/]{64}={0,2}$")

# The public asset route the browser actually requests for a plugin file. The
# import-map ``integrity`` keys MUST be these fully-resolved absolute URLs — a
# relative manifest-filename key silently fails to match and integrity does not
# apply.
_ASSET_URL_PREFIX = "/api/plugins/{name}/studio/"

# The excluded structured manifest file: never served publicly, checked on the
# realpath basename (a raw-string check is bypassed by ``sub/../studio-manifest.json``).
STUDIO_MANIFEST_FILENAME = "studio-manifest.json"

# Studio SDK subpaths reserved for the shell alone: the host registry API
# (``/host``) and the test harness (``/testing``). A plugin bundle must import
# only the plugin-facing ``@tai42/studio-sdk`` surface. The browser import map is
# shared by shell + plugins on one page, so it cannot hide these specifiers from
# plugin code; the enforceable gate is a static byte-scan for the literal
# specifier over every integrity-listed served file. Because the asset route
# serves ONLY integrity-listed files, every byte the browser can load from a
# plugin has been scanned. Accepted residual: a hostile author who builds the
# import specifier dynamically at runtime (e.g. string concatenation) evades a
# static byte-scan — that is out of scope under the trusted, reviewed-plugin
# model; runtime iframe/worker isolation is the future answer.
_FORBIDDEN_PLUGIN_SPECIFIERS: tuple[str, ...] = (
    "@tai42/studio-sdk/host",
    "@tai42/studio-sdk/testing",
)

# The shared-vendor ESM assets the host emits with STABLE un-hashed filenames.
# The import map keys these bare/subpath specifiers to the served files; every
# real React 19 plugin imports ``react/jsx-runtime`` (automatic JSX) and usually
# ``react-dom/client``, so the subpath entries are required, not optional.
# Served relative to the SPA dist root.
VENDOR_MODULES: dict[str, str] = {
    "react": "vendor/react.js",
    "react/jsx-runtime": "vendor/react-jsx-runtime.js",
    "react-dom": "vendor/react-dom.js",
    "react-dom/client": "vendor/react-dom-client.js",
    "@tai42/studio-sdk": "vendor/studio-sdk.js",
    "@tai42/studio-sdk/host": "vendor/studio-sdk-host.js",
}


def _is_valid_asset_path(value: str) -> bool:
    """Charset-valid AND no ``..`` segment. The charset permits dots (for
    ``index.min.js``), so ``..`` slips the regex — reject it explicitly here as
    defense in depth; the realpath check at load is the backstop."""
    return bool(_ASSET_PATH_RE.match(value)) and ".." not in value.split("/")


class StudioPluginError(RuntimeError):
    """A listed Studio plugin is missing, malformed, or unsafe. Raised at
    startup/reload so a broken deployment fails loudly instead of silently
    dropping a plugin's UI."""


class Contributions(BaseModel):
    """What a Studio plugin contributes to the shell. ``tool_panels`` is keyed by
    the tool name whose run panel the plugin replaces; a tool with no entry here
    falls back to the auto-form. ``nav_entries`` declares the ``pages`` the plugin
    surfaces a nav entry for — the manifest-declared form of the shell's runtime
    ``registerNavEntry``, so the catalog/manifest read-surface reports a
    nav-contributing plugin."""

    model_config = ConfigDict(extra="forbid")

    tool_panels: dict[str, str] = {}
    pages: list[str] = []
    settings_tabs: list[str] = []
    nav_entries: list[str] = []

    @model_validator(mode="after")
    def _validate_nav_entries(self) -> Contributions:
        # Every nav entry must link to a page the plugin also contributes — the
        # shell's own "a nav entry must reference a declared page" rule. A dangling
        # entry is a loud rejection at manifest load, not a nav item that renders a
        # broken route.
        dangling = sorted(entry for entry in self.nav_entries if entry not in self.pages)
        if dangling:
            raise ValueError(f"nav_entries {dangling!r} do not appear in the plugin's pages")
        return self


class StudioPluginManifest(BaseModel):
    """The parsed, validated ``studio-manifest.json``. ``extra="forbid"`` so an
    unknown field is a loud rejection, not a silently-ignored typo."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    api_version: int
    entry: str
    # Maps every emitted asset filename (the entry bundle AND every lazy chunk)
    # to its ``sha384-<base64>`` hash. Single-file bundles make this one entry.
    integrity: dict[str, str]
    contributions: Contributions

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _PKG_NAME_RE.match(value):
            raise ValueError(f"studio plugin name {value!r} is not a valid package name")
        return value

    @field_validator("entry")
    @classmethod
    def _validate_entry(cls, value: str) -> str:
        if not _is_valid_asset_path(value):
            raise ValueError(f"studio plugin entry {value!r} is not a valid asset path")
        return value

    @field_validator("integrity")
    @classmethod
    def _validate_integrity(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("studio plugin integrity map must not be empty")
        for filename, digest in value.items():
            if not _is_valid_asset_path(filename):
                raise ValueError(f"studio plugin integrity filename {filename!r} is not a valid asset path")
            if not _SHA384_RE.match(digest):
                raise ValueError(f"studio plugin integrity value for {filename!r} is not a sha384-<base64> hash")
        return value


class InstalledStudioPlugin(BaseModel):
    """A validated, installed Studio plugin: its parsed manifest, the realpath of
    its ``studio/`` dist root, and the served-URL → sha384 integrity entries the
    import map injects."""

    model_config = ConfigDict(frozen=True)

    package: str
    dist_root: Path
    manifest: StudioPluginManifest
    # Absolute served URL -> ``sha384-<base64>``. Keys are the exact URLs the
    # browser requests (``/api/plugins/{name}/studio/<file>``).
    integrity_by_url: dict[str, str]


class StudioPluginRegistry(BaseModel):
    """The built registry both the authed registry route and the SPA import-map
    injection read (once per request, no per-request re-walk)."""

    model_config = ConfigDict(frozen=True)

    plugins: dict[str, InstalledStudioPlugin] = {}
    # Vendor served-URL -> sha384, sourced by hashing the deployed SPA's
    # ``vendor/`` dir. Empty when no SPA dist is configured.
    vendor_integrity_by_url: dict[str, str] = {}

    def manifest_contents(self) -> list[dict]:
        """The authed registry listing payload: each plugin's parsed manifest."""
        return [p.manifest.model_dump() for p in self.plugins.values()]

    def import_map(self) -> dict:
        """The import map injected into ``index.html``: bare/subpath vendor
        specifiers plus the ``integrity`` block covering every vendor asset AND
        every plugin bundle/chunk, keyed by fully-resolved absolute URL."""
        imports = {spec: f"/{rel}" for spec, rel in VENDOR_MODULES.items()}
        integrity = dict(self.vendor_integrity_by_url)
        for plugin in self.plugins.values():
            integrity.update(plugin.integrity_by_url)
        return {"imports": imports, "integrity": integrity}


def _hash_bytes(data: bytes) -> str:
    return "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _reject_forbidden_specifiers(package: str, filename: str, data: bytes) -> None:
    """Reject a plugin file whose bytes contain a shell-only Studio SDK subpath
    specifier. Called for every integrity-listed file during load; combined with
    the asset route serving only integrity-listed files, this scans the exact
    bytes the browser can load. The check is a literal byte match, so a
    dynamically-built specifier is out of scope (see the module comment on
    ``_FORBIDDEN_PLUGIN_SPECIFIERS``)."""
    for specifier in _FORBIDDEN_PLUGIN_SPECIFIERS:
        if specifier.encode("ascii") in data:
            raise StudioPluginError(
                f"studio plugin {package!r} file {filename!r} imports host-only Studio SDK "
                f"specifier {specifier!r} — plugin bundles must import only ``@tai42/studio-sdk``"
            )


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` (realpath, following symlinks) and
    verify it stays inside — raise otherwise. The single traversal primitive."""
    root_real = root.resolve()
    target = (root_real / relative).resolve()
    if root_real != target and root_real not in target.parents:
        raise StudioPluginError(f"path {relative!r} escapes the studio dist root {root_real}")
    return target


def _studio_root(package: str) -> Path:
    """The on-disk ``studio/`` dist root of an installed plugin package."""
    if not _PKG_NAME_RE.match(package):
        raise StudioPluginError(f"studio plugin package name {package!r} is invalid")
    try:
        base = importlib.resources.files(package)
    except ModuleNotFoundError as exc:
        raise StudioPluginError(f"studio plugin package {package!r} is not importable") from exc
    root = Path(str(base)) / "studio"
    if not root.is_dir():
        raise StudioPluginError(f"studio plugin package {package!r} has no ``studio/`` dist directory at {root}")
    return root


def _load_plugin(package: str) -> InstalledStudioPlugin:
    import json

    root = _studio_root(package)
    manifest_path = root / STUDIO_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise StudioPluginError(f"studio plugin {package!r} is missing {STUDIO_MANIFEST_FILENAME} at {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StudioPluginError(f"studio plugin {package!r} manifest is not readable JSON: {exc}") from exc
    try:
        manifest = StudioPluginManifest.model_validate(raw)
    except ValueError as exc:
        raise StudioPluginError(f"studio plugin {package!r} manifest is invalid: {exc}") from exc

    # The registry keys plugins and integrity URLs by the Python ``package``, but
    # the authed listing and the shell's bundle URL are built from ``manifest.name``.
    # A mismatch would pass startup validation, then 404 on every browser load —
    # reject it here so the failure is loud at load, not late and per-user.
    if manifest.name != package:
        raise StudioPluginError(
            f"studio plugin {package!r} declares manifest name {manifest.name!r}; "
            "the manifest name must equal the package name or its bundle URL can never load"
        )

    # Every integrity filename must resolve to a real file under the studio root,
    # and the entry must be one of them.
    if manifest.entry not in manifest.integrity:
        raise StudioPluginError(f"studio plugin {package!r} entry {manifest.entry!r} has no integrity hash")
    integrity_by_url: dict[str, str] = {}
    url_prefix = _ASSET_URL_PREFIX.format(name=package)
    for filename, declared in manifest.integrity.items():
        resolved = resolve_under(root, filename)
        if not resolved.is_file():
            raise StudioPluginError(f"studio plugin {package!r} lists {filename!r} but no such file exists")
        data = resolved.read_bytes()
        _reject_forbidden_specifiers(package, filename, data)
        actual = _hash_bytes(data)
        if actual != declared:
            raise StudioPluginError(
                f"studio plugin {package!r} file {filename!r} sha384 mismatch: "
                f"manifest declares {declared}, file hashes to {actual}"
            )
        integrity_by_url[f"{url_prefix}{filename}"] = declared

    return InstalledStudioPlugin(
        package=package,
        dist_root=root.resolve(),
        manifest=manifest,
        integrity_by_url=integrity_by_url,
    )


def _vendor_integrity(dist_path: str | None) -> dict[str, str]:
    """Hash every required shared-vendor asset under the deployed SPA's
    ``vendor/`` dir. A missing required asset is loud — the app cannot boot
    without it, so a silent omission would ship an unresolvable import map."""
    if dist_path is None:
        return {}
    dist_root = Path(dist_path)
    result: dict[str, str] = {}
    for rel in VENDOR_MODULES.values():
        asset = resolve_under(dist_root, rel)
        if not asset.is_file():
            raise StudioPluginError(
                f"shared-vendor asset {rel!r} is missing from the SPA dist at {dist_root} — "
                "the import map cannot resolve React/the SDK without it"
            )
        result[f"/{rel}"] = _hash_file(asset)
    return result


def build_registry(studio_plugins: list[str], dist_path: str | None) -> StudioPluginRegistry:
    """Validate every listed plugin and hash the vendor assets. Loud on any
    failure — never returns a partial registry."""
    plugins: dict[str, InstalledStudioPlugin] = {}
    for package in studio_plugins:
        if package in plugins:
            raise StudioPluginError(f"studio plugin {package!r} is listed more than once in ``studio_plugins``")
        plugins[package] = _load_plugin(package)
    return StudioPluginRegistry(
        plugins=plugins,
        vendor_integrity_by_url=_vendor_integrity(dist_path),
    )


# -- Process-global current registry, swapped atomically by the build pass -----

_current: StudioPluginRegistry | None = None


def set_current_registry(registry: StudioPluginRegistry) -> None:
    global _current
    _current = registry


def current_registry() -> StudioPluginRegistry:
    """The registry built by the last startup/reload pass. Raises if the pass has
    not run (the app is not started) — never returns a silent empty registry that
    would hide a boot-order bug."""
    if _current is None:
        raise StudioPluginError("studio plugin registry has not been built — is the app started?")
    return _current


async def rebuild_studio_plugin_registry() -> None:
    """Startup/reload handler: rebuild the registry from the LIVE manifest's
    ``studio_plugins`` and the configured SPA dist path. Registered on both
    ``lifecycle.on_startup`` and ``lifecycle.on_reload`` so a reload that changes
    ``studio_plugins`` reflects without a process restart. Reads at call time via
    the live-manifest seam — never captured at import.
    """
    from tai_contract.app import tai_app

    from tai_skeleton.plugins.settings import plugins_settings

    live_manifest = tai_app.admin.live_manifest
    studio_plugins = live_manifest.get("studio_plugins", [])
    dist_path = plugins_settings().studio_dist_path
    registry = build_registry(studio_plugins, dist_path)
    set_current_registry(registry)
    logger.info(
        "studio plugin registry built: %d plugin(s), %d vendor asset(s)",
        len(registry.plugins),
        len(registry.vendor_integrity_by_url),
    )
