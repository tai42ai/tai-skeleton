"""Contract-compat verdicts for installed plugin distributions.

The one place the "does this installed plugin run against the running
``tai42-contract``?" question is answered. A plugin distribution declares its
supported contract line as an ordinary ``Requires-Dist: tai42-contract<range>``
in its packaging metadata, so the verdict needs no registry call and no stored
state — it is read from the installed dist via :mod:`importlib.metadata` at
evaluation time, which is what makes it correct even for a dist installed
BEFORE a core upgrade (the stale-plugin case this mechanism exists for).

Verdict semantics (:class:`CompatVerdict`):

* ``compatible`` — the dist declares a ``tai42-contract`` requirement and the
  running version satisfies it. ``prereleases=True`` everywhere, the same
  semantics as the installer's install-time contract check, so a dev-versioned
  editable contract checkout is never spuriously refused.
* ``incompatible`` — declared and NOT satisfied. The reason names both versions
  and the remedy, because it becomes the quarantine entry / abort message the
  operator acts on.
* ``unknown`` — no verdict is derivable: the module maps to no installed dist,
  the dist has no metadata, or it declares no ``tai42-contract`` requirement.
  Treated as compatible (a plugin that predates the declaration must keep
  loading), but the caller logs a note — never a silent pass.

A requirement line gated behind an extra (``; extra == "dev"``) is not part of
the runtime dependency set and is skipped; a malformed requirement line in a
dist's metadata cannot be interpreted and is skipped with a warning (only the
``tai42-contract`` line matters here, and refusing to boot over a stranger's
garbled metadata line would be the crash-loop this module removes).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from logging import getLogger
from typing import Any, Literal

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tai42_skeleton.marketplace.errors import LocalStateError, RegistryResponseError

logger = getLogger(__name__)

# The canonical dist name whose requirement line carries a plugin's supported
# contract range — the same dist whose running version is checked against it.
_CONTRACT_DIST = "tai42-contract"


class CorePluginBootError(RuntimeError):
    """A SCALAR-slot plugin (backend/storage/monitoring) is contract-incompatible
    or failed to import. Raised to ABORT boot: the server cannot run without its
    scalar slots, so they get a typed loud failure instead of a quarantine
    entry. The message names the plugin, the versions in play, and the remedy."""


@dataclass(frozen=True)
class CompatVerdict:
    """One dist's (or module's) verdict against the running contract. ``reason``
    is ``None`` only for ``compatible``; for ``incompatible`` it is the
    operator-facing quarantine/abort text, for ``unknown`` the note to log."""

    status: Literal["compatible", "incompatible", "unknown"]
    reason: str | None = None

    def as_payload(self) -> dict[str, str | None]:
        """The ``{status, reason}`` object the installed listing serves per row."""
        return {"status": self.status, "reason": self.reason}


def running_contract_version() -> str:
    """The installed ``tai42-contract`` version — the value every verdict here
    checks against and the ``contract`` param the resolve client sends."""
    return metadata.version(_CONTRACT_DIST)


def distribution_map() -> dict[str, list[str]]:
    """A snapshot of top-level package → owning distribution(s), for one boot
    pass to reuse across every manifest module (the underlying read walks every
    installed dist, so per-module fresh reads would be quadratic)."""
    return dict(metadata.packages_distributions())


def _declared_contract_specifier(dist_name: str) -> SpecifierSet | None:
    """The dist's declared ``tai42-contract`` specifier, or ``None`` when no
    verdict is derivable (dist not installed, no metadata, no declaration).

    Extra-gated requirement lines are skipped (not part of the runtime set);
    multiple runtime declarations AND together. A malformed line is skipped with
    a warning — see the module docstring for why that is not a boot failure.
    """
    try:
        requires = metadata.requires(dist_name)
    except metadata.PackageNotFoundError:
        return None
    if not requires:
        return None
    combined: SpecifierSet | None = None
    for line in requires:
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            logger.warning("distribution %s carries an unparsable Requires-Dist line %r; skipping it", dist_name, line)
            continue
        if canonicalize_name(requirement.name) != _CONTRACT_DIST:
            continue
        # ``extra == "..."`` markers never hold for the runtime dependency set;
        # the empty extra makes them evaluate False while the ordinary
        # environment markers (python_version, sys_platform) still apply.
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        combined = requirement.specifier if combined is None else combined & requirement.specifier
    return combined


def dist_compat(dist_name: str) -> CompatVerdict:
    """The verdict for one installed distribution, by its declared range."""
    specifier = _declared_contract_specifier(dist_name)
    if specifier is None:
        return CompatVerdict(
            "unknown",
            f"distribution {dist_name!r} declares no tai42-contract requirement; treating it as compatible",
        )
    installed = running_contract_version()
    if specifier.contains(installed, prereleases=True):
        return CompatVerdict("compatible")
    return CompatVerdict(
        "incompatible",
        f"{dist_name} requires tai42-contract {specifier}, but {installed} is running — "
        f"update the plugin to a release supporting {installed} (tai plugins update) or move the core "
        f"onto a contract the plugin supports",
    )


def module_compat(module: str, dist_map: dict[str, list[str]] | None = None) -> CompatVerdict:
    """The verdict for a manifest MODULE: its top-level package is mapped to the
    owning distribution(s) via ``importlib.metadata.packages_distributions`` and
    each mapped dist is evaluated — any incompatible dist decides the verdict
    (naming that dist), else compatible when at least one dist declared a range,
    else unknown.

    ``dist_map`` lets one boot pass snapshot the (sys.path-scan-priced) mapping
    once and reuse it across every manifest module; omitted, the mapping is read
    fresh.
    """
    top_level = module.partition(".")[0]
    mapping = metadata.packages_distributions() if dist_map is None else dist_map
    dists = mapping.get(top_level)
    if not dists:
        return CompatVerdict(
            "unknown",
            f"module {module!r} maps to no installed distribution; treating it as compatible",
        )
    verdicts = [dist_compat(dist_name) for dist_name in dict.fromkeys(dists)]
    for verdict in verdicts:
        if verdict.status == "incompatible":
            return verdict
    if any(verdict.status == "compatible" for verdict in verdicts):
        return CompatVerdict("compatible")
    return verdicts[0]


@dataclass(frozen=True)
class UpdateTargets:
    """The update picture for one installed ref, computed from the registry's
    version rows against the running contract: the newest published version
    (``latest``), the newest published COMPATIBLE version
    (``latest_compatible``), whether that compatible version beats the installed
    one (``update_available``), and the newest published INCOMPATIBLE version
    newer than the installed one (``incompatible_newer`` — the "an update
    exists, but it needs a newer core" signal), or ``None`` where no such
    version exists."""

    latest: str | None
    latest_compatible: str | None
    update_available: bool
    incompatible_newer: str | None


def update_targets(version_rows: list[Any], *, installed_version: str, contract_version: str) -> UpdateTargets:
    """Compute :class:`UpdateTargets` from the registry's version rows.

    Only ``status == "published"`` rows count. A row's ``contract_range`` gates
    compatibility with the SAME semantics as the dist verdict (``None`` — no
    declared range — counts compatible; ``prereleases=True``). Garbled registry
    data (a non-object row, a published row with a missing/non-PEP440 version,
    a non-string or malformed range) raises the typed registry-data fault so the
    boundary answers 502, never a silent skip; a garbled LOCAL installed version
    is corrupt local state, never blamed on the registry.
    """
    try:
        installed = Version(installed_version)
    except (InvalidVersion, TypeError) as exc:
        raise LocalStateError(f"the stored installed version {installed_version!r} is not PEP 440: {exc}") from exc

    latest: Version | None = None
    latest_compatible: Version | None = None
    incompatible_newer: Version | None = None
    for row in version_rows:
        if not isinstance(row, dict):
            raise RegistryResponseError("registry versions response contains a non-object element", status=None)
        if row.get("status") != "published":
            continue
        raw_version = row.get("version")
        try:
            version = Version(raw_version)  # pyright: ignore[reportArgumentType] — the except is the type guard
        except (InvalidVersion, TypeError) as exc:
            raise RegistryResponseError(
                f"registry served an unusable published version {raw_version!r}: {exc}", status=None
            ) from exc
        contract_range = row.get("contract_range")
        if contract_range is None:
            compatible = True
        elif isinstance(contract_range, str):
            try:
                compatible = SpecifierSet(contract_range).contains(contract_version, prereleases=True)
            except InvalidSpecifier as exc:
                raise RegistryResponseError(
                    f"registry served an unusable contract_range {contract_range!r} for version {version}: {exc}",
                    status=None,
                ) from exc
        else:
            raise RegistryResponseError(
                f"registry versions response contract_range for {version} is not a string", status=None
            )
        if latest is None or version > latest:
            latest = version
        if compatible and (latest_compatible is None or version > latest_compatible):
            latest_compatible = version
        if not compatible and version > installed and (incompatible_newer is None or version > incompatible_newer):
            incompatible_newer = version

    return UpdateTargets(
        latest=str(latest) if latest is not None else None,
        latest_compatible=str(latest_compatible) if latest_compatible is not None else None,
        update_available=latest_compatible is not None and latest_compatible > installed,
        incompatible_newer=str(incompatible_newer) if incompatible_newer is not None else None,
    )
