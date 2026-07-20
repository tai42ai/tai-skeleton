"""The marketplace installer — resolve, install, patch, reload, attribute.

:class:`Installer` performs the three environment-mutating flows the marketplace
surface exposes — ``install``, ``uninstall``, ``update`` — each as an ordered
sequence with abort-on-any-step and an explicit reverse unwind, so a mid-flight
failure never leaves the app half-registered.

Every collaborator is injected (the registry client, the pip runner, the
attribution store, the config-mutation pipeline, the fleet lock, and the config
manager), so a test can fake each seam. The defaults are the real ones.

The one manifest write door
---------------------------
Every manifest edit crosses :class:`~tai_skeleton.config.service.ConfigService`:
:meth:`~tai_skeleton.config.service.ConfigService.apply_change` takes a mutator
that patches the manifest in place, and the pipeline validates the RESOLVED
projection of the compose (``!ENV`` markers materialized, so a marker on a
non-string field validates against its resolved value) — a compose the resolved
schema or the backend-needs-bus invariant rejects raises loudly inside the
transaction instead of corrupting the stored manifest, and the marketplace maps
that fault to a typed :class:`~tai_skeleton.marketplace.errors.ManifestComposeError`
(see :meth:`_apply_composed`). The pipeline owns the transaction, the persist, the
local reload through the process reload gate, and the confirmed fleet broadcast on
the worker bus. The reload re-reads the
persisted manifest and re-imports every manifest-named module, so a package
pip-installed moments earlier is importable on the very next reload with no
process restart. An unwind restores the pre-change manifest through the same
pipeline (:meth:`~tai_skeleton.config.service.ConfigService.apply_replace`), so a
rollback reload+broadcast reaches the fleet too.

Concurrency
-----------
Two layers, each honest about what it covers:

- The fleet-wide **PostgreSQL advisory lock** (:func:`_fleet_lock`) is the
  correctness layer for the VENV: it serializes marketplace operations across
  every worker PROCESS, so two pips never mutate one venv at once. It is held
  across the ENTIRE operation (pip run, manifest apply, attribution write). The
  manifest change's own cross-process atomicity is owned by the pipeline's
  transaction, not this lock.
- The per-worker :data:`_operation_lock` is the UX fast path: a second request
  arriving in the SAME worker while an operation runs is refused immediately with
  a retriable :class:`OperationInProgressError` instead of queueing for minutes.
  It is a fast path only — never the mutual-exclusion story, since each worker is
  a separate process with its own copy of it.

The honest boundary: the fleet lock serializes MARKETPLACE operations against
each other so no two overlap a single venv; the manifest pipeline serializes the
manifest write itself across the whole fleet.

The pip transaction boundary
----------------------------
An unwind (or an uninstall) fully reverts the manifest, the attribution row, and
the live registration, but the venv is only as transactional as pip itself:
``pip uninstall`` removes just the named distribution, so dependencies pip
installed or upgraded in place during the attempt remain. Only skeleton state
reverts completely; that caveat rides the install/update response text.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import logging
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import ValidationError
from tai_contract.app import tai_app
from tai_contract.plugins import KIND_MANIFEST_BINDINGS, PluginItem, PluginSpec
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import PostgresClient

from tai_skeleton.app.boot_rules import BackendNeedsBusError
from tai_skeleton.config.service import ApplyResult, ConfigService
from tai_skeleton.marketplace.client import RegistryClient
from tai_skeleton.marketplace.errors import (
    ContractIncompatibleError,
    InstallStateError,
    InstallUnwindError,
    LocalStateError,
    MalformedRefError,
    ManifestCollisionError,
    ManifestComposeError,
    OperationInProgressError,
    RegistryResponseError,
    VersionRefusedError,
)
from tai_skeleton.marketplace.manifest_patch import apply_provides, collisions, remove_provides
from tai_skeleton.marketplace.pip import (
    PipRunner,
    ensure_pip_available,
    fetch_verified_artifact,
    install_args,
    run_pip,
    uninstall_args,
)
from tai_skeleton.marketplace.settings import marketplace_store_settings
from tai_skeleton.marketplace.store import InstallRecord, MarketplaceInstallStore
from tai_skeleton.operations._broadcast import FleetBroadcastError, fleet_fanout

logger = logging.getLogger(__name__)

# Per-worker fast-path lock: a second same-worker operation is refused
# immediately rather than queued. NOT the correctness layer — each uvicorn worker
# is a separate process with its own instance of this.
_operation_lock = asyncio.Lock()

# Fixed key for the fleet-wide session advisory lock that serializes marketplace
# operations. Session- and transaction-scoped advisory locks share ONE key space
# in PostgreSQL, and this feature's DSN commonly targets the same database as the
# connector store (which takes ``0x7461695F636F6E6E`` for category creation), so
# this MUST be a distinct value or the two would block against each other across
# the fleet. "tai_mktp" as ASCII bytes; the high byte 0x74 keeps it a positive
# bigint.
_MARKETPLACE_LOCK_KEY = 0x7461695F6D6B7470  # "tai_mktp"


def _reload_report(result: ApplyResult) -> dict[str, Any]:
    """The manifest apply's local reload result with the standard fleet fan-out
    summary folded in under ``fanout`` — the ``reload`` field every
    install/uninstall/update response carries. Reuses the shared
    :func:`~tai_skeleton.operations._broadcast.fleet_fanout` shaper so the key AND the
    value shape match every other fleet-report-embedding writer (backup, the
    operations ``apply_response``)."""
    return {**result.local, "fanout": fleet_fanout(result.fleet)}


@asynccontextmanager
async def _fleet_lock() -> AsyncIterator[None]:
    """Hold the fleet-wide marketplace advisory lock for the context body.

    Opens a DEDICATED one-shot ``PostgresClient`` (``fresh=True``, pool bounds
    pinned to 1) rather than a shared-pool checkout: the kit's fresh path closes
    the dedicated pool and its connection on ANY exit — normal return, exception,
    or a ``CancelledError`` from a client disconnect during the minutes-long pip
    run — so the session-scoped lock releases deterministically with the
    connection. A shared checkout would only RETURN the connection on exit (never
    close it), and a cancellation skipping the explicit unlock would put a
    still-locked session back in the shared pool, wedging the fleet with 503s
    until restart.

    The connection is set to autocommit BEFORE the try-lock: psycopg defaults
    autocommit off, so otherwise the lock SELECT would open a transaction idling
    for the whole pip run, and a managed-PG ``idle_in_transaction_session_timeout``
    would kill the session mid-install and silently drop the lock. Autocommit
    closes that hole but NOT the idle-SESSION one: a managed-PG
    ``idle_session_timeout`` or a load-balancer idle drop can still kill the whole
    session mid-pip and release the lock while pip is still mutating the venv — an
    ACCEPTED residual risk whose mitigation is to disable ``idle_session_timeout``
    (and keep TCP keepalives on) for this DSN so a long pip run never trips a
    cutoff.

    ``pg_try_advisory_lock`` returning false means another worker holds it →
    :class:`OperationInProgressError` (retriable 503). The ``pg_advisory_unlock``
    in the ``finally`` is the polite release; the connection close is the
    guarantee, so its failure on a possibly-dead connection is logged and
    suppressed, never allowed to mask the body's original error. A crashed or
    cancelled worker therefore cannot leave the fleet wedged.

    Side benefit of the dedicated client: a psycopg ``OperationalError`` raised
    inside the body stays on this client and cannot evict the SHARED pool through
    the kit's disconnection rewrap.
    """
    kwargs = marketplace_store_settings().client_kwargs()
    kwargs["min_size"] = 1
    kwargs["max_size"] = 1
    async with (
        client_ctx(PostgresClient, fresh=True, **kwargs) as pool,
        pool.connection() as conn,
    ):
        await conn.set_autocommit(True)
        async with conn.cursor() as cur:
            await cur.execute("SELECT pg_try_advisory_lock(%s)", (_MARKETPLACE_LOCK_KEY,))
            row = await cur.fetchone()
        if not (row and row[0]):
            raise OperationInProgressError("another marketplace operation is in progress; retry shortly")
        try:
            yield
        finally:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT pg_advisory_unlock(%s)", (_MARKETPLACE_LOCK_KEY,))
            except Exception:
                logger.warning(
                    "marketplace: advisory unlock failed; the connection close releases the lock", exc_info=True
                )


class Installer:
    """Install, uninstall, and update marketplace plugins with abort-and-unwind.

    Each public method acquires the per-worker fast-path lock and then the
    fleet-wide advisory lock before touching any state, and holds both across the
    whole operation, so a lock-held-elsewhere refusal makes no store, registry,
    pip, or manifest call.
    """

    def __init__(
        self,
        *,
        registry: RegistryClient | None = None,
        pip_runner: PipRunner = run_pip,
        store: MarketplaceInstallStore | None = None,
        config_service: ConfigService | None = None,
        fleet_lock: Callable[[], AbstractAsyncContextManager[None]] = _fleet_lock,
        config_manager: Any | None = None,
    ) -> None:
        self._registry = registry or RegistryClient()
        self._pip_runner = pip_runner
        self._store = store or MarketplaceInstallStore()
        self._config_service = config_service
        self._fleet_lock = fleet_lock
        self._config_manager = config_manager

    # -- infrastructure -----------------------------------------------------

    def _cm(self) -> Any:
        """The live config manager (read seam), or the injected fake.

        Reached live per use because a reload swaps app internals; never cached
        across a reload — unless a fake was injected, in which case it is used
        throughout. Used only for the pre-flight manifest READS; every WRITE crosses
        :meth:`_svc`.
        """
        if self._config_manager is not None:
            return self._config_manager
        return tai_app.config.config_manager

    def _svc(self) -> ConfigService:
        """The config-mutation pipeline, or the injected fake.

        Wired live per use (:meth:`ConfigService.from_app`) because a reload swaps
        app internals; every manifest write — forward and unwind — crosses it, so a
        write is always validated, persisted, locally reloaded, and broadcast.
        """
        if self._config_service is not None:
            return self._config_service
        return ConfigService.from_app()

    @asynccontextmanager
    async def _guard(self) -> AsyncIterator[None]:
        """Acquire the per-worker lock then the fleet lock, both for the body.

        Refuses immediately (no store/registry/pip/manifest call) when either
        lock is held elsewhere.
        """
        if _operation_lock.locked():
            raise OperationInProgressError("another marketplace operation is in progress; retry shortly")
        async with _operation_lock, self._fleet_lock():
            yield

    async def _apply_composed(self, mutator: Callable[[dict[str, Any]], None]) -> ApplyResult:
        """Apply a manifest mutation through the pipeline, mapping a composed-manifest
        fault to the typed compose error.

        The pipeline validates the RESOLVED projection of the mutated document (``!ENV``
        markers materialized) before it persists, so a marker on a non-string field
        (e.g. ``api_tools.expose_destructive``) validates against its resolved value —
        the marketplace does no schema validation of its own. Because the marketplace
        fully controls the composed document, a resolved-projection schema failure
        (:class:`~pydantic.ValidationError`) OR a registered-backend-without-a-bus refusal
        (:class:`~tai_skeleton.app.boot_rules.BackendNeedsBusError`, which the whole-fleet
        boundary would otherwise let escape untyped) is a registry-spec + local fault,
        re-raised as :class:`ManifestComposeError` so the boundary attributes it a loud
        500 rather than a bare one. Either raises inside the transaction, so nothing
        persists. A resolved-secret leak (a ``ResolvedSecretError``) cannot arise here:
        the structural provides patch ingests no resolved secret value, so the pipeline's
        secret seal is a no-op."""
        try:
            return await self._svc().apply_change(mutator)
        except (ValidationError, BackendNeedsBusError) as exc:
            raise ManifestComposeError(f"the composed manifest is invalid: {exc}") from exc

    async def _pip_install(
        self, package: str, version: str, source: str, artifact_ref: str | None, sha256: str | None
    ) -> str:
        """Run the ``pip install`` for a pinned version and return its output.

        A pypi source pins ``package==version`` directly. A github source first
        downloads the registry-named artifact and verifies its sha256 through
        :func:`fetch_verified_artifact` into a temporary directory kept alive
        across the whole pip run (pip must read the local tarball before it is
        cleaned up), then installs that verified tarball — never a mutable
        ``git+url@tag`` clone. A checksum mismatch or fetch failure raises out of
        here with no pip call and no fallback.
        """
        if source == "github":
            with tempfile.TemporaryDirectory() as tmp:
                verified = await fetch_verified_artifact(package, version, artifact_ref or "", sha256 or "", Path(tmp))
                return await self._pip_runner(install_args(package, version, source, verified))
        return await self._pip_runner(install_args(package, version, source))

    # -- install ------------------------------------------------------------

    async def install(self, ref: str, version: str | None = None) -> dict[str, Any]:
        """Install a marketplace plugin, aborting and unwinding on any failure.

        Steps, in order: parse the ref and pre-flight local state (not already
        installed, pip present); resolve the pinned version via the registry (the
        ONE pinning call — critical-advisory refusal, spec validation, and
        contract-range compatibility included); collision pre-flight against the
        manifest BEFORE pip; ``pip install`` the locally-composed pin; patch the
        manifest through the one door and reload; write the attribution row.

        Each step past the pip install unwinds the applied steps in reverse on
        failure (manifest restored, live app converged back, pip uninstall) and
        re-raises; a failed unwind escalates to :class:`InstallUnwindError`. Only
        skeleton state fully reverts — see the pip-transaction caveat.
        """
        async with self._guard():
            return await self._install_locked(ref, version)

    async def _install_locked(self, ref: str, version: str | None) -> dict[str, Any]:
        ns, name = _parse_ref(ref)
        existing = await self._store.get(ref)
        if existing is not None:
            raise InstallStateError(f"{ref} is already installed (version {existing.version}); use update")
        ensure_pip_available()

        resolved = await self._registry.resolve(ns, name, version)
        spec, source = self._prepare_resolved(resolved)
        pinned_version = _require(resolved, "version")

        cm = self._cm()
        found = collisions(dict(cm.read_manifest_preserved()), spec)
        if found:
            raise ManifestCollisionError("; ".join(found))

        # Step 3 — pip install (github: fetch + verify + install the local
        # tarball). A failure propagates as-is: the venv may hold a
        # partially-resolved state pip itself left, and pip's own transaction
        # handling is the boundary.
        pip_output = await self._pip_install(
            spec.package, pinned_version, source, resolved.get("artifact_ref"), resolved.get("sha256")
        )

        saved_manifest = copy.deepcopy(cm.read_manifest_preserved())
        manifest_persisted = False
        try:
            # Step 4 — manifest patch through the pipeline: the mutator applies the
            # provides, the pipeline validates the resolved compose (a compose the
            # resolved schema or the backend-needs-bus invariant rejects raises inside
            # the transaction so nothing persists — mapped to the typed compose error),
            # persists, reloads locally, and broadcasts.
            def mutator(document: dict[str, Any]) -> None:
                apply_provides(document, spec)

            apply_result = await self._apply_composed(mutator)
            manifest_persisted = True
            # Step 5 — attribution write.
            repo_url, tag, artifact_ref, sha256 = _pin_provenance(resolved, source)
            await self._store.record(
                ref, pinned_version, source, repo_url, tag, artifact_ref, sha256, spec.model_dump(mode="json")
            )
        except Exception as step_error:
            # The pipeline aborts a failed mutation with nothing persisted, so the
            # manifest needs restoring only when the change actually landed — the
            # attribution step failed after a good apply, or the apply's local reload
            # failed AFTER its persist committed (:class:`FleetBroadcastError`).
            persisted = manifest_persisted or isinstance(step_error, FleetBroadcastError)
            await self._unwind_install(
                step_error, package=spec.package, saved_manifest=saved_manifest if persisted else None
            )
            raise

        return {
            "ref": ref,
            "version": pinned_version,
            "package": spec.package,
            "advisories": _require_list(resolved, "advisories"),
            "notes": _install_notes(spec),
            "reload": _reload_report(apply_result),
            "pip_output": pip_output,
        }

    async def _unwind_install(
        self, step_error: Exception, *, package: str, saved_manifest: dict[str, Any] | None
    ) -> None:
        """Reverse an install whose Step 4/5 failed: restore the manifest through the
        pipeline (converging the live app and the fleet back) when the change had
        persisted, then pip uninstall the freshly-installed package. A failed
        sub-step escalates to :class:`InstallUnwindError`; otherwise the caller
        re-raises the original step error."""
        try:
            if saved_manifest is not None:
                await self._svc().apply_replace(saved_manifest)
        except Exception as unwind_error:
            raise InstallUnwindError(step_error, unwind_error) from step_error
        try:
            await self._pip_runner(uninstall_args(package))
        except Exception as unwind_error:
            raise InstallUnwindError(step_error, unwind_error) from step_error

    # -- uninstall ----------------------------------------------------------

    async def uninstall(self, ref: str) -> dict[str, Any]:
        """Uninstall a marketplace-installed plugin — convergent and
        registry-free.

        Order: pip pre-flight (a pipless environment fails HERE, before the
        manifest is touched); read the stored record (unknown ref → not-installed
        state error); reconstruct the installed ``PluginSpec`` from LOCAL truth
        (a corrupt row → :class:`LocalStateError`); unpatch the manifest through
        the one door and reload to CONVERGENCE (a re-run whose manifest is already
        stripped still reloads when the spec targets manifest fields, so a prior
        run's failed deregister reload is re-attempted; the reload is skipped only
        for a plugin with no manifest provides — all env-selected); pip uninstall;
        drop the attribution row.

        There is no unwind-to-installed: re-running forward is the recovery, and
        every failure states exactly which steps remain. Only skeleton state
        reverts fully — see the pip-transaction caveat.
        """
        async with self._guard():
            return await self._uninstall_locked(ref)

    async def _uninstall_locked(self, ref: str) -> dict[str, Any]:
        ensure_pip_available()
        row = await self._store.get(ref)
        if row is None:
            raise InstallStateError(f"{ref} is not installed", not_installed=True)
        spec = _spec_from_row(row)

        cm = self._cm()
        reload_result: dict[str, Any] | None
        changed = remove_provides(dict(cm.read_manifest_preserved()), spec)
        if changed or _has_manifest_provides(spec):
            # Converge the live registration to the stripped manifest through the
            # pipeline BEFORE removing the package. The apply runs even when nothing
            # changed now but the spec targets manifest fields: a prior run may have
            # persisted the stripped manifest and then failed the deregister reload
            # (the store row still exists), leaving the tools live — and a re-run MUST
            # re-attempt the persist+reload so it does not pip uninstall + drop the row
            # while the tools stay registered. The mutator re-strips the seam's own
            # document (idempotent when it was already clean).
            def mutator(document: dict[str, Any]) -> None:
                remove_provides(document, spec)

            reload_result = _reload_report(await self._apply_composed(mutator))
        else:
            # A plugin whose provides are all env-selected wrote no manifest
            # entry, so there is nothing to deregister — no apply, no reload.
            reload_result = None

        # A failure here leaves the app converged but the package present;
        # re-running uninstall (step 3 now a no-op) completes the removal.
        await self._pip_runner(uninstall_args(spec.package))
        await self._store.delete(ref)

        return {"ref": ref, "uninstalled": True, "reload": reload_result, "notes": _uninstall_notes(spec)}

    # -- update -------------------------------------------------------------

    async def update(self, ref: str, version: str | None = None) -> dict[str, Any]:
        """Update an installed plugin to a newer (or named) version — an install
        of the new version with the same pre-flights, in one manifest
        read-modify-write.

        Order: parse the ref (a malformed ref is the caller's 400, checked before
        the store read so an unparseable ref is never a phantom 404); pip
        pre-flight; read the stored record (unknown ref →
        not-installed state error) for the old spec and pin; resolve the target
        (kill/advisory/spec-validation/contract checks; target == installed →
        state error); collision pre-flight for the NEW spec against the manifest
        with the OLD spec's entries removed in memory (a rename inside one plugin
        must not self-collide); ``pip install`` the new pin (upgrades in place);
        one pipeline apply removing the old entries and applying the new (persist +
        reload + broadcast); upsert the attribution row.

        A Step 5/6 failure unwinds in order — reinstall the OLD pin (a github old pin
        re-fetched and re-verified from the stored row's artifact_ref + sha256), then
        restore the manifest through the pipeline (the old wheel must be back before
        the restore's reload, or reloading the old manifest against the new wheel
        fails when a module moved) — and a failed sub-step escalates to
        :class:`InstallUnwindError`. Only skeleton state reverts fully — see the
        pip-transaction caveat.
        """
        async with self._guard():
            return await self._update_locked(ref, version)

    async def _update_locked(self, ref: str, version: str | None) -> dict[str, Any]:
        ns, name = _parse_ref(ref)
        ensure_pip_available()
        row = await self._store.get(ref)
        if row is None:
            raise InstallStateError(f"{ref} is not installed", not_installed=True)
        old_spec = _spec_from_row(row)

        resolved = await self._registry.resolve(ns, name, version)
        new_spec, source = self._prepare_resolved(resolved)
        pinned_version = _require(resolved, "version")
        if pinned_version == row.version:
            raise InstallStateError(f"{ref} is already at {pinned_version}")

        cm = self._cm()
        preview = dict(cm.read_manifest_preserved())
        remove_provides(preview, old_spec)
        found = collisions(preview, new_spec)
        if found:
            raise ManifestCollisionError("; ".join(found))

        # Step 4 — pip upgrade in place (github: fetch + verify + install the
        # local tarball). A failure propagates as-is: nothing skeleton-side has
        # changed yet, and pip's own transaction is the boundary.
        pip_output = await self._pip_install(
            new_spec.package, pinned_version, source, resolved.get("artifact_ref"), resolved.get("sha256")
        )

        saved_manifest = copy.deepcopy(cm.read_manifest_preserved())
        manifest_persisted = False
        try:
            # Step 5 — one pipeline apply: the mutator removes the old provides and
            # applies the new; the pipeline validates the resolved compose (mapped to
            # the typed compose error on a fault), persists, reloads, and broadcasts.
            def mutator(document: dict[str, Any]) -> None:
                remove_provides(document, old_spec)
                apply_provides(document, new_spec)

            apply_result = await self._apply_composed(mutator)
            manifest_persisted = True
            # Step 6 — attribution upsert.
            repo_url, tag, artifact_ref, sha256 = _pin_provenance(resolved, source)
            await self._store.record(
                ref, pinned_version, source, repo_url, tag, artifact_ref, sha256, new_spec.model_dump(mode="json")
            )
        except Exception as step_error:
            persisted = manifest_persisted or isinstance(step_error, FleetBroadcastError)
            await self._unwind_update(
                step_error,
                old_package=old_spec.package,
                row=row,
                saved_manifest=saved_manifest if persisted else None,
            )
            raise

        return {
            "ref": ref,
            "version": pinned_version,
            "package": new_spec.package,
            "advisories": _require_list(resolved, "advisories"),
            "notes": _install_notes(new_spec),
            "reload": _reload_report(apply_result),
            "pip_output": pip_output,
        }

    async def _unwind_update(
        self,
        step_error: Exception,
        *,
        old_package: str,
        row: InstallRecord,
        saved_manifest: dict[str, Any] | None,
    ) -> None:
        """Reverse an update whose Step 5/6 failed: reinstall the OLD pin, then
        restore the manifest through the pipeline (when the change had persisted).
        The old wheel goes back BEFORE the restore's reload so the old manifest never
        loads against the new wheel. A github old pin is reinstalled through the SAME
        fetch-and-verify path as a forward install, using the stored row's
        ``artifact_ref`` + ``sha256`` — the old version's integrity is re-checked,
        never cloned from a mutable tag. A failed sub-step (including an integrity
        mismatch on the old artifact) escalates to :class:`InstallUnwindError`."""
        try:
            await self._pip_install(old_package, row.version, row.source, row.artifact_ref, row.sha256)
            if saved_manifest is not None:
                await self._svc().apply_replace(saved_manifest)
        except Exception as unwind_error:
            raise InstallUnwindError(step_error, unwind_error) from step_error

    # -- shared resolve handling -------------------------------------------

    def _prepare_resolved(self, resolved: dict[str, Any]) -> tuple[PluginSpec, str]:
        """Validate and vet a resolve response, returning ``(spec, source)``.

        Refuses a non-withdrawn critical advisory, validates the shipped
        ``PluginSpec`` (a malformed spec is a registry-data fault → 502, never a
        caller 400), requires the github artifact provenance (repository_url + tag
        for display, artifact_ref + sha256 for the verified fetch), and checks the
        plugin's ``contract_range`` against the installed ``tai-contract`` version.
        """
        for advisory in _require_list(resolved, "advisories"):
            if advisory.get("withdrawn_at") is None and advisory.get("severity") == "critical":
                summary = advisory.get("summary", "no summary")
                raise VersionRefusedError(f"a critical advisory affects this version: {summary}")

        try:
            spec = PluginSpec.model_validate(_require(resolved, "spec"))
        except ValidationError as exc:
            raise RegistryResponseError(f"registry served an invalid plugin spec: {exc}", status=None) from exc

        source = _require(resolved, "source")
        if source == "github":
            if not resolved.get("repository_url") or not resolved.get("tag"):
                raise RegistryResponseError(
                    "registry resolve response for a github source is missing repository_url or tag",
                    status=None,
                )
            if not resolved.get("artifact_ref") or not resolved.get("sha256"):
                raise RegistryResponseError(
                    "registry resolve response for a github source is missing artifact_ref or sha256",
                    status=None,
                )
        elif source != "pypi":
            raise RegistryResponseError(f"registry returned an unknown install source {source!r}", status=None)

        self._check_contract(_require(resolved, "contract_range"))
        return spec, source

    def _check_contract(self, contract_range: str) -> None:
        """Require the installed ``tai-contract`` version to satisfy the plugin's
        ``contract_range``.

        ``prereleases=True`` so a dev-versioned tai-contract (an editable checkout
        reporting e.g. ``0.5.0.dev3``) inside the range still passes — a developer
        environment is not spuriously refused. A malformed range is registry data
        → 502, never a caller 400.
        """
        # ``contract_range`` is a non-null string by construction — ``_require``
        # rejects a null and the registry client's resolve boundary types the field
        # when present — so this check owns only the string's FORMAT: a non-PEP440
        # specifier set is garbled registry data → 502, never a caller 400.
        try:
            specifier = SpecifierSet(contract_range)
        except InvalidSpecifier as exc:
            raise RegistryResponseError(
                f"registry served an unusable contract_range {contract_range!r}: {exc}", status=None
            ) from exc
        installed = importlib.metadata.version("tai-contract")
        if not specifier.contains(installed, prereleases=True):
            raise ContractIncompatibleError(
                f"plugin requires tai-contract {contract_range}, but {installed} is installed"
            )


def _parse_ref(ref: str) -> tuple[str, str]:
    """Split ``namespace/name`` into its two lowercase halves.

    Raises :class:`MalformedRefError` (surfaced by the boundary as a 400) on
    anything but exactly one ``/`` with two non-empty lowercase halves. The typed
    error is distinct from a server-side invariant fault, so the operation layer
    maps ONLY a malformed ref to a bad-request response.
    """
    parts = ref.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise MalformedRefError(f"ref must be 'namespace/name', got {ref!r}")
    namespace, name = parts
    if namespace != namespace.lower() or name != name.lower():
        raise MalformedRefError(f"ref must be lowercase 'namespace/name', got {ref!r}")
    return namespace, name


def _spec_from_row(row: InstallRecord) -> PluginSpec:
    """Reconstruct the stored ``PluginSpec`` from an attribution row — LOCAL
    truth, no registry call. A row that no longer validates is corrupt local
    state (:class:`LocalStateError`, a 500), never the caller's request."""
    try:
        return PluginSpec.model_validate(row.spec)
    except ValidationError as exc:
        raise LocalStateError(f"the stored spec for {row.ref} is corrupt: {exc}") from exc


def _pin_provenance(resolved: dict[str, Any], source: str) -> tuple[str | None, str | None, str | None, str | None]:
    """The ``(repository_url, tag, artifact_ref, sha256)`` to store for the pin:
    the resolve values only for a github source, else all ``None`` — so a pypi row
    keeps every pin column NULL even when the resolve response carries them. The
    stored ``artifact_ref`` + ``sha256`` are what let update-unwind reinstall the
    old github pin through the same verified fetch path."""
    if source == "github":
        return (
            resolved.get("repository_url"),
            resolved.get("tag"),
            resolved.get("artifact_ref"),
            resolved.get("sha256"),
        )
    return None, None, None, None


def _has_manifest_provides(spec: PluginSpec) -> bool:
    """Whether the spec provides any item that wires into a manifest field (a
    non-env-selected item).

    Such a plugin's live registration must be converged by an uninstall reload
    even when the manifest is already stripped — a prior partial run may have
    persisted the stripped manifest but failed the deregister reload, so the tools
    stay live until a re-run reloads.
    """
    for item in spec.provides:
        binding = KIND_MANIFEST_BINDINGS.get(item.kind)
        if binding is not None and binding.mode != "env_selected":
            return True
    return False


def _env_selected_items(spec: PluginSpec) -> list[PluginItem]:
    """The spec's provides items that wire into no manifest field (the
    env-selected ``config`` kind) — pip install/uninstall is their whole
    registration."""
    items: list[PluginItem] = []
    for item in spec.provides:
        binding = KIND_MANIFEST_BINDINGS.get(item.kind)
        if binding is not None and binding.mode == "env_selected":
            items.append(item)
    return items


def _install_notes(spec: PluginSpec) -> list[str]:
    """One activation note per env-selected item: installed but inactive until
    ``TAI_CONFIG_MODE`` selects it, and only providers the skeleton's fixed
    mode→module map covers can be selected (a new provider needs a skeleton-side
    enum/map entry)."""
    return [
        f"{item.name!r} is installed but inactive: TAI_CONFIG_MODE selects config providers from the "
        "skeleton's fixed mode->module map, so activation needs the mode to exist there "
        "(tai-config-k8s is the one provider covered today; a new provider needs a skeleton-side map entry)"
        for item in _env_selected_items(spec)
    ]


def _uninstall_notes(spec: PluginSpec) -> list[str]:
    """One warning per env-selected item: if ``TAI_CONFIG_MODE`` currently selects
    the removed provider, the next boot fails importing it until the operator
    re-points or unsets the env var."""
    return [
        f"{item.name!r} was a config provider: if TAI_CONFIG_MODE currently selects it, the next boot will "
        "fail importing the removed provider until you re-point or unset TAI_CONFIG_MODE"
        for item in _env_selected_items(spec)
    ]


def _require(resolved: dict[str, Any], key: str) -> Any:
    """A required resolve-response field — present AND non-null — or a typed
    registry-data fault (502).

    The client boundary type-checks a field only when it is present and non-null
    (a null there is legitimate for the github-only optional pins), so ``required``
    has to mean non-null here: a null in an always-present field (``version``,
    ``contract_range``, ``spec``, ``source``) is missing data, and rejecting it
    keeps the ``str``-typed parsers downstream honest — they never see ``None``."""
    value = resolved.get(key)
    if value is None:
        raise RegistryResponseError(f"registry resolve response is missing {key!r}", status=None)
    return value


def _require_list(resolved: dict[str, Any], key: str) -> list[Any]:
    """A required list-shaped resolve-response field, or a typed registry-data
    fault (502)."""
    value = _require(resolved, key)
    if not isinstance(value, list):
        raise RegistryResponseError(f"registry resolve response {key!r} is not a list", status=None)
    return value
