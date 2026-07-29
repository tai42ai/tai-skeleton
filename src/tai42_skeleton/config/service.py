"""The single manifest-mutation pipeline.

Every manifest / env mutation crosses one pipeline so no writer can forget a step:

    transaction → read → mutate → VALIDATE → SEAL → persist → local reload → broadcast

:class:`ConfigService` exposes three entrypoints — :meth:`apply_change` (a
read-modify-write through the config manager's transaction), :meth:`apply_replace`
(a whole-document replace), and :meth:`apply_env_change` (an env override) — and
each returns a structured :class:`ApplyResult` carrying the persisted document
(where applicable), the local reload result, and the awaited per-origin fleet
report.

Validation runs on the RESOLVED projection of the change (``!ENV`` markers
materialized in memory for validation ONLY — the PRESERVED document is what
persists) and rejects an invalid change before anything is persisted. The
backend-needs-bus invariant runs in the same step, in both directions and in
agreement with the boot/reload-time rule.

The SEAL step (:meth:`ConfigService.apply_change` /
:meth:`ConfigService.apply_replace`, never the env change) then guarantees no
resolved secret bakes to disk: because the only read surface for the manifest's
``mcp`` section is the RESOLVED view, a natural round-trip (read the resolved view,
edit, post it back) hands the pipeline resolved secret values. Before the document
persists, :func:`~tai42_skeleton.config.secret_seal.seal_resolved_secrets` retags any
leaf that still equals a currently-resolved secret back to its ``!ENV`` marker, and
refuses (a :class:`~tai42_skeleton.config.secret_seal.ResolvedSecretError`, a
``ValueError`` the operations layer maps to a 400) any stranded resolved secret with
no marker origin — so a resolved round-trip preserves the operator's markers and a
plaintext secret never reaches the store.

The broadcast tail is the whole fleet, always: ``publish({"op": "reload_config"},
targets=None, local=<local reload result>)`` — a persisted change reaches every
worker, and each subscriber re-reads the persisted store itself. An unconfirmed
origin is a loud ERROR log and an explicit entry in the report, but the call still
succeeds because persist + local reload landed; recovery is the fleet-reload door.
Once the persist has committed the contract is structural: EVERY subsequent failure
surfaces as a :class:`~tai42_skeleton.operations._broadcast.FleetBroadcastError`
carrying a fleet report, never a raw exception. If the LOCAL reload raises after the
persist landed, the broadcast STILL goes out (siblings converge on the persisted
state) and the call re-raises the local failure with the fleet report attached; if
the broadcast itself raises anything the bus does not fold into a returned
bus-unreachable report, that too is re-raised as a ``FleetBroadcastError`` with the
unreachable report shape — a stranded fleet never hides behind a local error, and a
committed persist never escapes as a raw broadcast fault.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from pyaml_env import parse_config
from tai42_kit.utils.data import dump_manifest

from tai42_skeleton.app.boot_rules import check_backend_needs_bus
from tai42_skeleton.app.bus import FleetResult, LocalApplyResult, OpOutcome
from tai42_skeleton.app.bus_settings import bus_settings
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.config.secret_seal import seal_resolved_secrets
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations._broadcast import FleetBroadcastError, fleet_fanout, log_non_convergence

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ruamel.yaml.comments import CommentedMap

# The env var that configures the worker bus; an env change that empties it removes
# the bus. Kept in sync with the boot rules' bus-var name.
_BUS_VAR = "TAI_BUS_REDIS_URL"


class _ManifestStore(Protocol):
    """The config-manager surface the pipeline drives — the transactional seams plus
    the reads the env-change validation needs. The concrete provider is the active
    :class:`~tai42_contract.config.manager.ConfigManager`."""

    def mutate_manifest(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]: ...

    def replace_manifest(self, document: dict[str, Any]) -> dict[str, Any]: ...

    def write_env(self, config: dict[str, str]) -> None: ...

    def read_env(self) -> dict[str, str]: ...

    def read_manifest_preserved(self) -> dict[str, Any]: ...


class _ReloadAdmin(Protocol):
    """The reload seam the pipeline drives on the local worker after a persist."""

    def reload_config(self) -> dict[str, Any]: ...


class _FleetPublisher(Protocol):
    """The worker-bus publish surface the pipeline broadcasts through."""

    async def publish(
        self, op: dict[str, Any], targets: list[str] | None, local: LocalApplyResult | None
    ) -> FleetResult: ...


@dataclass(frozen=True)
class ApplyResult:
    """The structured outcome of one pipeline run.

    ``document`` is the persisted PRESERVED-view manifest (``!ENV`` markers intact)
    for :meth:`ConfigService.apply_change` / :meth:`ConfigService.apply_replace`, and
    ``None`` for :meth:`ConfigService.apply_env_change` (no manifest document changed).
    ``local`` is this worker's local reload result. ``fleet`` is the awaited per-origin
    broadcast report, which carries the two honest failure shapes itself — a reachable
    bus with an unconfirmed origin named in ``fleet.results`` (``fleet.ok`` is then
    ``False``), or a bus-unreachable result (``fleet.reachable`` is ``False`` with only
    ``fleet.error``). Both are returned successes: persist + local reload landed, and
    recovery is a fleet reload.
    """

    fleet: FleetResult
    local: dict[str, Any]
    document: dict[str, Any] | None = None

    @property
    def fanout(self) -> dict[str, Any]:
        """The mode-wrapped fan-out summary of this run's broadcast — the same shape
        :func:`~tai42_skeleton.operations._broadcast.apply_response` embeds under
        ``fanout`` (local-only / fleet / unreachable). A writer that returns a bare
        result (not the ``apply_response`` merge) embeds this value directly."""
        return fleet_fanout(self.fleet)


class ConfigService:
    """The one pipeline every manifest / env mutation crosses.

    Constructed per call site from the running app via :meth:`from_app`, or directly
    with the config manager, the reload seam, and the worker bus for testing.
    """

    def __init__(self, config_manager: _ManifestStore, admin: _ReloadAdmin, bus: _FleetPublisher) -> None:
        self._config_manager = config_manager
        self._admin = admin
        self._bus = bus

    @classmethod
    def from_app(cls) -> ConfigService:
        """Wire the pipeline from the running app: the active config manager, the admin
        reload seam, and this process's worker bus."""
        from tai42_contract.app import tai42_app

        from tai42_skeleton.app import instance

        return cls(config_manager=tai42_app.config.config_manager, admin=tai42_app.admin, bus=instance.app.bus)

    async def apply_change(self, mutator: Callable[[dict[str, Any]], None]) -> ApplyResult:
        """Read-modify-write: run ``mutator`` on the PRESERVED manifest inside the
        config-manager transaction, VALIDATE the resolved projection of the mutated
        document, persist, locally reload, and broadcast to the whole fleet.

        ``mutator`` edits the passed document IN PLACE and must be pure / re-runnable
        (the transaction may re-run it on a concurrency conflict). An invalid mutation
        raises inside the transaction, so nothing is persisted. The mutated document
        is SEALED against the pre-mutation document before it persists, so a mutator
        that ingests resolved secret values (a resolved round-trip) cannot bake a
        secret to disk — see :meth:`_seal_secrets`."""

        def guarded(document: dict[str, Any]) -> None:
            # Snapshot the preserved document the mutator received BEFORE it runs, so
            # the seal can retag the mutated document against the current manifest's
            # markers and their resolved values.
            current = copy.deepcopy(document)
            mutator(document)
            self._validate_manifest(document)
            self._seal_secrets(document, current)

        persisted = self._config_manager.mutate_manifest(guarded)
        return await self._reload_and_broadcast(document=persisted)

    async def apply_replace(self, document: dict[str, Any]) -> ApplyResult:
        """Replace the whole manifest: VALIDATE the resolved projection of ``document``
        BEFORE it is persisted (a replace has no mutator to abort), then replace,
        locally reload, and broadcast to the whole fleet.

        The caller supplies the PRESERVED-view document (``!ENV`` markers, never
        resolved values) — the seam persists it verbatim. The document is SEALED
        against the CURRENT persisted manifest before it persists, so a replacement
        carrying a resolved secret (a resolved round-trip) is retagged or refused —
        see :meth:`_seal_secrets`."""
        self._validate_manifest(document)
        self._seal_secrets(document, self._read_preserved_manifest())
        persisted = self._config_manager.replace_manifest(document)
        return await self._reload_and_broadcast(document=persisted)

    async def apply_env_change(self, changes: dict[str, str]) -> ApplyResult:
        """Apply env overrides: VALIDATE the effective/resolved config (the manifest's
        ``!ENV`` markers materialized against the post-change env) through the SAME
        backend-needs-bus gate, then merge the overrides, locally reload, and broadcast
        to the whole fleet. An invalid effective config raises before anything is
        written."""
        self._validate_env(changes)
        self._config_manager.write_env(changes)
        return await self._reload_and_broadcast(document=None)

    # -- Validation ------------------------------------------------------------

    def _validate_manifest(self, document: Mapping[str, Any]) -> None:
        """Validate the resolved projection of a manifest change: the pydantic
        ``Manifest`` schema plus the backend-needs-bus invariant. The env is unchanged
        by a manifest mutation, so markers resolve against the current process env and
        the bus configuration is the current one."""
        manifest = self._validated_projection(document)
        check_backend_needs_bus(backend_module=manifest.backend_module, bus_configured=bus_settings().enabled)

    def _validate_env(self, changes: dict[str, str]) -> None:
        """Validate the effective config an env change produces: resolve the persisted
        manifest's ``!ENV`` markers against the post-change env (so a marker that
        materializes a backend participates) and evaluate the backend-needs-bus
        invariant against the post-change bus configuration (so removing the bus while
        a backend remains is rejected too)."""
        effective = self._effective_env(changes)
        with _environ(effective):
            manifest = self._validated_projection(self._read_preserved_manifest())
        bus_configured = bool(effective.get(_BUS_VAR, "").strip())
        check_backend_needs_bus(backend_module=manifest.backend_module, bus_configured=bus_configured)

    def _validated_projection(self, document: Mapping[str, Any]) -> Manifest:
        """Build the RESOLVED in-memory projection of a PRESERVED document (``!ENV``
        markers materialized for validation only) and validate it against the
        ``Manifest`` schema. Raises on an invalid document."""
        return Manifest.model_validate(self._resolve(document))

    def _resolve(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """The RESOLVED projection of a PRESERVED document — ``!ENV`` markers
        materialized against the current process env, structure otherwise intact. The
        one resolution the pipeline uses for both validation and the secret seal."""
        return cast("dict[str, Any]", parse_config(data=dump_manifest(cast("CommentedMap", document))) or {})

    # -- Secret seal -----------------------------------------------------------

    def _seal_secrets(self, document: dict[str, Any], current_preserved: Mapping[str, Any]) -> None:
        """Seal *document* against the current manifest before it persists.

        *current_preserved* is the CURRENT manifest in its preserved view (``!ENV``
        markers) — the pre-mutation document for :meth:`apply_change`, the persisted
        manifest for :meth:`apply_replace`. It is resolved once to know each marker's
        resolved value, then :func:`~tai42_skeleton.config.secret_seal.seal_resolved_secrets`
        retags any leaf of *document* that still equals a resolved secret back to its
        marker (a resolved round-trip preserves the operator's markers) and raises a
        :class:`~tai42_skeleton.config.secret_seal.ResolvedSecretError` on a stranded
        resolved secret with no marker origin. A pure in-place mutation whose leaves
        already carry markers is a no-op. *document* is mutated in place."""
        seal_resolved_secrets(document, cast("dict[str, Any]", current_preserved), self._resolve(current_preserved))

    def _read_preserved_manifest(self) -> dict[str, Any]:
        """The persisted manifest in its PRESERVED view, or an empty document when no
        manifest exists yet — a deployment with no manifest registers no backend, so
        the env-change invariant has nothing to reject."""
        try:
            return self._config_manager.read_manifest_preserved()
        except FileNotFoundError:
            return {}

    def _effective_env(self, changes: dict[str, str]) -> dict[str, str]:
        """The effective env an :meth:`apply_env_change` produces, as the reloaded
        process would see it.

        The stored env is merged with ``changes`` (empties are dropped — the store
        filters them), then overlaid onto the current process env, which a reload
        applies with ``os.environ.update``. A key the change empties is treated as
        removed so a bus-removing change is visible to the invariant."""
        stored = self._read_stored_env()
        removed = {key for key, value in changes.items() if value == ""}
        merged = {key: value for key, value in {**stored, **changes}.items() if value != ""}
        effective = {key: value for key, value in os.environ.items() if key not in removed}
        effective.update(merged)
        return effective

    def _read_stored_env(self) -> dict[str, str]:
        """The stored env map, treating a never-written store as empty."""
        try:
            return self._config_manager.read_env()
        except FileNotFoundError:
            return {}

    # -- Persist tail ----------------------------------------------------------

    async def _reload_and_broadcast(self, *, document: dict[str, Any] | None) -> ApplyResult:
        """Locally reload through the gate, then broadcast the reload to the whole
        fleet and embed the report.

        Post-persist contract: this runs only after the persist has committed, and it
        guarantees EVERY subsequent failure — the local reload OR the broadcast itself
        — surfaces as a :class:`FleetBroadcastError` carrying a fleet report, never a
        raw exception. So every caller can key on one shape: a raised
        ``FleetBroadcastError`` means the change LANDED but its propagation failed
        (restore / converge), while a bare raise below this point would mean nothing
        persisted.

        A local reload that raises after the persist landed does NOT abort the
        broadcast — the siblings must still converge on the persisted state — so the
        reload publishes anyway with a ``failed`` self entry and then re-raises the
        local failure with the fleet report attached. A broadcast that raises anything
        other than the transport-unreachable shape the bus already folds into a
        returned report (e.g. a redis ``ResponseError``, or a malformed presence key
        the census cannot parse) is caught here and re-raised as a
        ``FleetBroadcastError`` carrying the honest bus-unreachable report (no origin
        list, only the error) — the same shape the bus returns for a transport
        failure — so a raw broadcast error can never escape the committed persist. An
        unconfirmed origin on the happy path is a loud ERROR log and an explicit
        report entry, but the call returns successfully."""
        op_name = "reload_config"
        local_failure: Exception | None = None
        local_result: dict[str, Any] | None = None
        try:
            local_result = await reload_gate.run(self._admin.reload_config)
        except Exception as exc:
            local_failure = exc
            local = LocalApplyResult(outcome=OpOutcome.failed, error=f"{type(exc).__name__}: {exc}")
        else:
            local = LocalApplyResult(outcome=OpOutcome.applied, payload=local_result)

        try:
            report = await self._bus.publish({"op": op_name}, None, local)
        except Exception as broadcast_error:
            # The persist already committed, so a raw broadcast failure must not escape
            # the post-persist contract. Surface it as a FleetBroadcastError carrying
            # the honest bus-unreachable report (no origin list, only the error) — the
            # same shape WorkerBus.publish returns for a transport failure — so every
            # caller treats the change as landed-but-propagation-failed and restores or
            # converges rather than half-writing. A local reload that ALSO failed is
            # noted in the report error so it is not lost behind the broadcast error.
            error = f"{type(broadcast_error).__name__}: {broadcast_error}"
            if local_failure is not None:
                error = f"{error} (local reload also failed: {type(local_failure).__name__}: {local_failure})"
            report = FleetResult(op=op_name, reachable=False, error=error)
            raise FleetBroadcastError(op_name, report, broadcast_error) from broadcast_error

        # The report rides the response, but an unconfirmed origin is also a loud,
        # visible failure — never a silently stale sibling.
        log_non_convergence(report)
        if local_failure is not None:
            raise FleetBroadcastError(report.op, report, local_failure) from local_failure
        # local_result is set on the success path (no local_failure).
        return ApplyResult(fleet=report, local=cast("dict[str, Any]", local_result), document=document)


@contextmanager
def _environ(env: dict[str, str]) -> Iterator[None]:
    """Temporarily replace ``os.environ`` with ``env`` for the duration of the block,
    restoring it exactly afterwards. Used to resolve a manifest's ``!ENV`` markers
    against a proposed post-change env without mutating the real process env."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
