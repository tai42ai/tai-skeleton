"""Deployment-config operations — ``/api/config/*``.

A thin skin over the config facets (``tai_app.config``), the admin reload seam
(``tai_app.admin``), the settings registry, and the worker bus:

* ``read_env`` — the stored env map plus the operator's secret-key marks.
* ``write_env`` — merge a ``{key: value}`` env map (all values strings) through the
  :class:`~tai_skeleton.config.service.ConfigService` pipeline: validate the effective
  config, write the env, hot-reload the process, and broadcast the reload to the fleet.
* ``read_mode`` — the active config backend mode (``file`` / ``k8s``).
* ``read_settings_schema`` — every registered settings group with per-field current
  resolved values.
* ``reload_config`` — a soft-restart (refresh env, reset settings caches,
  re-initialize from the manifest) on this worker, broadcast to the fleet. Distinct
  from the fleet soft-restart door ``fleet_reload_config`` (``/api/fleet/reload-config``).

``write_env`` and ``reload_config`` mutate the running deployment, so both are
``destructive`` and honor the reload gate.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, RootModel
from tai_contract.app import tai_app
from tai_kit.settings import registered_settings

from tai_skeleton.app.boot_rules import BackendNeedsBusError
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.config import config_mode
from tai_skeleton.config.service import ConfigService
from tai_skeleton.operations import BadRequestError, operation
from tai_skeleton.operations._broadcast import apply_response, broadcast

# Importing this module registers ``EnvSecretMarksSettings`` (registration runs at
# class-definition time) so the marks group appears in the settings schema, and
# exposes the reload-aware accessor used by the env read.
from tai_skeleton.settings.env_secret_marks import env_secret_marks_settings


class EnvUpdate(RootModel[dict[str, str]]):
    """An env override map — a ``{name: value}`` object whose values are all
    strings, merged into the stored env config before a hot reload."""


class ReloadConfigRequest(BaseModel):
    """A local reload-config request — an optional ``targets`` list restricting the
    fan-out of the soft-restart to named workers (all workers when omitted)."""

    targets: list[str] | None = None


def _stored_env() -> dict[str, str]:
    """The stored env map, treating a missing ``.env`` as an empty map.

    A never-written env store (``FileNotFoundError``) means "no stored overrides",
    not a failure — the caller falls back to process env / defaults.
    """
    try:
        return tai_app.config.config_manager.read_env()
    except FileNotFoundError:
        return {}


@operation(summary="Read the stored env config and secret-key marks", tags=["config"])
async def read_env() -> dict:
    """Return the stored env map alongside the operator's secret-key marks.

    ``data.env`` is the stored env key-value map (a never-written store yields
    an empty map). ``data.secret_keys`` is the current
    ``EnvSecretMarksSettings.secret_keys`` — the env key names the operator
    marked secret so the editor masks them on display.
    """
    return {"env": _stored_env(), "secret_keys": env_secret_marks_settings().secret_keys}


@operation(summary="Read the active config backend mode", tags=["config"])
async def read_mode() -> dict:
    return {"config_mode": config_mode()}


@operation(summary="List settings groups with their resolved field values", tags=["config"])
async def read_settings_schema() -> dict:
    """Return every registered settings group with per-field current values.

    Each group carries the settings class' field metadata plus a ``value`` per
    field, resolved with pydantic-settings precedence: ``os.environ`` wins (in
    k8s config mode the cluster injects vars the dotenv store never sees), then
    the stored env override, then the field default. Nested-group reference
    fields (``env_var == ""``) are non-editable and report ``value: null``.
    Secret fields report their real value — this authed surface round-trips
    values through the editor; masking is display-side only, never on the wire.

    Only IMPORTED settings classes appear: in a running skeleton the kit,
    skeleton, and manifest-loaded plugin modules — the intended scope.
    """
    stored = _stored_env()
    groups = []
    for cls_info in registered_settings():
        fields = []
        for field in cls_info.fields:
            payload = field.model_dump()
            if field.env_var and field.env_var in os.environ:
                payload["value"] = os.environ[field.env_var]
            elif field.env_var and field.env_var in stored:
                payload["value"] = stored[field.env_var]
            elif field.env_var:
                payload["value"] = field.default
            else:
                payload["value"] = None
            fields.append(payload)
        groups.append(
            {
                "name": cls_info.name,
                "module": cls_info.module,
                "qualname": cls_info.qualname,
                "fields": fields,
            }
        )
    return {"groups": groups}


@operation(
    summary="Merge env overrides and hot-reload the process config",
    tags=["config"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError],
    request_model=EnvUpdate,
)
async def write_env(env: dict[str, str]) -> dict:
    # Merge the env overrides through the pipeline: ConfigService validates the
    # effective (resolved) config against the backend-needs-bus invariant, writes the
    # env, reloads locally, and broadcasts the reload to the whole fleet. An invalid
    # env key or effective config rejects before anything is written and maps to 400.
    # ``BackendNeedsBusError`` is a RuntimeError (a boot-time refusal must still crash
    # loudly), so the mutate-time path catches it explicitly to map it to a loud,
    # actionable 400 naming TAI_BUS_REDIS_URL rather than letting it escape as a 500.
    try:
        result = await ConfigService.from_app().apply_env_change(env)
    except (ValueError, BackendNeedsBusError) as exc:
        raise BadRequestError(str(exc)) from exc
    return apply_response(result)


@operation(
    summary="Soft-restart the process from its manifest",
    tags=["config"],
    destructive=True,
    reload_gated=True,
    request_model=ReloadConfigRequest,
)
async def reload_config(targets: list[str] | None = None) -> Any:
    """Soft-restart: refresh env from the config manager, reset every settings cache,
    and re-initialize from the manifest — in-process, no pod restart. Applied on this
    worker (when it is a target) and broadcast to the fleet (all workers, or only
    ``targets``); the response embeds the per-origin fleet report.

    Heavy (a full re-init); meant for env/config saves, not tool edits. A convergence
    op: its whole purpose is aligning siblings to persisted state, so a failed local
    reload still broadcasts and then re-raises with the fleet report attached.
    """
    # Run the heavy sync reload on a worker thread through the gate so this call (on
    # the serving loop) does not freeze it.
    return await broadcast(
        {"op": "reload_config"},
        targets,
        lambda: reload_gate.run(tai_app.admin.reload_config),
        publish_on_local_failure=True,
    )
