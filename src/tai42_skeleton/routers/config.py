"""HTTP routes for the deployment config surface — ``/api/config/*``.

AUTHED thin adapters over operations in ``tai42_skeleton.operations.config``:

* ``GET /api/config/env`` — the stored env config plus the operator's secret-key
  marks (a secret-bearing read; acceptable on this authed route).
* ``POST /api/config/env`` — merge a ``{key: value}`` env map (all values strings),
  then hot-reload the process config; returns the reload result.
* ``GET /api/config/mode`` — the active config backend mode (``file`` / ``k8s``).
* ``GET /api/config/settings-schema`` — every registered settings group with its
  field metadata and each field's current resolved value.
* ``POST /api/config/reload`` — a local soft-restart (refresh env, reset settings
  caches, re-initialize from the manifest), fanned out to every worker when a
  backend is configured.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.config import read_env as _read_env_op
from tai42_skeleton.operations.config import read_mode as _read_mode_op
from tai42_skeleton.operations.config import read_settings_schema as _read_settings_schema_op
from tai42_skeleton.operations.config import reload_config as _reload_config_op
from tai42_skeleton.operations.config import write_env as _write_env_op


async def _extract_env_update(request: Request) -> dict[str, Any]:
    """Parse and validate the env-merge body at the HTTP edge, preserving the door's
    hand-authored 400 messages (a plain request-model parse would answer 422). Yields
    the operation's flat ``env`` kwarg."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object of env values")
    if not all(isinstance(v, str) for v in body.values()):
        raise BadRequestError("all env values must be strings")
    return {"env": body}


read_env = register_operation_route(
    tai42_app,
    operation_metadata_of(_read_env_op),
    path="/api/config/env",
    method="GET",
)

write_env = register_operation_route(
    tai42_app,
    operation_metadata_of(_write_env_op),
    path="/api/config/env",
    method="POST",
    context_extractor=_extract_env_update,
)

read_mode = register_operation_route(
    tai42_app,
    operation_metadata_of(_read_mode_op),
    path="/api/config/mode",
    method="GET",
)

read_settings_schema = register_operation_route(
    tai42_app,
    operation_metadata_of(_read_settings_schema_op),
    path="/api/config/settings-schema",
    method="GET",
)

reload_config = register_operation_route(
    tai42_app,
    operation_metadata_of(_reload_config_op),
    path="/api/config/reload",
    method="POST",
)
