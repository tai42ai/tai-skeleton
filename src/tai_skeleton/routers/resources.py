"""Resources HTTP surface — ``/api/resources/*`` (AUTHED).

An AUTHED thin adapter over the operation in ``tai_skeleton.operations.resources``:

- ``POST /api/resources/get`` — load a stored resource by id/URL and optionally
  render it as a Jinja template; returns text or a media block.

Success bodies are ``{"data": ...}`` (a text string or a media wire block); failures
are ``{"error": "<message>"}``. A missing resource is a ``404``; a traversal-escaping
id, a render of media, or broken Jinja is a ``400``. A READ door — it never mutates
the store, so the operation is not destructive.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.operations import operation_metadata_of, register_operation_route
from tai_skeleton.operations.resources import get_resource_by_id as _get_resource_by_id_op

get_resource_by_id = register_operation_route(
    tai_app,
    operation_metadata_of(_get_resource_by_id_op),
    path="/api/resources/get",
    method="POST",
)
