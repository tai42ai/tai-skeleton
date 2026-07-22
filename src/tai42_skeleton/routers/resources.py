"""Resources HTTP surface — ``/api/resources/*`` (AUTHED).

An AUTHED thin adapter over the operation in ``tai42_skeleton.operations.resources``.
The single ``get_resource_by_id`` operation backs both methods of ``/api/resources/get``:

- ``GET /api/resources/get?resource_id=...`` — the plain fetch-as-is door. It reads
  ``resource_id`` from the query string, never renders (no ``template_kwargs``), and is
  ``action="read"``, so a role with the ``resources`` READ level can fetch a stored
  resource.
- ``POST /api/resources/get`` — the render door. Its ``template_kwargs`` is arbitrary
  nested JSON that legitimately needs a request body, so it is a write-classed POST;
  the READ level does not open it.

Both methods are registered over the SAME operation. The GET is registered FIRST so the
shared operation metadata's ``http_method`` settles on the POST (the render door) for
the projected MCP tool's tool-edge authorization.

Success bodies are ``{"data": ...}`` (a text string or a media wire block); failures
are ``{"error": "<message>"}``. A missing resource is a ``404``; a traversal-escaping
id, a render of media, or broken Jinja is a ``400``. Both methods only READ — they
never mutate the store, so the operation is not destructive.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.resources import get_resource_by_id as _get_resource_by_id_op

# The plain fetch-as-is door: ``GET /api/resources/get?resource_id=...``. The adapter
# parses the flat ``resource_id`` from the query string (a GET never reads a body), so
# ``template_kwargs`` is always ``None`` and the resource is returned as-is. Registered
# BEFORE the POST so the shared operation metadata's ``http_method`` ends on the POST.
fetch_resource = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_resource_by_id_op),
    path="/api/resources/get",
    method="GET",
    action="read",
)

# The render door: ``POST /api/resources/get``. Its ``template_kwargs`` is arbitrary
# nested JSON that needs a request body, so it is a write-classed POST.
get_resource_by_id = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_resource_by_id_op),
    path="/api/resources/get",
    method="POST",
    action="write",
)
