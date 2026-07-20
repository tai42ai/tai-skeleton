"""Templates operations — the Studio templates surface (``/api/*template*``).

A thin skin over the app's resource manager (``tai42_app.storage.resource_manager``):

* ``list_templates`` — the stored template ids/paths.
* ``get_template`` — one template's content plus its inferred input schema.
* ``upload_template`` — write (create or overwrite) a template.
* ``delete_template`` — remove a stored template.
* ``render_template`` — render inline content OR a stored template with kwargs.
* ``clear_templates_cache`` — drop the compiled-template cache.

**Path-argument hardening:** every logical template key that reaches the store —
the upload/delete ``path`` and the render/fetch ``template_id`` — runs through the
shared lexical containment guard (:func:`safe_template_path`) INSIDE the operation,
so the guard defends the MCP tool edge and the CLI as well as the HTTP route: a
``..`` escape, an absolute key, an embedded backslash/NUL, or an empty key is
refused loudly (a ``400`` on the route, a ``ToolError`` on the tool). Upload/delete
are arbitrary-file-write primitives and render/fetch by id are arbitrary-file-read
primitives; the guard is their first line, defense in depth over the store's own
root defense.

``upload_template`` and ``delete_template`` mutate the store, so both are
``destructive``.
"""

from __future__ import annotations

from typing import Any

from jinja2 import TemplateError
from pydantic import BaseModel
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, NotFoundError, operation
from tai42_skeleton.template import TemplateNotFoundError
from tai42_skeleton.template.path_guard import UnsafeTemplatePathError, safe_template_path


class TemplateFetch(BaseModel):
    """Fetch a template's content and its input schema by id."""

    template_id: str


class TemplateUpload(BaseModel):
    """Upload (create or overwrite) a template at ``path``."""

    path: str
    content: str


class TemplateDelete(BaseModel):
    """Delete the template at ``path``."""

    path: str


class TemplateRender(BaseModel):
    """Render a template — inline ``content`` OR a stored ``template_id`` — with
    ``kwargs``."""

    content: str | None = None
    template_id: str | None = None
    kwargs: dict[str, object] = {}


def _safe_key(key: object) -> str:
    """Guard a caller-supplied template key, mapping an escape to a loud ``400``."""
    try:
        return safe_template_path(key)
    except UnsafeTemplatePathError as exc:
        raise BadRequestError(str(exc)) from exc


@operation(summary="List templates", tags=["templates"])
async def list_templates() -> list[str]:
    """List the stored template ids/paths from the active storage provider."""
    return await tai42_app.storage.resource_manager.list_resources()


@operation(
    summary="Fetch a template and its schema",
    tags=["templates"],
    errors=[BadRequestError, NotFoundError],
    request_model=TemplateFetch,
)
async def get_template(template_id: str) -> dict:
    """Return a stored template's content and its inferred input schema.

    A missing stored template is a ``404`` (never a leaked storage ``500``); a
    stored template with broken Jinja is author error — the schema inference parses
    it — surfaced as a ``400``. Genuine storage failures raise other types (``500``).
    """
    # A field-specific 400 for a blank/absent id, ahead of the lexical path guard, so
    # the message names ``template_id`` rather than the guard's generic ``path``.
    if not isinstance(template_id, str) or not template_id:
        raise BadRequestError("template_id must be a non-empty string")
    key = _safe_key(template_id)
    manager = tai42_app.storage.resource_manager
    try:
        content = await manager.fetch_template(key)
        schema = await manager.get_template_schema(template_id=key)
    except TemplateNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except TemplateError as exc:
        raise BadRequestError(f"template error: {exc}") from exc
    return {"template": content, "schema": schema}


@operation(
    summary="Upload a template",
    tags=["templates"],
    destructive=True,
    errors=[BadRequestError],
    request_model=TemplateUpload,
)
async def upload_template(path: str, content: str) -> dict:
    """Write ``content`` to the template store under ``path`` (create or overwrite).

    The write target is guarded against a root escape before it reaches the store.
    """
    key = _safe_key(path)
    if not isinstance(content, str):
        raise BadRequestError("content must be a string")
    await tai42_app.storage.resource_manager.upload_template(path=key, content=content)
    return {"path": key, "uploaded": True}


@operation(
    summary="Delete a template",
    tags=["templates"],
    destructive=True,
    errors=[BadRequestError],
    request_model=TemplateDelete,
)
async def delete_template(path: str) -> dict:
    """Delete the stored template at ``path``.

    Idempotent: an already-absent path is a no-op success (the store treats a
    missing template as a no-op rather than raising), so this returns ``200``.
    """
    key = _safe_key(path)
    await tai42_app.storage.resource_manager.delete_template(key)
    return {"path": key, "deleted": True}


@operation(
    summary="Render a template",
    tags=["templates"],
    errors=[BadRequestError, NotFoundError],
    request_model=TemplateRender,
)
async def render_template(
    content: str | None = None,
    template_id: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> dict:
    """Render an inline ``content`` OR a stored ``template_id`` with ``kwargs``.

    Exactly one of ``content``/``template_id`` is required. A missing stored
    template is a ``404``; broken client-supplied Jinja (a syntax error, a
    sandbox-blocked dunder traversal → ``SecurityError``, an undefined access) is
    author error → ``400``. Genuine storage failures raise other types (``500``).
    """
    kwargs = kwargs or {}
    if content is None and template_id is None:
        raise BadRequestError("one of 'content' or 'template_id' is required")
    if content is not None and template_id is not None:
        raise BadRequestError("provide either 'content' or 'template_id', not both")
    if content is not None and not isinstance(content, str):
        raise BadRequestError("'content' must be a string")
    if not isinstance(kwargs, dict):
        raise BadRequestError("'kwargs' must be a JSON object")
    if template_id is not None:
        template_id = _safe_key(template_id)
    try:
        rendered = await tai42_app.storage.resource_manager.render_by_id_or_content(
            content=content,
            template_id=template_id,
            kwargs=kwargs,
        )
    except TemplateNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except TemplateError as exc:
        raise BadRequestError(f"template error: {exc}") from exc
    return {"rendered": rendered}


@operation(summary="Clear the template render cache", tags=["templates"])
async def clear_templates_cache() -> dict:
    """Drop every compiled template from the render cache."""
    tai42_app.storage.resource_manager.clear_cache()
    return {"cleared": True}
