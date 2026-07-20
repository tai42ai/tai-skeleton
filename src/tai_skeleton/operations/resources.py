"""Resource operations — load a stored resource by id (``/api/resources/*``).

``get_resource_by_id`` loads a stored resource by its id/URL and optionally renders
it as a Jinja template, returning the loaded (and optionally rendered) content —
text or a :data:`~tai_skeleton.template.media.MediaBlock`. It is the op behind
``POST /api/resources/get`` and the ``tai resources get`` CLI — a thin skin over
the app's resource manager (``load_file`` + ``render_by_id_or_content``).

A READ door: it never mutates the store, so it is NOT destructive and stays outside
the admin-mutation deny-fence.
"""

from __future__ import annotations

from typing import Any

from jinja2 import TemplateError
from pydantic import BaseModel
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, NotFoundError, operation
from tai_skeleton.template import TemplateNotFoundError
from tai_skeleton.template.media import MediaBlock
from tai_skeleton.template.path_guard import UnsafeTemplatePathError


class ResourceGet(BaseModel):
    """Load a stored resource by ``resource_id``, optionally rendering it.

    When ``template_kwargs`` is omitted the loaded content is returned as-is (a
    template/text resource as its text, a document as extracted text, media as a
    ``MediaBlock``). When provided (any object, including ``{}``) the caller intends
    a render: text is rendered as a Jinja template with these variables; media
    cannot be rendered and is refused with a ``400``.
    """

    resource_id: str
    template_kwargs: dict[str, Any] | None = None


@operation(
    summary="Load a stored resource by id, optionally rendering it",
    tags=["resources"],
    errors=[BadRequestError, NotFoundError],
    request_model=ResourceGet,
)
async def get_resource_by_id(resource_id: str, template_kwargs: dict[str, Any] | None = None) -> str | MediaBlock:
    """Load a stored resource by its id, optionally rendering it as a template.

    Args:
        resource_id: The id (path) of the resource to load.
        template_kwargs: When omitted, the resource's loaded content is returned
            as-is. When provided (any dict, incl. ``{}``), text is rendered as a
            Jinja template with these variables; media raises loudly (``400``).

    Returns:
        The loaded (and optionally rendered) content — text or a ``MediaBlock``.

    A missing resource is a ``404``; a traversal-escaping id, a render of media, or
    broken client-supplied Jinja is a ``400``. Genuine storage/transport failures
    raise other types and propagate as a ``500``.
    """
    manager = tai_app.storage.resource_manager
    try:
        content = await manager.load_file(resource_id)
    except UnsafeTemplatePathError as exc:
        raise BadRequestError(str(exc)) from exc
    except (FileNotFoundError, TemplateNotFoundError) as exc:
        raise NotFoundError(f"resource {resource_id!r} not found") from exc
    if template_kwargs is None:
        return content
    if isinstance(content, MediaBlock):
        raise BadRequestError(f"Cannot render media resource {resource_id!r} with template_kwargs.")
    try:
        return await manager.render_by_id_or_content(content=content, kwargs=template_kwargs)
    except TemplateError as exc:
        raise BadRequestError(f"template error: {exc}") from exc
