"""The ``file_loader`` builtin: load a resource from a url OR a storage id and
return its text content or a media block.

A thin shim over the app's ``resource_manager`` — it delegates to
:meth:`ResourceManager.load_file`, which resolves the source (SSRF-pinned url
fetch, data URI, or storage read), detects the type with Magika, and extracts
text or passes media through as a ``MediaBlock``. The type-detection deps are the
opt-in ``files`` extra: ``pip install 'tai42-skeleton[files]'``.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.template.media import MediaBlock


@tai42_app.tools.tool
async def file_loader(source: str) -> str | MediaBlock:
    """Load a file from a url or a storage resource id and return its content.

    Args:
        source: A ``http(s)`` url, a ``data:`` URI, or a bare storage resource id.

    Returns:
        The extracted text (PDF, DOCX, XLSX, CSV, HTML, and more) or a media block
        (image/audio).
    """
    return await tai42_app.storage.resource_manager.load_file(source)
