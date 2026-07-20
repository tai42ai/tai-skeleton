"""Media content types produced and consumed by the :class:`ResourceManager`.

``MediaBlock`` is the model-side media union returned by
:meth:`ResourceManager.load_file` and the media tools — fastmcp already knows how
to turn each member into MCP content (``ImageContent`` / ``AudioContent`` /
``EmbeddedResource``). ``ContentPart`` is the LangChain content-part dict emitted
by :meth:`ResourceManager.normalize_media` for PRE-run model input (e.g. VqaAgent).
The two are distinct: ``MediaBlock`` is a tool RETURN, ``ContentPart`` is model
INPUT.
"""

from __future__ import annotations

from typing import TypedDict

from fastmcp.utilities.types import Audio, File, Image

# The fastmcp media union a loaded resource decodes into. fastmcp serializes each
# member to MCP content via ``.to_image_content()`` / ``.to_audio_content()`` /
# ``.to_resource_content()``.
MediaBlock = Image | Audio | File


class _ImageUrl(TypedDict):
    url: str


class ContentPart(TypedDict):
    """A LangChain ``image_url`` content part — a public URL or a base64 ``data:``
    URI the model dereferences. The ``image_url`` shape is image-only at the model
    boundary."""

    type: str
    image_url: _ImageUrl


__all__ = ["ContentPart", "MediaBlock"]
