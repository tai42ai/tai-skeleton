"""Fixture tools returning fastmcp media blocks, for the direct-run
``run_tool`` serialization path: an ``Image`` and an ``Audio`` become the media
wire dict, a ``File`` falls through to a JSON ``EmbeddedResource``, and a plain
text tool is left untouched."""

from fastmcp.utilities.types import Audio, File, Image
from tai42_contract.app import tai42_app

# A minimal PNG header — enough bytes to base64-encode; the tool never decodes it.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n"
_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVE"
_FILE_BYTES = b"arbitrary blob"


@tai42_app.tools.tool
def make_image() -> Image:
    """Return an image media block."""
    return Image(data=_PNG_BYTES, format="png")


@tai42_app.tools.tool
def make_audio() -> Audio:
    """Return an audio media block."""
    return Audio(data=_WAV_BYTES, format="wav")


@tai42_app.tools.tool
def make_file() -> File:
    """Return a file media block."""
    return File(data=_FILE_BYTES, format="bin")


@tai42_app.tools.tool
def make_text() -> str:
    """Return plain text."""
    return "just text"
