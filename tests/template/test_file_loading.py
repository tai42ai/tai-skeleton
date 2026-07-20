"""The type-detect + dispatch registry: Magika-driven dispatch, bytes->temp-file
materialization for path loaders, the url-suffix fallback, media pass-through, and
the loud raise on an unknown type. Magika is faked so dispatch is deterministic."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastmcp.utilities.types import Audio, Image

from tai_skeleton.template import file_loading as fl


class _FakeMagika:
    def __init__(self, label: str, mime: str, extensions: list[str]) -> None:
        self._output = SimpleNamespace(label=label, mime_type=mime, extensions=extensions)

    def identify_bytes(self, data: bytes) -> SimpleNamespace:
        return SimpleNamespace(output=self._output)


def _fake_magika(label: str, mime: str, extensions: list[str]):
    return lambda: _FakeMagika(label, mime, extensions)


def test_detect_reads_label_mime_and_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("txt", "text/plain", ["txt"]))
    assert fl.detect(b"hello", None) == ("txt", "text/plain", ".txt")


def test_detect_falls_back_to_url_suffix_when_unclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("unknown", "application/octet-stream", []))
    _label, _mime, ext = fl.detect(b"\x00\x01", "https://example.com/report.pdf?token=abc")
    assert ext == ".pdf"


def test_registry_materializes_bytes_to_temp_file_for_path_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, path: str) -> None:
            captured["path"] = path
            captured["exists"] = os.path.exists(path)
            with open(path, "rb") as handle:
                captured["content"] = handle.read()

        def load(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(page_content="PAGE ONE"), SimpleNamespace(page_content="PAGE TWO")]

    monkeypatch.setattr(fl, "PyPDFLoader", _FakeLoader)
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("pdf", "application/pdf", ["pdf"]))

    result = fl.load_content(b"%PDF-1.4 raw", "https://example.com/report.pdf")

    assert result == "PAGE ONE\nPAGE TWO"
    assert captured["exists"] is True  # the loader saw a real file on disk
    assert captured["content"] == b"%PDF-1.4 raw"
    assert str(captured["path"]).endswith(".pdf")
    assert not os.path.exists(str(captured["path"]))  # cleaned up after


def test_registry_honors_url_suffix_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Magika can't classify -> the url suffix (.txt) decides, routing to the
    # plain-text handler which decodes the bytes directly.
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("unknown", "application/octet-stream", []))
    result = fl.load_content(b"plain body text", "https://example.com/notes.txt?token=1")
    assert result == "plain body text"


def test_registry_returns_image_media_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("png", "image/png", ["png"]))
    result = fl.load_content(b"\x89PNG\r\n", "logo.png")
    assert isinstance(result, Image)


def test_registry_returns_audio_media_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("wav", "audio/wav", ["wav"]))
    result = fl.load_content(b"RIFF....WAVE", "clip.wav")
    assert isinstance(result, Audio)


def test_registry_raises_on_unknown_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("unknown", "application/x-weird", []))
    with pytest.raises(ValueError, match="Unsupported file type"):
        fl.load_content(b"\x00\x01\x02", None)


def test_plain_text_strict_decode_raises_on_non_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    # A text-classified but non-utf-8 payload raises loudly rather than corrupting.
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("txt", "text/plain", ["txt"]))
    with pytest.raises(UnicodeDecodeError):
        fl.load_content(b"\xff\xfe not utf-8", "note.txt")


def test_detect_mime_returns_magika_mime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("png", "image/png", ["png"]))
    assert fl.detect_mime(b"\x89PNG") == "image/png"


def test_load_content_rejects_over_cap_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Untrusted document bytes over ``FILE_LOADING_MAX_BYTES`` raise loudly
    BEFORE any detection/parse (a storage-id source bypasses fetch_url's cap)."""
    monkeypatch.setattr(fl, "file_loading_settings", lambda: SimpleNamespace(max_bytes=8))
    # A guard failure means detection never runs; a magika call here would be a bug.
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("txt", "text/plain", ["txt"]))
    with pytest.raises(ValueError, match="FILE_LOADING_MAX_BYTES"):
        fl.load_content(b"way too many bytes", "big.txt")


def test_load_content_allows_at_cap_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bytes exactly at the cap are allowed (the bound is strictly greater-than)."""
    monkeypatch.setattr(fl, "file_loading_settings", lambda: SimpleNamespace(max_bytes=5))
    monkeypatch.setattr(fl, "_magika_client", _fake_magika("txt", "text/plain", ["txt"]))
    assert fl.load_content(b"hello", "ok.txt") == "hello"
