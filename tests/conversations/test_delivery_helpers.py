"""Pure delivery-executor helpers: long-answer splitting, the signed-callback digest, the
retry backoff, and canonical address forms."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from tai42_skeleton.conversations.address import canonical_address
from tai42_skeleton.conversations.delivery import _backoff_seconds, _sign, split_message
from tai42_skeleton.conversations.settings import ConversationsSettings


def test_split_short_message_is_one_chunk():
    assert split_message("hello", 100) == ["hello"]


def test_split_preserves_and_reorders_nothing():
    text = "word " * 100  # 500 chars
    chunks = split_message(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks) == text
    assert len(chunks) > 1


def test_split_hard_cuts_an_unbreakable_run():
    text = "x" * 90
    chunks = split_message(text, 40)
    assert chunks == ["x" * 40, "x" * 40, "x" * 10]
    assert "".join(chunks) == text


def test_split_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="must be positive"):
        split_message("hi", 0)


def test_sign_is_hmac_sha256_hex_with_prefix():
    body = b'{"answer":"hi"}'
    signature = _sign("s3cr3t", body)
    assert signature.startswith("sha256=")
    expected = hmac.new(b"s3cr3t", body, hashlib.sha256).hexdigest()
    assert signature == f"sha256={expected}"


def test_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_BASE_SECONDS", "8")
    monkeypatch.setenv("CONVERSATIONS_DELIVERY_BACKOFF_MAX_SECONDS", "900")
    settings = ConversationsSettings()
    assert _backoff_seconds(settings, 1) == 8
    assert _backoff_seconds(settings, 2) == 16
    assert _backoff_seconds(settings, 3) == 32
    # Capped.
    assert _backoff_seconds(settings, 20) == 900


def test_canonical_address_trims_and_rejects_blank():
    assert canonical_address("  +1555  ") == "+1555"
    with pytest.raises(ValueError, match="non-blank"):
        canonical_address("   ")
