"""AES-GCM-256 wrap/unwrap for connector token blobs.

Blob layout: [1-byte format version] || [12-byte nonce] || [ciphertext + 16-byte
GCM tag]. The leading version byte (``0x01``) makes KEK rotation possible without
re-reading a headerless blob: encryption always uses the current ``CONNECTORS_KEK``,
while decryption tries the whole key-ring (current + optional
``CONNECTORS_KEK_PREVIOUS``). The connection_id is bound as AAD so a blob cannot be
swapped between connections.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tai42_contract.connectors.errors import ConnectorError

from tai42_skeleton.connectors.settings import connector_crypto_secrets

_KEK_FORMAT_VERSION = 0x01
_NONCE_LEN = 12
_TAG_LEN = 16


class ConnectorEncryptionConfigError(ConnectorError):
    """Raised when CONNECTORS_KEK is missing or malformed at use time."""


def ensure_kek() -> bytes:
    """Return the current KEK (the encryption key), or raise a config error."""
    try:
        return connector_crypto_secrets().require_kek_bytes()
    except (RuntimeError, ValueError) as exc:
        raise ConnectorEncryptionConfigError(str(exc)) from exc


def _decrypt_ring() -> list[bytes]:
    """The decryption key-ring: current KEK first, then the optional previous KEK."""
    try:
        return connector_crypto_secrets().kek_ring_bytes()
    except (RuntimeError, ValueError) as exc:
        raise ConnectorEncryptionConfigError(str(exc)) from exc


def _aad(connection_id: str) -> bytes:
    return connection_id.encode("ascii")


def encrypt(plaintext: bytes, *, connection_id: str) -> bytes:
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("plaintext must be bytes")
    kek = ensure_kek()
    nonce = os.urandom(_NONCE_LEN)
    ct = AESGCM(kek).encrypt(nonce, bytes(plaintext), _aad(connection_id))
    return bytes([_KEK_FORMAT_VERSION]) + nonce + ct


def decrypt(blob: bytes, *, connection_id: str) -> bytes:
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("blob must be bytes")
    blob = bytes(blob)
    if len(blob) < 1 + _NONCE_LEN + _TAG_LEN:
        raise ValueError("blob too short to contain version+nonce+tag")
    version = blob[0]
    if version != _KEK_FORMAT_VERSION:
        raise ValueError(f"unsupported connector token-blob format version byte: {version:#04x}")
    nonce, ct = blob[1 : 1 + _NONCE_LEN], blob[1 + _NONCE_LEN :]
    aad = _aad(connection_id)
    # Try every key in the ring so a blob written under the previous KEK still
    # decrypts across a rotation. The final InvalidTag propagates if none match; an
    # empty ring is a config fault (the settings guarantee at least the current KEK)
    # and raises loudly rather than returning a silently-undecrypted blob.
    last_error: InvalidTag | None = None
    for kek in _decrypt_ring():
        try:
            return AESGCM(kek).decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            last_error = exc
    if last_error is None:
        raise ConnectorEncryptionConfigError("decryption key-ring is empty; no KEK to decrypt the token blob")
    raise last_error
