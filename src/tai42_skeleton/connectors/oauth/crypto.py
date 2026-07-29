"""AES-GCM-256 wrap/unwrap for connector token blobs.

Blob layout: [1-byte format version] || [12-byte nonce] || [ciphertext + 16-byte
GCM tag]. The leading version byte (``0x01``) names the blob format version, so a
reader detects a blob written in a different format instead of misreading it.
Encryption and decryption both use ``CONNECTORS_KEK``. The connection_id is bound as
AAD so a blob cannot be swapped between connections.
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tai42_contract.connectors.errors import ConnectorError

from tai42_skeleton.connectors.settings import connector_crypto_secrets

_KEK_FORMAT_VERSION = 0x01
_NONCE_LEN = 12
_TAG_LEN = 16


class ConnectorEncryptionConfigError(ConnectorError):
    """Raised when CONNECTORS_KEK is missing or malformed at use time."""


def ensure_kek() -> bytes:
    """Return the KEK used to both encrypt and decrypt blobs, or raise a config error."""
    try:
        return connector_crypto_secrets().require_kek_bytes()
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
    # A blob the KEK cannot open is unreadable: the ``InvalidTag`` propagates
    # loudly rather than returning a silently-undecrypted blob.
    return AESGCM(ensure_kek()).decrypt(nonce, ct, _aad(connection_id))
