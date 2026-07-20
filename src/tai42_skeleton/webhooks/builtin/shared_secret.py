"""The builtin ``shared_secret`` webhook verifier.

A provider-less core primitive: it checks that a named request header equals a
shared secret, giving any ``universal_webhook`` topic a minimal lock with zero
provider code. Header-based (not a body signature), so it works over any
delivery method — it is NOT ``post_only``.

Config: ``{"header": <header name>, "secret_env": <env var name>}``. The secret
value is NEVER stored in the config — only the NAME of the env var holding it,
resolved at verify time. A missing env var raises loudly (fails CLOSED).

Registered under the name ``shared_secret`` on import; load it with a manifest
``lifecycle_modules`` entry: ``tai42_skeleton.webhooks.builtin.shared_secret``.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from typing import Any

from tai42_contract.app import tai42_app
from tai42_contract.webhooks import WebhookVerificationError


class SharedSecretVerifier:
    """Verifies a named header equals a shared secret via a constant-time compare."""

    # Header-based: the secret rides a header, not a body signature, so any
    # delivery method (POST or GET) is fine.
    post_only = False

    async def verify(self, body: bytes, headers: Mapping[str, str], config: dict[str, Any]) -> None:
        # A missing/malformed config key is an operator misconfiguration of the
        # BINDING (not a request-level failure), so it raises a plain exception ->
        # HTTP 500 (fails CLOSED), distinct from the 401 a bad request signature
        # gets — mirroring the github verifier's ``config[...]`` KeyError.
        header_name = config.get("header")
        secret_env = config.get("secret_env")
        if not isinstance(header_name, str) or not header_name:
            raise ValueError("shared_secret verifier config requires a non-empty 'header'")
        if not isinstance(secret_env, str) or not secret_env:
            raise ValueError("shared_secret verifier config requires a non-empty 'secret_env'")

        # A missing env var is an operator misconfiguration, not a request-level
        # failure: raise loudly (KeyError -> HTTP 500) so the door fails CLOSED
        # rather than silently treating an unresolved secret as a match/mismatch.
        secret = os.environ[secret_env]

        # Case-insensitive header lookup (HTTP header names are case-insensitive;
        # a plain Mapping is case-sensitive, so scan on lowered keys).
        provided: str | None = None
        wanted = header_name.lower()
        for key, value in headers.items():
            if key.lower() == wanted:
                provided = value
                break
        if provided is None:
            raise WebhookVerificationError("shared_secret verification failed")

        # compare_digest is constant-time even for a plain token compare, so a
        # timing side channel can't leak the secret one byte at a time. Compare on
        # UTF-8 bytes: a header value can carry non-ASCII (Starlette decodes headers
        # as latin-1), and compare_digest raises TypeError on non-ASCII ``str`` —
        # encoding first turns that into an ordinary mismatch (401), not a 500.
        if not hmac.compare_digest(provided.encode("utf-8"), secret.encode("utf-8")):
            raise WebhookVerificationError("shared_secret verification failed")


tai42_app.webhook_verifiers.register("shared_secret", SharedSecretVerifier())
