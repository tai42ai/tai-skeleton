"""Fixture registering a webhook verifier ON IMPORT.

Loaded via a manifest ``webhook_verifier_modules`` entry so each ``start()``
re-imports it and re-runs the ``tai42_app.webhook_verifiers.register(...)``
side-effect — exactly as a real verifier plugin module does. The registry is
reset each ``start()``, so the repeated registration is clean, never a
duplicate-name crash.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tai42_contract.app import tai42_app


class _FixtureVerifier:
    async def verify(self, body: bytes, headers: Mapping[str, str], config: dict[str, Any]) -> None:
        return None


tai42_app.webhook_verifiers.register("fixture_verifier", _FixtureVerifier())
