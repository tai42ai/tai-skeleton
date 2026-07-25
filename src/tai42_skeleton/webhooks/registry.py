"""The process-wide webhook-verifier registry — the body behind
``app.webhook_verifiers``.

A verifier plugin registers under a name via an import-only
``webhook_verifier_modules`` manifest entry (importing the module runs its
``tai42_app.webhook_verifiers.register(...)`` call). A public webhook door
resolves a bound verifier by name at bind time.

The registry is reset on every ``start()`` (like the agent binding) so a reload
re-imports the verifier modules and re-registers cleanly; a duplicate name
within one load raises loudly (a silent overwrite could swap a topic's verifier
out from under a live binding).
"""

from __future__ import annotations

from tai42_contract.webhooks import WebhookVerifier


class WebhookVerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, WebhookVerifier] = {}

    def register(self, name: str, verifier: WebhookVerifier) -> None:
        if name in self._verifiers:
            raise ValueError(f"webhook verifier {name!r} is already registered")
        self._verifiers[name] = verifier

    def get(self, name: str) -> WebhookVerifier:
        try:
            return self._verifiers[name]
        except KeyError:
            raise KeyError(f"unknown webhook verifier {name!r} (registered: {sorted(self._verifiers)})") from None

    def names(self) -> list[str]:
        return sorted(self._verifiers)

    def reset(self) -> None:
        self._verifiers.clear()
