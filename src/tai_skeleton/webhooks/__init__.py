"""Webhook-security surface: the verifier registry, per-topic bindings, ingress
settings, and the builtin ``shared_secret`` verifier.

The public webhook doors (``/universal_webhook/{topic}`` and the interactions
callback) authenticate an inbound request over its raw bytes BEFORE parsing, via
a named :class:`~tai_contract.webhooks.WebhookVerifier` resolved from the
registry here.
"""

from __future__ import annotations

from tai_skeleton.webhooks.registry import WebhookVerifierRegistry

__all__ = ["WebhookVerifierRegistry"]
