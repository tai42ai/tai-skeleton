"""Skeleton builtin webhook verifiers.

Each module registers a :class:`~tai42_contract.webhooks.WebhookVerifier` through
the app's ``webhook_verifiers`` facet; a manifest ``lifecycle_modules`` entry
loads it (importing the module runs its registration). Discovery is by module
path — nothing is re-exported here.
"""
