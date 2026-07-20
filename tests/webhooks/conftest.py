"""Webhook test wiring.

The builtin ``shared_secret`` module registers through the ``tai_app`` contract
handle at import time (exactly as external verifier plugins do). Test modules
import it at collection, so bind the process app singleton first.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.app import instance

tai_app.bind(instance.build_app())
