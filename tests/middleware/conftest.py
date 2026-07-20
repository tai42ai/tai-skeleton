"""Middleware test wiring.

The middleware modules register through the ``tai42_app`` contract handle at import
time (``@tai42_app.http.middleware``), exactly as external middleware plugins do.
Test modules import them at collection, so bind the process app singleton first.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance

tai42_app.bind(instance.build_app())
