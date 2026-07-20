"""Middleware test wiring.

The middleware modules register through the ``tai_app`` contract handle at import
time (``@tai_app.http.middleware``), exactly as external middleware plugins do.
Test modules import them at collection, so bind the process app singleton first.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.app import instance

tai_app.bind(instance.build_app())
