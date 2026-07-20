"""Extension test wiring.

The builtin extension modules register through the ``tai_app`` contract handle
at import time (their ``@tai_app.extensions.extension`` decorator), exactly as
external plugins do. Test modules import them at collection, so bind the process
app singleton before those imports run — mirroring ``tests/routers/conftest.py``.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.app import instance

tai_app.bind(instance.build_app())
