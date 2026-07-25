"""Extension test wiring.

The builtin extension modules register through the ``tai42_app`` contract handle
at import time (their ``@tai42_app.extensions.extension`` decorator), exactly as
external plugins do. Test modules import them at collection, so bind the process
app singleton before those imports run — mirroring ``tests/routers/conftest.py``.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance

tai42_app.bind(instance.build_app())
