"""Router test wiring.

The router modules register their routes through the ``tai_app`` contract
handle at import time, exactly as external plugins do; the runtime imports them
only after ``start()`` binds the handle. Test modules import the routers at
collection, so mirror that order here: bind the process app singleton before
the router test modules are imported.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.app import instance

tai_app.bind(instance.build_app())
