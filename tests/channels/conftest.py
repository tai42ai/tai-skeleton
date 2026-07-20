"""Channel test wiring.

Channel plugins and the channels router register through the ``tai42_app``
contract handle at import time; the runtime imports them only after ``start()``
binds the handle. Test modules import them at collection, so bind the process
app singleton first.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance

tai42_app.bind(instance.build_app())
