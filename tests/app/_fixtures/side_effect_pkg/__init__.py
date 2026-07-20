"""Fixture package whose ``__init__`` has a side effect.

Used to prove ``import_or_reload_package`` imports each module exactly once per
call: the discovery step must not import the package to recurse (that would run
this ``__init__`` a second time — the "already registered" double-register bug).
"""

from tests.app._fixtures import counter_probe

counter_probe.INIT_CALLS.append("side_effect_pkg")
