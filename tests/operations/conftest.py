"""Isolate the process-global registries around each operations test.

Adapter/spec tests record routes and operations into the process-wide
``route_registry`` / ``operation_registry``; snapshot and restore both so a
fixture route can never leak into the product OpenAPI spec (the byte-identical
pin) or into another test.

The registry has a PAIRED module-global — ``operations._leaf_snapshot``, the
one-time capture ``reregister_operations`` primes on its first call and replays
on every later call. It is shared state the registry must stay consistent with: a
``reregister_operations()`` replay re-adds the snapshot's records, and the
duplicate-name guard rejects them if the registry still holds the differently
identified collection-time records. So rebuild the registry from a fresh
``clear()`` + ``reregister_operations()`` before every test — exactly what
``start()`` does at boot — leaving each test the primed, consistent surface a
full-app boot gives the canonical whole-suite run. Without it, a scoped run that
never boots a full app leaves the registry on its collection-time records while an
earlier operations test (one that DID boot) advances the snapshot, and the next
replay trips the guard (``already registered``).

An operation-declaration module can register a lifecycle handler at import
(``operations.tool_runs`` wires the supervisor-drain shutdown handler), which
needs the process app singleton bound — exactly as the runtime binds it before
importing operation/router modules. Mirror that order here so op modules import
at collection, as ``tests/routers/conftest`` does for the routers.
"""

from __future__ import annotations

import pytest
from tai_contract.app import tai_app

from tai_skeleton.app import instance
from tai_skeleton.app.route_registry import route_registry
from tai_skeleton.operations import reregister_operations
from tai_skeleton.operations.registry import operation_registry

tai_app.bind(instance.build_app())


@pytest.fixture(autouse=True)
def _isolate_registries():
    # Rebuild the operation registry from its leaf snapshot before every test,
    # exactly as ``start()`` does at boot, so each test opens on the primed,
    # consistent surface (registry records == snapshot records) regardless of what
    # an earlier operations test left behind. The first call re-imports the leaves
    # and captures the snapshot; every later call replays it (no ``sys.modules``
    # churn), and the ``clear()`` keeps that replay off the duplicate-name guard.
    operation_registry.clear()
    reregister_operations()

    routes_snapshot = dict(route_registry._routes)
    ops_snapshot = dict(operation_registry._operations)
    try:
        yield
    finally:
        route_registry._routes = routes_snapshot
        operation_registry._operations = ops_snapshot
