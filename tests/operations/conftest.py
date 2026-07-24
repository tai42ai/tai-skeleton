"""Isolate the process-global registries around each operations test.

Adapter/spec tests record routes and operations into the process-wide
``route_registry`` / ``operation_registry``; isolate both so a fixture route can
never leak into the product OpenAPI spec (the byte-identical pin) or into another
test. Routes are cleaned by TARGETED DELETION of the rows each test added (a
whole-snapshot restore would drop a route another suite recorded once at import);
operations are rebuilt from their leaf snapshot before every test, as below.

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

The handle is process-global, so the fixture below binds the singleton itself SCOPED
to the test and restores whatever it found.
"""

from __future__ import annotations

import pytest
from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.operations import reregister_operations
from tai42_skeleton.operations.registry import operation_registry

tai42_app.bind(instance.build_app())


@pytest.fixture(autouse=True)
def _isolate_registries():
    # The app singleton is bound for the whole test; scoped, so another suite's binding
    # is restored on teardown rather than erased.
    with tai42_app.bound(instance.build_app()):
        # Rebuild the operation registry from its leaf snapshot before every test, as
        # ``start()`` does at boot, so each test opens on the primed surface. ``clear()``
        # keeps the replay off the duplicate-name guard.
        operation_registry.clear()
        reregister_operations()

        # Routes restore by TARGETED DELETION of the rows this test added, never a
        # whole-snapshot restore: a router records its routes once per process, so
        # reinstating a snapshot would drop another suite's routes with nothing to
        # re-record them.
        routes_before = set(route_registry._routes)
        ops_snapshot = dict(operation_registry._operations)
        try:
            yield
        finally:
            for key in set(route_registry._routes) - routes_before:
                del route_registry._routes[key]
            operation_registry._operations = ops_snapshot
