"""Marketplace client, installer, attribution store, and advisory polling.

This package lets the running server browse the marketplace registry, install
and uninstall registry-published plugins into its own interpreter environment,
and keep a local record of what it installed.

Its pieces:

- :mod:`.settings` — the ``MARKETPLACE_*`` registry endpoint / advisory-poll
  knobs and the ``MARKETPLACE_STORE_*`` Postgres connection for the attribution
  table.
- :mod:`.errors` — the typed failure hierarchy raised inside the client and
  installer, translated to the operation-layer error vocabulary at the boundary.
- :mod:`.client` — the thin async client over the registry's public read API.
- :mod:`.installer` — the resolve/install/uninstall/update flows with
  abort-and-unwind and the fleet-wide advisory lock.
- :mod:`.store` — the Postgres attribution store recording each installed plugin.
- :mod:`.manifest_patch` — the pure patch/unpatch functions that wire a plugin's
  ``provides`` index into the manifest.
- :mod:`.pip` — the ``sys.executable -m pip`` subprocess seam: pre-flights, the
  runner, and local pin-argv composition.
- :mod:`.advisories` — the installed-plugin advisory cache, its on-demand refresh,
  and the documented background poll.

No re-exports: consumers import the concrete symbol from its module.
"""

from __future__ import annotations
