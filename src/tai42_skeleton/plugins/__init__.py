"""Studio-plugin registry + host-side serving support.

A *Studio plugin* is the browser bundle a platform (Python) plugin ships under
``<package>/studio/``. This package validates those bundles at startup/reload,
builds the registry the SPA host reads, and supplies the import-map + asset
serving primitives the ``routers/plugins.py`` HTTP surface exposes.
"""
