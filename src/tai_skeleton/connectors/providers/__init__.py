"""Connector provider registry — the engine's in-memory provider catalog.

The skeleton ships NO concrete provider. Registration is manifest-driven: a
provider plugin module (named in the manifest) calls
``tai_app.connectors.register_connector(descriptor)`` on import, which forwards
to :func:`tai_skeleton.connectors.providers.registry.register_connector`.
"""
