"""Connector provider registry — the engine's in-memory provider catalog.

The skeleton ships NO concrete provider. Registration is manifest-driven: a
provider plugin module (named in the manifest) calls
``tai42_app.connectors.register_connector(descriptor)`` on import, which forwards
to :func:`tai42_skeleton.connectors.providers.registry.register_connector`.
"""
