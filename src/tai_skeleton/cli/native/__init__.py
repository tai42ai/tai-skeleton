"""CLI-native commands — local operations with no ``/api/*`` counterpart.

The runtime launchers (``serve``/``backend``/``metrics``) are the re-homed
click commands in the sibling launcher modules; the modules here carry the
database, diagnostics, catalog, OpenAPI, completion, and version commands.
"""
