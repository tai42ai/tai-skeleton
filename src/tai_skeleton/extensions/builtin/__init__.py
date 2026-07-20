"""Skeleton builtin tool extensions.

Tool extensions that ship with the platform and depend on skeleton features
(plugins extend the platform; a tool extension extends a single tool). Each
module registers through the app's ``extensions`` facet, and a manifest
``extensions_modules`` entry loads it — importing the module fires its
registration decorator. Nothing is re-exported here; discovery is by module
path, not by importing this package.
"""
