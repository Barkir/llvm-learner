"""
Importing this package registers the built-in feature sets.

To add your own, drop a module here and import it below (or import it from your
own code before building an Extractor — registration happens at class creation).
"""

from . import core, memory, dialects      # noqa: F401  (import = register)

__all__ = ['core', 'memory', 'dialects']
