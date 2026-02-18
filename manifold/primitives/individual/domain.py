"""Shim: re-exports from pmtvs. See rudder-framework/primitives."""
try:
    from pmtvs.individual.domain import *  # noqa: F401,F403
except ImportError:
    pass  # Module not yet migrated to pmtvs
