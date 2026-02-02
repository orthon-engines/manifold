"""
PRISM Engines — DEPRECATED
===========================

This module is a backward-compatibility redirect.
Use the pipeline-stage modules directly:

    prism.signal_vector.signal    — per-signal engines
    prism.signal_vector.rolling   — rolling window engines
    prism.signal_vector.sql       — SQL engines
    prism.state_vector            — state + geometry
    prism.geometry_pairwise       — pair engines
    prism.geometry_laplace        — dynamics engines

This redirect will be removed in a future version.
"""

import warnings

warnings.warn(
    "prism.engines is deprecated. "
    "Use prism.signal_vector, prism.state_vector, "
    "prism.geometry_pairwise, prism.geometry_laplace instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    """Redirect old imports to new locations."""
    if name == 'signal':
        from prism.signal_vector import signal
        return signal
    elif name == 'rolling':
        from prism.signal_vector import rolling
        return rolling
    elif name == 'sql':
        from prism.signal_vector import sql
        return sql
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['signal', 'rolling', 'sql']
