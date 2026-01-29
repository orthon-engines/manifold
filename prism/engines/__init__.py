"""
PRISM Engines
=============

Atomic engines organized by execution type:

    signal/   - Signal-level (one value per signal)
    rolling/  - Observation-level (rolling window)
    sql/      - Pure SQL (DuckDB)

Each engine computes ONE thing. No domain prefixes.
Engines compose primitives from prism.primitives.
"""

# Lazy imports to allow direct engine access
__all__ = ['signal', 'rolling', 'sql']


def __getattr__(name):
    """Lazy import of subpackages."""
    if name == 'signal':
        from . import signal
        return signal
    elif name == 'rolling':
        from . import rolling
        return rolling
    elif name == 'sql':
        from . import sql
        return sql
    # Legacy aliases for backwards compatibility
    elif name == 'python':
        import warnings
        warnings.warn(
            "prism.engines.python is deprecated, use prism.engines.signal",
            DeprecationWarning,
            stacklevel=2
        )
        from . import signal
        return signal
    elif name == 'python_windowed':
        import warnings
        warnings.warn(
            "prism.engines.python_windowed is deprecated, use prism.engines.rolling",
            DeprecationWarning,
            stacklevel=2
        )
        from . import rolling
        return rolling
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
