"""
PRISM Signal Vector — Stage 1
==============================

Per-signal features computed at each index I.
Reads: observations.parquet + manifest.yaml
Writes: signal_vector.parquet

Subdirectories:
    signal/   — Per-signal engines (kurtosis, spectral, harmonics, etc.)
    rolling/  — Rolling window engines (rolling_kurtosis, rolling_volatility, etc.)
    sql/      — SQL-based engines (zscore, statistics)

Each engine computes ONE thing. No domain prefixes.
Only scale-invariant engines are active. Deprecated engines (rms, peak, mean,
std) remain for backward compatibility but should not be used.

The manifest determines which engines run for each signal.
PRISM does not decide — it executes.
"""

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
