"""
PRISM Geometry Pairwise — Stage 3
==================================

Signal-to-signal relationships.
Reads: signal_vector.parquet + state_vector.parquet
Writes: signal_pairwise.parquet

Modules:
    signal_pairwise.py  — Pairwise computation orchestrator
    granger.py          — Directional: Granger causality (A → B)
    transfer_entropy.py — Directional: information flow (A → B)
    correlation.py      — Symmetric: linear relationship
    mutual_info.py      — Symmetric: nonlinear relationship
    cointegration.py    — Symmetric: long-run equilibrium

Pair engines moved here from engines/signal/ because they operate
BETWEEN signals, not on individual signals. They belong in the
pairwise stage of the pipeline, not the signal vector stage.

N signals → N(N-1)/2 unique pairs per index.
Directional pairs: granger, transfer_entropy (A→B ≠ B→A)
Symmetric pairs: correlation, mutual_info, cointegration (A↔B = B↔A)
"""

from .signal_pairwise import compute_signal_pairwise

__all__ = [
    'compute_signal_pairwise',
    'granger',
    'transfer_entropy',
    'correlation',
    'mutual_info',
    'cointegration',
]
