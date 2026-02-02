"""
PRISM State Vector — Stage 2
=============================

System-level state derived from signal features.
Reads: signal_vector.parquet
Writes: state_vector.parquet, state_geometry.parquet, signal_geometry.parquet

Modules:
    state_vector.py     — Centroid computation (WHERE the system is)
    state_geometry.py   — SVD → eigenvalues (SHAPE of the signal cloud)
    signal_geometry.py  — Per-signal distance to centroid

KEY SEPARATION:
    state_vector  = centroid (mean position in feature space)
    state_geometry = eigenvalues from SVD (shape of signal distribution)

    Eigenvalues ONLY in state_geometry.py. Never in state_vector.py.
    This separation is architectural — do not merge them.
"""

from .state_vector import compute_state_vector
from .state_geometry import compute_state_geometry
from .signal_geometry import compute_signal_geometry

__all__ = [
    'compute_state_vector',
    'compute_state_geometry',
    'compute_signal_geometry',
]
