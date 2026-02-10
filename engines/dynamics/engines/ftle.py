"""
FTLE engine -- forward + backward finite-time Lyapunov exponents.

Wraps engines.manifold.dynamics.ftle.compute() to provide a scale-agnostic
interface. The underlying function does not care whether the trajectory comes
from a signal time series, a cohort centroid path, or any other source.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(
    y: np.ndarray,
    direction: str = 'forward',
    min_samples: int = 200,
    method: str = 'rosenstein',
    emb_dim: Optional[int] = None,
    emb_tau: Optional[int] = None,
    dim_method: str = 'cao',
    tau_method: str = 'mutual_info',
) -> Dict[str, Any]:
    """Compute FTLE for a 1-D trajectory. Scale-agnostic.

    Works on signal time series or cohort scalar trajectories.
    For backward FTLE, the time series is reversed before computation.

    Args:
        y:            1-D array of values (ordered by I).
        direction:    'forward' (repelling LCS) or 'backward' (attracting LCS).
        min_samples:  Minimum samples required for reliable FTLE.
        method:       'rosenstein' (default) or 'kantz'.
        emb_dim:      Embedding dimension (auto-detect via Cao's method if None).
        emb_tau:      Embedding delay (auto-detect via mutual info if None).
        dim_method:   'cao' (parameter-free) or 'fnn'.
        tau_method:   'mutual_info', 'autocorr', or 'autocorr_e'.

    Returns:
        Dict with: ftle, ftle_std, embedding_dim, embedding_tau, confidence,
        embedding_dim_method, tau_method, is_deterministic, E1_saturation_dim.
    """
    from engines.manifold.dynamics.ftle import compute as _ftle_compute

    y = np.asarray(y).flatten()

    if direction == 'backward':
        y = y[::-1].copy()

    return _ftle_compute(
        y,
        min_samples=min_samples,
        method=method,
        emb_dim=emb_dim,
        emb_tau=emb_tau,
        dim_method=dim_method,
        tau_method=tau_method,
    )
