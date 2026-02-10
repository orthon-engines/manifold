"""
Correlation engines -- pearson, spearman, cross-correlation, mutual information.

Delegates to:
    - engines.manifold.pairwise.correlation.compute          (Pearson/Spearman)
    - engines.manifold.pairwise.correlation.compute_cross_correlation  (xcorr)
    - engines.manifold.pairwise.correlation.compute_mutual_info        (MI)
    - engines.manifold.pairwise.correlation.compute_all       (all at once)
"""

import numpy as np
from typing import Dict, Any


def compute(x: np.ndarray, y: np.ndarray, **params) -> Dict[str, Any]:
    """
    Compute all correlation measures between two vectors.

    Args:
        x, y: Input vectors (1D arrays).
        **params:
            max_lag: int -- Maximum lag for cross-correlation (default 50).
            n_bins: int -- Number of bins for mutual information (default 10).

    Returns:
        Dict with:
            correlation: Pearson correlation
            correlation_abs: |Pearson|
            spearman: Spearman rank correlation
            max_xcorr: Maximum cross-correlation value
            lag_at_max: Lag at maximum cross-correlation
            xcorr_symmetric: Symmetry of cross-correlation function
            mutual_info: Mutual information (bits)
            normalized_mi: Normalized mutual information
    """
    from engines.manifold.pairwise.correlation import compute_all

    max_lag = params.get('max_lag', 50)
    return compute_all(x, y, max_lag=max_lag)


def compute_pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute Pearson correlation only."""
    from engines.manifold.pairwise.correlation import compute as _compute
    return _compute(x, y, method='pearson')


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute Spearman rank correlation only."""
    from engines.manifold.pairwise.correlation import compute as _compute
    return _compute(x, y, method='spearman')


def compute_cross_correlation(
    x: np.ndarray, y: np.ndarray, max_lag: int = 50
) -> Dict[str, Any]:
    """Compute cross-correlation and lag at maximum."""
    from engines.manifold.pairwise.correlation import compute_cross_correlation as _compute_xcorr
    return _compute_xcorr(x, y, max_lag=max_lag)


def compute_mutual_info(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Compute mutual information."""
    from engines.manifold.pairwise.correlation import compute_mutual_info as _compute_mi
    return _compute_mi(x, y, n_bins=n_bins)
