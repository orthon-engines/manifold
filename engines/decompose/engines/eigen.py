"""
Eigendecomposition engine -- eigenvalues, eigenvectors, loadings.

Wraps engines.manifold.state.eigendecomp.compute() to provide a
scale-agnostic interface. The underlying function does not care whether
rows are signals or cohorts -- it decomposes any (N x D) feature matrix.
"""

import numpy as np
from typing import Dict, Any


def compute(matrix: np.ndarray, **params) -> Dict[str, Any]:
    """Eigendecompose ANY feature matrix. Scale-agnostic.

    Args:
        matrix: (N_entities, D_features) -- rows are entities, columns are features.
        **params: Forwarded to eigendecomp.compute():
            centroid    -- pre-computed centroid (D,). If None, computed from data.
            norm_method -- "zscore" | "robust" | "mad" | "none" (default "zscore")
            min_signals -- minimum valid rows required (default 2)

    Returns:
        Dict with:
            eigenvalues              -- (k,) array, descending
            explained_ratio          -- (k,) array, fraction of total variance
            total_variance           -- float
            effective_dim            -- float, participation ratio
            eigenvalue_entropy       -- float
            eigenvalue_entropy_normalized -- float
            condition_number         -- float
            ratio_2_1               -- float, lambda_2 / lambda_1
            ratio_3_1               -- float, lambda_3 / lambda_1
            principal_components     -- (k, D) array (Vt from SVD)
            signal_loadings          -- (N, k) array (U from SVD)
            n_signals               -- int, rows used
            n_features              -- int, columns
    """
    from engines.manifold.state.eigendecomp import compute as _eigen_compute

    return _eigen_compute(matrix, **params)
