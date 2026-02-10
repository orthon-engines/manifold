"""
Effective dimensionality -- eigenvalue entropy to intrinsic dimension.

Computes participation ratio (effective_dim) and Shannon entropy of
the eigenvalue spectrum. Scale-agnostic: works on any eigenvalue array.
"""

import numpy as np
from typing import Dict


def compute(eigenvalues: np.ndarray, **params) -> Dict[str, float]:
    """Compute effective dimension from eigenvalue spectrum. Scale-agnostic.

    Args:
        eigenvalues: 1-D array of eigenvalues (need not be sorted).
        **params: Reserved for future use.

    Returns:
        Dict with:
            effective_dim          -- participation ratio (total^2 / sum of squares)
            eigenvalue_entropy     -- Shannon entropy of normalised spectrum
            eigenvalue_entropy_norm -- entropy / log(n), in [0, 1]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[eigenvalues > 0]

    total = eigenvalues.sum()
    if total <= 0 or len(eigenvalues) == 0:
        return {
            'effective_dim': float('nan'),
            'eigenvalue_entropy': float('nan'),
            'eigenvalue_entropy_norm': float('nan'),
        }

    p = eigenvalues / total

    # Participation ratio
    effective_dim = float(total ** 2 / np.sum(eigenvalues ** 2))

    # Shannon entropy of normalised spectrum
    entropy = float(-np.sum(p * np.log(p)))

    # Normalised entropy (0 = all variance in one component, 1 = uniform)
    if len(eigenvalues) > 1:
        entropy_norm = float(entropy / np.log(len(eigenvalues)))
    else:
        entropy_norm = 0.0

    return {
        'effective_dim': effective_dim,
        'eigenvalue_entropy': entropy,
        'eigenvalue_entropy_norm': entropy_norm,
    }
