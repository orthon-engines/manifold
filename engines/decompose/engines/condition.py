"""
Condition number and spectral gap from eigenvalue spectrum.

Computes diagnostic metrics that characterise the shape of the
eigenvalue distribution. Scale-agnostic: works on any eigenvalue array.
"""

import numpy as np
from typing import Dict


def compute(eigenvalues: np.ndarray, **params) -> Dict[str, float]:
    """Compute condition metrics from eigenvalues. Scale-agnostic.

    Args:
        eigenvalues: 1-D array of eigenvalues (need not be sorted).
        **params: Reserved for future use.

    Returns:
        Dict with (subset present depends on spectrum length):
            condition_number -- lambda_max / lambda_min (inf if min <= 0)
            ratio_2_1        -- lambda_2 / lambda_1
            spectral_gap     -- lambda_1 - lambda_2
            ratio_3_1        -- lambda_3 / lambda_1  (only if >= 3 eigenvalues)
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]

    results: Dict[str, float] = {}

    # Condition number
    if len(eigenvalues) >= 1 and eigenvalues[-1] > 0:
        results['condition_number'] = float(eigenvalues[0] / eigenvalues[-1])
    else:
        results['condition_number'] = float('inf')

    # Ratios and spectral gap (need at least 2 eigenvalues)
    if len(eigenvalues) >= 2:
        if eigenvalues[0] > 0:
            results['ratio_2_1'] = float(eigenvalues[1] / eigenvalues[0])
        else:
            results['ratio_2_1'] = float('nan')
        results['spectral_gap'] = float(eigenvalues[0] - eigenvalues[1])

    # Third eigenvalue ratio
    if len(eigenvalues) >= 3:
        if eigenvalues[0] > 0:
            results['ratio_3_1'] = float(eigenvalues[2] / eigenvalues[0])
        else:
            results['ratio_3_1'] = float('nan')

    return results
