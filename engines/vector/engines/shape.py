"""
Shape engines — kurtosis, skewness, crest_factor.

Wraps engines.manifold.signal.statistics which provides shape-related
statistical measures. These are scale-agnostic features that describe
the distribution shape of a signal window.

Outputs:
    kurtosis      - Excess kurtosis (Fisher). >0 = heavy tails, <0 = light tails.
    skewness      - Asymmetry of distribution. 0 = symmetric.
    crest_factor  - Peak / RMS ratio. Higher = more impulsive.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, **params) -> Dict[str, float]:
    """Compute shape features from a 1D signal window.

    Delegates to engines.manifold.signal.statistics which imports from
    engines.primitives.individual.statistics (canonical).

    Args:
        y: 1D numpy array of signal values.
        **params: Unused. Accepted for uniform interface.

    Returns:
        Dict with kurtosis, skewness, crest_factor keys.
        Values are np.nan when insufficient samples.
    """
    from engines.manifold.signal.statistics import compute as _compute_statistics

    try:
        result = _compute_statistics(y)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {
        'kurtosis': np.nan,
        'skewness': np.nan,
        'crest_factor': np.nan,
    }
