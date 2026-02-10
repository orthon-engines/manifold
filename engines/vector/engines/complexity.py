"""
Complexity engines — sample_entropy, permutation_entropy, hurst, acf_decay.

Wraps engines.manifold.signal.complexity (entropy measures) and
engines.manifold.signal.memory (long-range dependence measures).

Outputs from complexity:
    sample_entropy        - Regularity measure. Higher = more complex.
    permutation_entropy   - Ordinal pattern complexity. Higher = more random.
    approximate_entropy   - Similar to sample_entropy, less bias.

Outputs from memory:
    hurst       - Long-range dependence. <0.5 = anti-persistent, >0.5 = persistent.
    hurst_r2    - R-squared of the Hurst fit.
    acf_lag1    - Autocorrelation at lag 1.
    acf_lag10   - Autocorrelation at lag 10.
    acf_half_life - Lag where ACF drops below 0.5.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, **params) -> Dict[str, Any]:
    """Compute complexity and memory features from a 1D signal window.

    Delegates to engines.manifold.signal.complexity and
    engines.manifold.signal.memory (canonical).

    Args:
        y: 1D numpy array of signal values.
        **params: Unused. Accepted for uniform interface.

    Returns:
        Dict with entropy and long-range dependence features.
        Values are np.nan when insufficient samples.
    """
    from engines.manifold.signal.complexity import compute as _compute_complexity
    from engines.manifold.signal.memory import compute as _compute_memory
    from engines.manifold.signal.memory import compute_acf_decay as _compute_acf_decay

    results = {}

    # Entropy measures (sample_entropy, permutation_entropy, approximate_entropy)
    try:
        r = _compute_complexity(y)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        results.update({
            'sample_entropy': np.nan,
            'permutation_entropy': np.nan,
            'approximate_entropy': np.nan,
        })

    # Long-range dependence (hurst, hurst_r2)
    try:
        r = _compute_memory(y)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        results.update({
            'hurst': np.nan,
            'hurst_r2': np.nan,
        })

    # Autocorrelation decay (acf_lag1, acf_lag10, acf_half_life)
    try:
        r = _compute_acf_decay(y)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        results.update({
            'acf_lag1': np.nan,
            'acf_lag10': np.nan,
            'acf_half_life': np.nan,
        })

    return results
