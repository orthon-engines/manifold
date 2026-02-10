"""
Distance engines -- euclidean, DTW, cosine similarity.

Delegates to engines.primitives.pairwise.distance which provides:
    - dynamic_time_warping(sig_a, sig_b, window=None)
    - euclidean_distance(sig_a, sig_b, normalized=False)
    - cosine_similarity(sig_a, sig_b)
    - manhattan_distance(sig_a, sig_b, normalized=False)
"""

import numpy as np
from typing import Dict


def compute(x: np.ndarray, y: np.ndarray, **params) -> Dict[str, float]:
    """
    Compute distance metrics between two vectors. Scale-agnostic.

    Args:
        x, y: Input vectors (1D arrays, same length preferred).
        **params:
            dtw_window: int or None -- Sakoe-Chiba band width for DTW.
            normalized: bool -- If True, normalize euclidean/manhattan by sqrt(n)/n.

    Returns:
        Dict with:
            distance_euclidean: Euclidean distance
            distance_euclidean_normalized: Euclidean / sqrt(n)
            distance_manhattan: Manhattan (L1) distance
            cosine_similarity: Cosine similarity in [-1, 1]
            dtw_distance: Dynamic Time Warping distance
    """
    from engines.primitives.pairwise.distance import (
        dynamic_time_warping,
        euclidean_distance,
        cosine_similarity as _cosine_similarity,
        manhattan_distance,
    )

    results = {}

    # Euclidean
    results['distance_euclidean'] = float(euclidean_distance(x, y, normalized=False))
    results['distance_euclidean_normalized'] = float(euclidean_distance(x, y, normalized=True))

    # Manhattan
    results['distance_manhattan'] = float(manhattan_distance(x, y, normalized=False))

    # Cosine similarity
    results['cosine_similarity'] = float(_cosine_similarity(x, y))

    # DTW
    try:
        dtw_window = params.get('dtw_window', None)
        results['dtw_distance'] = float(dynamic_time_warping(x, y, window=dtw_window))
    except Exception:
        results['dtw_distance'] = float('nan')

    return results
