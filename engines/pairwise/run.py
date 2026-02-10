"""
Pairwise operation -- compute relationship metrics between entity pairs.

Takes ANY two vectors. Computes all relationship metrics.
Returns one row per pair per window (or per pair summary).

This module does NOT know what scale it is operating at.
The caller decides whether x and y are raw signals, cohort
centroids, or system-level vectors -- the math is identical.

Delegates to:
    - engines.pairwise.engines.distance      (euclidean, DTW, cosine, manhattan)
    - engines.pairwise.engines.correlation   (pearson, spearman, xcorr, MI)
    - engines.pairwise.engines.information   (granger, TE, KL, JS)
    - engines.pairwise.engines.cointegration (Engle-Granger, ADF)
    - engines.pairwise.engines.copula        (Gaussian, Clayton, Gumbel, Frank)

For the full signal-level pipeline (parquet I/O, cohort grouping,
eigenvector gating), use stage_06 and stage_10 entry points directly.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    engines: Optional[list] = None,
    **params,
) -> Dict[str, Any]:
    """
    Compute all pairwise relationship metrics between two vectors.

    This is the primary entry point for pairwise computation at any scale.
    It runs each requested engine and merges results into a single dict.

    Args:
        x, y: Input vectors (1D numpy arrays).
        engines: List of engine names to run. If None, runs all:
                 ['distance', 'correlation', 'information', 'cointegration', 'copula']
        **params: Forwarded to individual engines. Common params:
            max_lag: int -- Granger / xcorr max lag
            te_lag: int -- Transfer entropy lag
            n_bins: int -- Discretization bins for MI/TE
            dtw_window: int -- DTW Sakoe-Chiba band
            significance: float -- Cointegration significance level

    Returns:
        Dict with all computed metrics, keyed by engine prefix where needed.
        Keys are flat (no nesting) for direct DataFrame row construction.
    """
    from engines.pairwise.engines import distance as _distance
    from engines.pairwise.engines import correlation as _correlation
    from engines.pairwise.engines import information as _information
    from engines.pairwise.engines import cointegration as _cointegration
    from engines.pairwise.engines import copula as _copula

    all_engines = ['distance', 'correlation', 'information', 'cointegration', 'copula']
    if engines is None:
        engines = all_engines

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    results = {}

    if 'distance' in engines:
        try:
            results.update(_distance.compute(x, y, **params))
        except Exception:
            pass

    if 'correlation' in engines:
        try:
            results.update(_correlation.compute(x, y, **params))
        except Exception:
            pass

    if 'information' in engines:
        try:
            results.update(_information.compute(x, y, **params))
        except Exception:
            pass

    if 'cointegration' in engines:
        try:
            results.update(_cointegration.compute(x, y, **params))
        except Exception:
            pass

    if 'copula' in engines:
        try:
            results.update(_copula.compute(x, y, **params))
        except Exception:
            pass

    return results


def run(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "signal_pairwise.parquet",
    **kwargs,
):
    """
    Run the full signal-level pairwise pipeline.

    This delegates to stage_06_signal_pairwise.run which handles:
        - Parquet I/O
        - Cohort grouping
        - Feature group selection
        - Eigenvector gating
        - Output writing

    For direct vector-to-vector computation without pipeline
    overhead, use compute_pairwise(x, y) instead.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path for signal_pairwise.parquet
        **kwargs: Forwarded to stage_06 run() -- includes:
            state_geometry_path: str
            coloading_threshold: float
            verbose: bool

    Returns:
        polars DataFrame (signal_pairwise)
    """
    from engines.entry_points.stage_06_signal_pairwise import run as _run
    return _run(signal_vector_path, state_vector_path, output_path, **kwargs)
