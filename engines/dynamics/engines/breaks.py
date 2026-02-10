"""
Break detection engine -- structural changes via Heaviside + Dirac.

Wraps engines.manifold.breaks.compute() to provide a scale-agnostic
interface. Detects steps (Heaviside), impulses (Dirac), and gradual
shifts using CUSUM, derivative magnitude, and local outlier detection.

ENGINES detects break locations and metrics. ORTHON classifies break types.
"""

import numpy as np
from typing import Dict, Any, List


def compute(
    y: np.ndarray,
    signal_id: str = 'unknown',
    sensitivity: float = 1.0,
    min_spacing: int = 10,
    context_window: int = 50,
) -> List[Dict[str, Any]]:
    """Detect structural breaks in a 1-D signal. Scale-agnostic.

    Args:
        y:              1-D array of values (ordered by I).
        signal_id:      Identifier passed through to output dicts.
        sensitivity:    Detection sensitivity (0.5=conservative, 2.0=aggressive).
        min_spacing:    Minimum samples between breaks.
        context_window: Samples for pre/post level computation.

    Returns:
        List of break dicts matching breaks.parquet schema.
        Empty list if no breaks detected.
    """
    from engines.manifold.breaks import compute as _breaks_compute

    return _breaks_compute(
        y,
        signal_id=signal_id,
        sensitivity=sensitivity,
        min_spacing=min_spacing,
        context_window=context_window,
    )


def summarize(breaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize a list of breaks into aggregate statistics.

    Args:
        breaks: List of break dicts (output of compute()).

    Returns:
        Dict with n_breaks, mean_magnitude, max_magnitude, etc.
    """
    from engines.manifold.breaks import summarize_breaks

    return summarize_breaks(breaks)
