"""
Rolling FTLE engine -- FTLE evolution over sliding time windows.

Wraps engines.manifold.dynamics.ftle.compute_rolling() and compute_trend()
to provide a scale-agnostic interface. Shows how stability evolves through
a trajectory's lifecycle.

A trajectory that is stable for 9000 samples and chaotic for 1000 samples
gets an average FTLE that describes neither state. Rolling FTLE gives
FTLE(I) -- the stability at each moment.
"""

import numpy as np
from typing import Dict, Any


def compute(
    y: np.ndarray,
    window: int = 500,
    stride: int = 50,
    min_samples: int = 200,
) -> Dict[str, np.ndarray]:
    """Compute rolling FTLE for temporal stability evolution. Scale-agnostic.

    Args:
        y:            1-D array of values (ordered by I).
        window:       Window size per FTLE computation (recommend 500+).
        stride:       Step size between windows.
        min_samples:  Minimum samples per window for reliable FTLE.

    Returns:
        Dict with:
            rolling_ftle            -- (N,) array, FTLE at each window endpoint
            rolling_ftle_std        -- (N,) array, FTLE std at each window
            rolling_ftle_confidence -- (N,) array, confidence per window
    """
    from engines.manifold.dynamics.ftle import compute_rolling as _rolling

    return _rolling(y, window=window, stride=stride, min_samples=min_samples)


def compute_trend(ftle_values: np.ndarray) -> Dict[str, float]:
    """Compute trend statistics on a rolling FTLE series. Scale-agnostic.

    Returns numbers only -- ORTHON interprets what "destabilizing" means.

    Args:
        ftle_values: 1-D array of FTLE values (may contain NaN).

    Returns:
        Dict with: ftle_slope, ftle_r2.
    """
    from engines.manifold.dynamics.ftle import compute_trend as _trend

    return _trend(ftle_values)
