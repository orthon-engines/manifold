"""
Ridge proximity engine -- urgency = velocity toward FTLE ridge.

Computes the rate of approach to an FTLE ridge from pre-computed FTLE
and velocity data. Scale-agnostic: operates on arrays, not parquet.

A system can be:
- Near a ridge but moving parallel  -> High FTLE, low urgency. Stable for now.
- Far from ridge but heading toward -> Low FTLE, HIGH urgency. Trouble coming.

Full pipeline-scale ridge proximity (reading parquet, per-cohort iteration,
urgency classification) is handled by stage_23_ridge_proximity. This engine
provides the core per-signal math.
"""

import numpy as np
from typing import Dict


def compute(
    ftle_values: np.ndarray,
    speeds: np.ndarray,
    ridge_threshold: float = 0.05,
    urgency_threshold: float = 0.001,
) -> Dict[str, np.ndarray]:
    """Compute ridge proximity from FTLE and velocity arrays. Scale-agnostic.

    Args:
        ftle_values:       1-D array of rolling FTLE values (aligned by I).
        speeds:            1-D array of state-space speeds (same length).
        ridge_threshold:   FTLE value considered "near ridge".
        urgency_threshold: Minimum urgency to be "approaching".

    Returns:
        Dict with:
            ftle_gradient   -- (N,) temporal gradient of FTLE
            urgency         -- (N,) speed * sign(grad) * |grad|
            time_to_ridge   -- (N,) estimated steps to reach ridge (inf if retreating)
    """
    ftle_values = np.asarray(ftle_values, dtype=float)
    speeds = np.asarray(speeds, dtype=float)

    n = len(ftle_values)
    if n < 3:
        empty = np.full(n, np.nan)
        return {
            'ftle_gradient': empty.copy(),
            'urgency': empty.copy(),
            'time_to_ridge': empty.copy(),
        }

    # Temporal gradient of FTLE field
    ftle_gradient = np.gradient(ftle_values)

    # Urgency: speed component along FTLE gradient direction
    # Positive = heading toward higher FTLE = approaching ridge
    urgency = speeds * np.sign(ftle_gradient) * np.abs(ftle_gradient)

    # Time-to-ridge estimate
    time_to_ridge = np.full(n, np.inf)
    approaching = urgency > urgency_threshold
    grad_nonzero = np.abs(ftle_gradient) > 1e-6
    valid = approaching & grad_nonzero
    ftle_remaining = np.maximum(0, ridge_threshold - ftle_values)
    time_to_ridge[valid] = ftle_remaining[valid] / (np.abs(urgency[valid]) + 1e-12)

    return {
        'ftle_gradient': ftle_gradient,
        'urgency': urgency,
        'time_to_ridge': time_to_ridge,
    }
