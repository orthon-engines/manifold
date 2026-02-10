"""
Velocity engine -- direction, speed, curvature in state space.

Computes the velocity field from a trajectory matrix. Scale-agnostic:
works on any (T, D) matrix of T timesteps in D-dimensional space.
"""

import numpy as np
from typing import Dict


def compute(trajectory: np.ndarray, **params) -> Dict[str, np.ndarray]:
    """Compute velocity field from a trajectory matrix. Scale-agnostic.

    Mathematical foundation (matching stage_21 pattern):
        Velocity:     v(t) = x(t+1) - x(t)
        Speed:        s(t) = |v(t)|
        Acceleration: a(t) = v(t+1) - v(t)
        Curvature:    kappa(t) = |a_perp| / |v|^2

    Args:
        trajectory: (T, D) matrix -- T timesteps, D dimensions.
                    If 1-D, treated as (T, 1).
        **params:   Reserved for future extensions.

    Returns:
        Dict with:
            speed                    -- (T,) array, NaN-padded at boundaries
            acceleration_magnitude   -- (T,) array, NaN-padded at boundaries
            curvature                -- (T,) array, NaN-padded at boundaries
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]

    T, D = trajectory.shape

    if T < 3:
        return {
            'speed': np.full(T, np.nan),
            'acceleration_magnitude': np.full(T, np.nan),
            'curvature': np.full(T, np.nan),
        }

    # Velocity: first difference  (T-1, D)
    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    # Acceleration: second difference  (T-2, D)
    accel = np.diff(velocity, axis=0)
    accel_mag = np.linalg.norm(accel, axis=1)

    # Curvature = |a_perp| / |v|^2
    curvature = np.full(T, np.nan)
    for t in range(len(accel)):
        v = velocity[t]
        a = accel[t]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            v_hat = v / v_norm
            a_parallel = np.dot(a, v_hat) * v_hat
            a_perp = a - a_parallel
            curvature[t + 1] = float(np.linalg.norm(a_perp) / (v_norm ** 2))

    # Pad speed and accel to match T (NaN at boundaries)
    speed_full = np.concatenate([[np.nan], speed])
    accel_full = np.concatenate([[np.nan, np.nan], accel_mag])

    return {
        'speed': speed_full,
        'acceleration_magnitude': accel_full,
        'curvature': curvature,
    }
