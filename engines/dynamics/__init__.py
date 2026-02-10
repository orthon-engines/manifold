"""
Dynamics operation module -- dynamical systems metrics on ANY trajectory.

This module is scale-agnostic. It takes a trajectory (time series of scalars
or vectors) and computes FTLE, velocity fields, break detection, and ridge
proximity. It does NOT know whether the trajectory comes from a single signal,
a cohort centroid, or any other source.

Entry points:
    run_ftle()          -- forward/backward FTLE (delegates to stage_08)
    run_ftle_rolling()  -- rolling FTLE over time (delegates to stage_22)
    run_velocity()      -- state-space velocity field (delegates to stage_21)
    run_ridge()         -- FTLE ridge proximity/urgency (delegates to stage_23)
    run_breaks()        -- structural break detection (delegates to stage_00)

Compute engines:
    engines.ftle           -- FTLE via Rosenstein/Kantz
    engines.ftle_rolling   -- rolling FTLE with trend statistics
    engines.velocity       -- speed, acceleration, curvature
    engines.ridge          -- urgency = velocity toward FTLE ridge
    engines.breaks         -- Heaviside + Dirac break detection
"""

from engines.dynamics.run import (
    run_ftle,
    run_ftle_rolling,
    run_velocity,
    run_ridge,
    run_breaks,
)
from engines.dynamics.engines.ftle import compute as compute_ftle
from engines.dynamics.engines.velocity import compute as compute_velocity

__all__ = [
    'run_ftle',
    'run_ftle_rolling',
    'run_velocity',
    'run_ridge',
    'run_breaks',
    'compute_ftle',
    'compute_velocity',
]
