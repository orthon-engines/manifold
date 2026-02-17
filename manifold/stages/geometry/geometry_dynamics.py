"""
Stage 07: Geometry Dynamics Entry Point
=======================================

Orchestration - reads parquet, calls core engine, writes output.

Inputs:
    - state_geometry.parquet

Output:
    - geometry_dynamics.parquet

Computes differential geometry of state evolution:
    - d1: velocity (first derivative)
    - d2: acceleration (second derivative)
    - d3: jerk (third derivative)
    - Curvature metrics
    - Collapse indicators
"""

import polars as pl
from typing import Optional

from manifold.core.geometry_dynamics import compute_geometry_dynamics
from manifold.io.writer import write_output


def run(
    state_geometry_path: str,
    data_path: str = ".",
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run geometry dynamics computation.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        data_path: Root data directory (for write_output)
        dt: Time step for derivatives (from config if None)
        smooth_window: Smoothing window (from config if None)
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 07: GEOMETRY DYNAMICS")
        print("Differential geometry of state evolution (d1/d2/d3)")
        print("=" * 70)

    state_geometry = pl.read_parquet(state_geometry_path)

    result = compute_geometry_dynamics(
        state_geometry,
        dt=dt,
        smooth_window=smooth_window,
        verbose=verbose,
    )

    write_output(result, data_path, 'geometry_dynamics', verbose=verbose)

    return result
