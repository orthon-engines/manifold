"""
Dynamics operation -- dynamical systems metrics on trajectories.

Takes ANY trajectory (time series of scalars or vectors).
Computes FTLE, velocity fields, break detection, ridge proximity.
Does not know if trajectory is signal or cohort.

Orchestration-only: delegates to the existing stage entry points which
already handle I/O, grouping, sidecar files, and verbose output.
"""


def run_ftle(observations_path, output_path="ftle.parquet", **kwargs):
    """Run FTLE computation. Delegates to existing stage_08.

    Args:
        observations_path: Path to observations.parquet
        output_path:       Output path for ftle.parquet
        **kwargs:          Forwarded (min_samples, method, direction, verbose, etc.)

    Returns:
        polars.DataFrame -- FTLE result
    """
    from engines.entry_points.stage_08_ftle import run as _run

    return _run(observations_path, output_path, **kwargs)


def run_velocity(observations_path, output_path="velocity_field.parquet", **kwargs):
    """Run velocity field computation. Delegates to existing stage_21.

    Args:
        observations_path: Path to observations.parquet
        output_path:       Output path for velocity_field.parquet
        **kwargs:          Forwarded (smooth, smooth_window, verbose, etc.)

    Returns:
        polars.DataFrame -- velocity_field result
    """
    from engines.entry_points.stage_21_velocity_field import run as _run

    return _run(observations_path, output_path, **kwargs)


def run_breaks(observations_path, output_path="breaks.parquet", **kwargs):
    """Run structural break detection. Delegates to existing stage_00.

    Args:
        observations_path: Path to observations.parquet
        output_path:       Output path for breaks.parquet
        **kwargs:          Forwarded (sensitivity, min_spacing, verbose, etc.)

    Returns:
        polars.DataFrame -- breaks result
    """
    from engines.entry_points.stage_00_breaks import run as _run

    return _run(observations_path, output_path, **kwargs)


def run_ftle_rolling(observations_path, output_path="ftle_rolling.parquet", **kwargs):
    """Run rolling FTLE computation. Delegates to existing stage_22.

    Args:
        observations_path: Path to observations.parquet
        output_path:       Output path for ftle_rolling.parquet
        **kwargs:          Forwarded (window_size, stride, direction, verbose, etc.)

    Returns:
        polars.DataFrame -- ftle_rolling result
    """
    from engines.entry_points.stage_22_ftle_rolling import run as _run

    return _run(observations_path, output_path, **kwargs)


def run_ridge(
    ftle_rolling_path,
    velocity_field_path,
    output_path="ridge_proximity.parquet",
    **kwargs,
):
    """Run ridge proximity/urgency computation. Delegates to existing stage_23.

    Args:
        ftle_rolling_path:    Path to ftle_rolling.parquet
        velocity_field_path:  Path to velocity_field.parquet
        output_path:          Output path for ridge_proximity.parquet
        **kwargs:             Forwarded (ridge_threshold, urgency_threshold, verbose, etc.)

    Returns:
        polars.DataFrame -- ridge_proximity result
    """
    from engines.entry_points.stage_23_ridge_proximity import run as _run

    return _run(ftle_rolling_path, velocity_field_path, output_path, **kwargs)
