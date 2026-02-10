"""
Decompose operation -- eigendecompose any feature matrix.

Takes ANY feature matrix (N entities x M features).
Eigendecomposes. Returns eigenvalues, eigenvectors, loadings.
Does not know if entities are signals or cohorts.

Orchestration-only: delegates to the existing stage entry points which
already handle I/O, grouping, sidecar files, and verbose output.
"""


def run(signal_vector_path, state_vector_path, output_path, **kwargs):
    """Run decomposition at signal scale. Delegates to existing stage_03.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path:  Path to state_vector.parquet
        output_path:        Output path for state_geometry.parquet
        **kwargs:           Forwarded (verbose, etc.)

    Returns:
        polars.DataFrame -- state_geometry result
    """
    from engines.entry_points.stage_03_state_geometry import run as _run

    return _run(signal_vector_path, state_vector_path, output_path, **kwargs)


def run_system(cohort_vector_path, output_path, **kwargs):
    """Run decomposition at cohort scale. Delegates to existing stage_26.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        output_path:        Output path for system_geometry.parquet
        **kwargs:           Forwarded (max_eigenvalues, verbose, etc.)

    Returns:
        polars.DataFrame -- system_geometry result
    """
    from engines.entry_points.stage_26_system_geometry import run as _run

    return _run(cohort_vector_path, output_path, **kwargs)
