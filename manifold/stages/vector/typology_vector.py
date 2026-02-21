"""
Stage 00a: Typology Vector
===========================

Orchestrator for per-window typology metrics.
Reads observations, calls core engine, writes output.

Output:
  1. typology_windows.parquet — per-window metrics (signal/ directory)
  2. typology_vector.parquet  — per-signal summary  (signal/ directory)
"""

import polars as pl
from typing import Dict, Any

from joblib import Parallel, delayed

from manifold.core.typology import (
    compute_signal_typology,
    summarize_windows,
    METRIC_NAMES,
    CV_THRESHOLD,
)
from manifold.io.writer import write_output


_N_WORKERS = 2


def run(
    observations_path: str,
    data_path: str = ".",
    manifest: Dict[str, Any] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run Stage 00a: Typology Vector.

    Writes two files:
      - typology_windows.parquet (per-window metrics)
      - typology_vector.parquet  (per-signal summary)

    Args:
        observations_path: Path to observations.parquet.
        data_path: Root data directory (for write_output).
        manifest: Manifest dict (must have system.window and system.stride).
        verbose: Print progress.

    Returns:
        typology_windows DataFrame (per-window metrics).
    """
    if verbose:
        print("=" * 70)
        print("STAGE 00a: TYPOLOGY VECTOR")
        print("Per-window typology metrics (11 measures)")
        print("=" * 70)

    if manifest is None:
        raise ValueError("Manifest required for typology_vector (need system.window/stride)")

    system = manifest.get('system', {})
    system_window = system.get('window')
    system_stride = system.get('stride')
    if system_window is None or system_stride is None:
        raise ValueError("Manifest system section must have 'window' and 'stride'")

    obs = pl.read_parquet(observations_path)
    has_cohort = 'cohort' in obs.columns

    signal_ids = obs['signal_id'].unique().sort().to_list()
    if verbose:
        print(f"Signals: {len(signal_ids)}, window={system_window}, stride={system_stride}")

    # Build tasks: (signal_id, signal_data, signal_0_data, cohort)
    tasks = []
    for sig_id in signal_ids:
        sig_df = obs.filter(pl.col('signal_id') == sig_id)

        if has_cohort:
            cohorts = sig_df['cohort'].unique().to_list()
            for cohort in cohorts:
                cohort_df = sig_df.filter(pl.col('cohort') == cohort).sort('signal_0')
                signal_data = cohort_df['value'].to_numpy()
                signal_0_data = cohort_df['signal_0'].to_numpy()
                tasks.append((sig_id, signal_data, signal_0_data, cohort))
        else:
            sorted_df = sig_df.sort('signal_0')
            signal_data = sorted_df['value'].to_numpy()
            signal_0_data = sorted_df['signal_0'].to_numpy()
            tasks.append((sig_id, signal_data, signal_0_data, ''))

    if verbose:
        print(f"Tasks: {len(tasks)} (signal x cohort combinations)")

    if not tasks:
        _write_empty(data_path, verbose)
        return pl.DataFrame()

    # Parallel computation via core engine
    if len(tasks) == 1:
        sig_id, signal_data, signal_0_data, cohort = tasks[0]
        all_rows = compute_signal_typology(
            signal_data, signal_0_data, system_window, system_stride, sig_id, cohort
        )
    else:
        results = Parallel(n_jobs=_N_WORKERS, prefer="processes")(
            delayed(compute_signal_typology)(
                signal_data, signal_0_data, system_window, system_stride, sig_id, cohort
            )
            for sig_id, signal_data, signal_0_data, cohort in tasks
        )
        all_rows = []
        for signal_rows in results:
            if signal_rows:
                all_rows.extend(signal_rows)

    if verbose:
        print(f"  {len(all_rows):,} window rows computed")

    if not all_rows:
        _write_empty(data_path, verbose)
        return pl.DataFrame()

    # Build windows DataFrame
    windows_df = pl.DataFrame(all_rows, infer_schema_length=None)
    windows_df = windows_df.cast({'window_id': pl.UInt32, 'n_obs': pl.UInt32})

    # Summarize into per-signal vector
    vector_df = summarize_windows(windows_df)

    # Write both outputs
    write_output(windows_df, data_path, 'typology_windows', verbose=verbose)
    write_output(vector_df, data_path, 'typology_vector', verbose=verbose)

    if verbose:
        n_varies = 0
        varies_cols = [c for c in vector_df.columns if c.endswith('_varies')]
        if varies_cols:
            for c in varies_cols:
                n_varies += vector_df[c].sum()
        print(f"\n  Windows: {windows_df.height} rows x {windows_df.width} columns")
        print(f"  Vector:  {vector_df.height} signals, {n_varies} metric-signal pairs vary (CV > {CV_THRESHOLD})")

    return windows_df


def _write_empty(data_path: str, verbose: bool) -> None:
    """Write empty schema files when no data."""
    window_schema = {
        'window_id': pl.UInt32, 'signal_id': pl.Utf8, 'cohort': pl.Utf8,
        'signal_0_start': pl.Float64, 'signal_0_end': pl.Float64,
        'signal_0_center': pl.Float64, 'n_obs': pl.UInt32,
    }
    for m in METRIC_NAMES:
        window_schema[m] = pl.Float64

    vector_schema = {'signal_id': pl.Utf8, 'cohort': pl.Utf8, 'n_windows': pl.UInt32}
    for m in METRIC_NAMES:
        vector_schema[f'{m}_mean'] = pl.Float64
        vector_schema[f'{m}_std'] = pl.Float64
        vector_schema[f'{m}_cv'] = pl.Float64
        vector_schema[f'{m}_varies'] = pl.Boolean

    write_output(pl.DataFrame(schema=window_schema), data_path, 'typology_windows', verbose=verbose)
    write_output(pl.DataFrame(schema=vector_schema), data_path, 'typology_vector', verbose=verbose)
