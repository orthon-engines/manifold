"""
Stage 00a: Typology Vector
===========================

Per-window typology metrics computed at system window/stride alignment.
Runs BEFORE signal_vector (Stage 01) so engine gating can use local character.

11 metrics per window per signal:
  hurst, perm_entropy, sample_entropy, lyapunov_proxy,
  spectral_flatness, kurtosis, cv, mean_abs_diff,
  range_norm, zero_crossing_rate, trend_strength

Uses pmtvs via manifold.core._pmtvs (same compat layer as signal_vector).
Writes typology_vector.parquet to signal/ subdirectory.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional

from joblib import Parallel, delayed

from manifold.core._pmtvs import (
    hurst_exponent,
    permutation_entropy,
    sample_entropy,
    lyapunov_rosenstein,
)
from manifold.io.writer import write_output


# Match signal_vector worker count
_N_WORKERS = 2

# Minimum observations per window for meaningful metrics
_MIN_OBS = 20

# Std threshold below which a window is considered constant
_CONSTANT_STD = 1e-10


def _compute_window_metrics(window_data: np.ndarray) -> Dict[str, float]:
    """Compute 11 typology metrics for a single window.

    Args:
        window_data: 1D numpy array of signal values in the window.

    Returns:
        Dict of metric_name -> float (NaN on failure).
    """
    n = len(window_data)
    std = np.std(window_data)

    # Guard: too few obs or near-constant
    if n < _MIN_OBS or std < _CONSTANT_STD:
        return {
            'hurst': float('nan'),
            'perm_entropy': float('nan'),
            'sample_entropy': float('nan'),
            'lyapunov_proxy': float('nan'),
            'spectral_flatness': float('nan'),
            'kurtosis': float('nan'),
            'cv': float('nan'),
            'mean_abs_diff': float('nan'),
            'range_norm': float('nan'),
            'zero_crossing_rate': float('nan'),
            'trend_strength': float('nan'),
        }

    metrics = {}

    # --- pmtvs metrics (try/except → NaN on failure) ---

    try:
        metrics['hurst'] = float(hurst_exponent(window_data))
    except Exception:
        metrics['hurst'] = float('nan')

    try:
        metrics['perm_entropy'] = float(permutation_entropy(window_data))
    except Exception:
        metrics['perm_entropy'] = float('nan')

    try:
        metrics['sample_entropy'] = float(sample_entropy(window_data))
    except Exception:
        metrics['sample_entropy'] = float('nan')

    try:
        result = lyapunov_rosenstein(window_data)
        # lyapunov_rosenstein returns (exponent, divergence_curve, lags) tuple
        if isinstance(result, tuple):
            metrics['lyapunov_proxy'] = float(result[0])
        else:
            metrics['lyapunov_proxy'] = float(result)
    except Exception:
        metrics['lyapunov_proxy'] = float('nan')

    # --- numpy-based metrics ---

    # spectral_flatness: geometric mean / arithmetic mean of PSD
    try:
        from numpy.fft import rfft
        fft_vals = np.abs(rfft(window_data - np.mean(window_data)))[1:]  # skip DC
        psd = fft_vals ** 2
        if len(psd) > 0 and np.all(psd >= 0):
            psd_pos = psd[psd > 0]
            if len(psd_pos) > 0:
                geo_mean = np.exp(np.mean(np.log(psd_pos)))
                arith_mean = np.mean(psd)
                metrics['spectral_flatness'] = float(geo_mean / arith_mean) if arith_mean > 0 else float('nan')
            else:
                metrics['spectral_flatness'] = float('nan')
        else:
            metrics['spectral_flatness'] = float('nan')
    except Exception:
        metrics['spectral_flatness'] = float('nan')

    # kurtosis (excess)
    try:
        mean = np.mean(window_data)
        m4 = np.mean((window_data - mean) ** 4)
        m2 = np.mean((window_data - mean) ** 2)
        metrics['kurtosis'] = float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else float('nan')
    except Exception:
        metrics['kurtosis'] = float('nan')

    # cv (coefficient of variation)
    try:
        mean = np.mean(window_data)
        metrics['cv'] = float(std / abs(mean)) if abs(mean) > _CONSTANT_STD else float('nan')
    except Exception:
        metrics['cv'] = float('nan')

    # mean_abs_diff (average |x[i+1] - x[i]|)
    try:
        diffs = np.abs(np.diff(window_data))
        metrics['mean_abs_diff'] = float(np.mean(diffs))
    except Exception:
        metrics['mean_abs_diff'] = float('nan')

    # range_norm (range / std)
    try:
        r = float(np.max(window_data) - np.min(window_data))
        metrics['range_norm'] = r / std if std > _CONSTANT_STD else float('nan')
    except Exception:
        metrics['range_norm'] = float('nan')

    # zero_crossing_rate
    try:
        centered = window_data - np.mean(window_data)
        crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
        metrics['zero_crossing_rate'] = float(crossings / (n - 1))
    except Exception:
        metrics['zero_crossing_rate'] = float('nan')

    # trend_strength: 1 - (var(residuals) / var(signal))
    try:
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, window_data, 1)
        fitted = np.polyval(coeffs, x)
        residuals = window_data - fitted
        var_signal = np.var(window_data)
        var_resid = np.var(residuals)
        metrics['trend_strength'] = max(0.0, float(1.0 - var_resid / var_signal)) if var_signal > 0 else 0.0
    except Exception:
        metrics['trend_strength'] = float('nan')

    return metrics


def _compute_single_signal(
    signal_id: str,
    signal_data: np.ndarray,
    signal_0_data: np.ndarray,
    system_window: int,
    system_stride: int,
    cohort: str = '',
) -> List[Dict[str, Any]]:
    """Compute typology metrics for all windows of one signal.

    Args:
        signal_id: Signal identifier.
        signal_data: 1D numpy array of signal values (sorted by signal_0).
        signal_0_data: 1D numpy array of signal_0 values.
        system_window: System window size in samples.
        system_stride: System stride in samples.
        cohort: Cohort identifier.

    Returns:
        List of row dicts, one per window.
    """
    rows = []
    window_id = 0

    for window_end in range(system_window - 1, len(signal_data), system_stride):
        window_start = max(0, window_end - system_window + 1)
        window_data = signal_data[window_start:window_end + 1]

        metrics = _compute_window_metrics(window_data)

        row = {
            'window_id': window_id,
            'signal_id': signal_id,
            'cohort': cohort,
            'signal_0_start': float(signal_0_data[window_start]),
            'signal_0_end': float(signal_0_data[window_end]),
            'signal_0_center': (float(signal_0_data[window_start]) + float(signal_0_data[window_end])) / 2,
            'n_obs': len(window_data),
        }
        row.update(metrics)
        rows.append(row)
        window_id += 1

    return rows


def run(
    observations_path: str,
    data_path: str = ".",
    manifest: Dict[str, Any] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run per-window typology vector computation.

    Args:
        observations_path: Path to observations.parquet.
        data_path: Root data directory (for write_output).
        manifest: Manifest dict (must have system.window and system.stride).
        verbose: Print progress.

    Returns:
        DataFrame with per-window typology metrics.
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
                if len(signal_data) >= system_window:
                    tasks.append((sig_id, signal_data, signal_0_data, cohort))
        else:
            sorted_df = sig_df.sort('signal_0')
            signal_data = sorted_df['value'].to_numpy()
            signal_0_data = sorted_df['signal_0'].to_numpy()
            if len(signal_data) >= system_window:
                tasks.append((sig_id, signal_data, signal_0_data, ''))

    if verbose:
        print(f"Tasks: {len(tasks)} (signal × cohort combinations with >= {system_window} samples)")

    if not tasks:
        empty_df = pl.DataFrame(schema={
            'window_id': pl.UInt32, 'signal_id': pl.Utf8, 'cohort': pl.Utf8,
            'signal_0_start': pl.Float64, 'signal_0_end': pl.Float64,
            'signal_0_center': pl.Float64, 'n_obs': pl.UInt32,
            'hurst': pl.Float64, 'perm_entropy': pl.Float64,
            'sample_entropy': pl.Float64, 'lyapunov_proxy': pl.Float64,
            'spectral_flatness': pl.Float64, 'kurtosis': pl.Float64,
            'cv': pl.Float64, 'mean_abs_diff': pl.Float64,
            'range_norm': pl.Float64, 'zero_crossing_rate': pl.Float64,
            'trend_strength': pl.Float64,
        })
        write_output(empty_df, data_path, 'typology_vector', verbose=verbose)
        return empty_df

    if len(tasks) == 1:
        sig_id, signal_data, signal_0_data, cohort = tasks[0]
        all_rows = _compute_single_signal(
            sig_id, signal_data, signal_0_data, system_window, system_stride, cohort
        )
    else:
        results = Parallel(n_jobs=_N_WORKERS, prefer="processes")(
            delayed(_compute_single_signal)(
                sig_id, signal_data, signal_0_data, system_window, system_stride, cohort
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
        empty_df = pl.DataFrame(schema={
            'window_id': pl.UInt32, 'signal_id': pl.Utf8, 'cohort': pl.Utf8,
            'signal_0_start': pl.Float64, 'signal_0_end': pl.Float64,
            'signal_0_center': pl.Float64, 'n_obs': pl.UInt32,
            'hurst': pl.Float64, 'perm_entropy': pl.Float64,
            'sample_entropy': pl.Float64, 'lyapunov_proxy': pl.Float64,
            'spectral_flatness': pl.Float64, 'kurtosis': pl.Float64,
            'cv': pl.Float64, 'mean_abs_diff': pl.Float64,
            'range_norm': pl.Float64, 'zero_crossing_rate': pl.Float64,
            'trend_strength': pl.Float64,
        })
        write_output(empty_df, data_path, 'typology_vector', verbose=verbose)
        return empty_df

    # Cast to correct types
    df = pl.DataFrame(all_rows, infer_schema_length=None)
    df = df.cast({
        'window_id': pl.UInt32,
        'n_obs': pl.UInt32,
    })

    write_output(df, data_path, 'typology_vector', verbose=verbose)

    if verbose:
        print(f"\n  Typology vector: {df.height} rows × {df.width} columns")

    return df
