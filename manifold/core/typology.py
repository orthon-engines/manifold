"""
Typology Engine
===============

Compute typology metrics for a window of signal data.

Public API:
    compute_window_typology(values)  — numpy array in, dict out
    compute_signal_typology(...)     — all windows for one signal
    summarize_windows(windows_df)    — per-signal aggregation

11 metrics per window:
    hurst, perm_entropy, sample_entropy, lyapunov_proxy,
    spectral_flatness, kurtosis, cv, mean_abs_diff,
    range_norm, zero_crossing_rate, trend_strength

Pure compute — no file I/O.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any

from manifold.core._pmtvs import (
    hurst_exponent,
    permutation_entropy,
    sample_entropy,
    lyapunov_rosenstein,
    spectral_flatness as pmtvs_spectral_flatness,
    kurtosis as pmtvs_kurtosis,
)

# CV threshold for "varies" — default, Prime can override
CV_THRESHOLD = 0.10


# ── Metric functions ─────────────────────────────────────────────

def _hurst(x: np.ndarray) -> float:
    if len(x) < 20:
        return np.nan
    return float(hurst_exponent(x))


def _perm_entropy(x: np.ndarray, order: int = 3) -> float:
    if len(x) < order + 1:
        return np.nan
    return float(permutation_entropy(x, order=order))


def _sample_entropy(x: np.ndarray, m: int = 2) -> float:
    if len(x) < m + 2:
        return np.nan
    return float(sample_entropy(x, m=m))


def _lyapunov_proxy(x: np.ndarray) -> float:
    if len(x) < 50:
        return np.nan
    result = lyapunov_rosenstein(x)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def _spectral_flatness(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    return float(pmtvs_spectral_flatness(x))


def _kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    return float(pmtvs_kurtosis(x, fisher=True))


def _cv(x: np.ndarray) -> float:
    mu = np.mean(x)
    if abs(mu) < 1e-10:
        return float(np.std(x, ddof=1))
    return float(np.std(x, ddof=1) / abs(mu))


def _range_norm(x: np.ndarray) -> float:
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float((np.max(x) - np.min(x)) / s)


def _zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    centered = x - np.mean(x)
    crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
    return float(crossings / (len(x) - 1))


def _trend_strength(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return np.nan
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, x, 1)
    fitted = np.polyval(coeffs, t)
    ss_res = np.sum((x - fitted) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    if ss_tot < 1e-20:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _mean_abs_diff(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))))


# All typology metrics, in order
METRICS = [
    ("hurst",              _hurst),
    ("perm_entropy",       _perm_entropy),
    ("sample_entropy",     _sample_entropy),
    ("lyapunov_proxy",     _lyapunov_proxy),
    ("spectral_flatness",  _spectral_flatness),
    ("kurtosis",           _kurtosis),
    ("cv",                 _cv),
    ("range_norm",         _range_norm),
    ("zero_crossing_rate", _zero_crossing_rate),
    ("trend_strength",     _trend_strength),
    ("mean_abs_diff",      _mean_abs_diff),
]

METRIC_NAMES = [name for name, _ in METRICS]


# ── Public API ───────────────────────────────────────────────────

def compute_window_typology(values: np.ndarray) -> dict:
    """Compute all typology metrics for a single window of data.

    Args:
        values: 1-D numpy array of signal values for one window.

    Returns:
        Dict mapping metric name -> float value. NaN for insufficient
        data or computation failures. Inf values are replaced with NaN.
    """
    result = {}
    for name, fn in METRICS:
        try:
            v = fn(values)
            result[name] = np.nan if (v is None or np.isinf(v)) else v
        except Exception:
            result[name] = np.nan
    return result


def compute_signal_typology(
    signal_data: np.ndarray,
    signal_0_data: np.ndarray,
    system_window: int,
    system_stride: int,
    signal_id: str = '',
    cohort: str = '',
) -> List[Dict[str, Any]]:
    """Compute typology metrics for all windows of one signal.

    Args:
        signal_data: 1-D array of signal values, sorted by signal_0.
        signal_0_data: 1-D array of signal_0 indices, same length.
        system_window: Window size in samples.
        system_stride: Stride between windows in samples.
        signal_id: Signal identifier (passed through).
        cohort: Cohort identifier (passed through).

    Returns:
        List of row dicts, one per window.
    """
    n = len(signal_data)
    rows = []
    window_id = 0

    if n < system_window:
        metrics = compute_window_typology(signal_data)
        rows.append({
            'window_id': 0,
            'signal_id': signal_id,
            'cohort': cohort,
            'signal_0_start': float(signal_0_data[0]),
            'signal_0_end': float(signal_0_data[-1]),
            'signal_0_center': (float(signal_0_data[0]) + float(signal_0_data[-1])) / 2,
            'n_obs': n,
            **metrics,
        })
        return rows

    for window_end in range(system_window - 1, n, system_stride):
        window_start = max(0, window_end - system_window + 1)
        window_data = signal_data[window_start:window_end + 1]
        metrics = compute_window_typology(window_data)

        rows.append({
            'window_id': window_id,
            'signal_id': signal_id,
            'cohort': cohort,
            'signal_0_start': float(signal_0_data[window_start]),
            'signal_0_end': float(signal_0_data[window_end]),
            'signal_0_center': (float(signal_0_data[window_start]) + float(signal_0_data[window_end])) / 2,
            'n_obs': len(window_data),
            **metrics,
        })
        window_id += 1

    return rows


def summarize_windows(windows_df: pl.DataFrame, cv_threshold: float = CV_THRESHOLD) -> pl.DataFrame:
    """Summarize per-window metrics into per-signal typology_vector.

    For each metric: mean, std, cv, varies (bool: cv > threshold).

    Args:
        windows_df: DataFrame with per-window typology metrics.
        cv_threshold: CV threshold for "varies" flag.

    Returns:
        One row per (signal_id, cohort).
    """
    group_cols = ['signal_id']
    if 'cohort' in windows_df.columns:
        group_cols.append('cohort')

    metrics = [m for m in METRIC_NAMES if m in windows_df.columns]

    agg_exprs = [pl.len().alias('n_windows')]
    for m in metrics:
        col = pl.col(m)
        agg_exprs.append(col.mean().alias(f'{m}_mean'))
        agg_exprs.append(col.std().alias(f'{m}_std'))

    result = windows_df.group_by(group_cols).agg(agg_exprs)

    for m in metrics:
        mean_col = f'{m}_mean'
        std_col = f'{m}_std'
        cv_col = f'{m}_cv'
        varies_col = f'{m}_varies'

        result = result.with_columns(
            pl.when(pl.col(mean_col).abs() > 1e-10)
            .then(pl.col(std_col) / pl.col(mean_col).abs())
            .otherwise(pl.col(std_col))
            .alias(cv_col)
        )
        result = result.with_columns(
            (pl.col(cv_col) > cv_threshold).alias(varies_col)
        )

    return result
