"""
Stage 00a: Typology Vector
===========================

Compute typology metrics per window per signal.
Summarize whether they vary.
Output as data. That's it.

Downstream layers decide what to do with this information.
This stage only measures.

Output:
  1. typology_windows.parquet — per-window metrics (signal/ directory)
  2. typology_vector.parquet  — per-signal summary  (signal/ directory)

11 metrics per window:
  hurst, perm_entropy, sample_entropy, lyapunov_proxy,
  spectral_flatness, kurtosis, cv, mean_abs_diff,
  range_norm, zero_crossing_rate, trend_strength

pmtvs is a performance upgrade, not a dependency.
When pmtvs is installed: Rust-accelerated primitives.
When it's not: numpy/scipy fallbacks (slower, mathematically equivalent).
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional

from joblib import Parallel, delayed
from scipy.stats import kurtosis as sp_kurtosis

from manifold.io.writer import write_output


# ── pmtvs: optional accelerator ─────────────────────────────────

try:
    from manifold.core._pmtvs import (
        hurst_exponent as _pmtvs_hurst,
        permutation_entropy as _pmtvs_perm_entropy,
        sample_entropy as _pmtvs_sample_entropy,
        lyapunov_rosenstein as _pmtvs_lyapunov,
    )
    _HAS_PMTVS = True
except ImportError:
    _HAS_PMTVS = False


# Match signal_vector worker count
_N_WORKERS = 2

# CV threshold for "varies" — default, Prime can override
_CV_THRESHOLD = 0.10


# ── Metric functions (pmtvs + fallback) ─────────────────────────

def _hurst(x: np.ndarray) -> float:
    if _HAS_PMTVS:
        try:
            return float(_pmtvs_hurst(x))
        except Exception:
            pass
    # R/S fallback
    n = len(x)
    if n < 20:
        return np.nan
    max_k = min(n // 2, 100)
    if max_k < 10:
        return np.nan
    rs_values = []
    for k in range(10, max_k + 1, max(1, (max_k - 10) // 10)):
        n_chunks = n // k
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = x[i * k:(i + 1) * k]
            mean_c = np.mean(chunk)
            y = np.cumsum(chunk - mean_c)
            r = np.max(y) - np.min(y)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((np.log(k), np.log(np.mean(rs_list))))
    if len(rs_values) < 3:
        return np.nan
    logs = np.array(rs_values)
    coeffs = np.polyfit(logs[:, 0], logs[:, 1], 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def _perm_entropy(x: np.ndarray, order: int = 3) -> float:
    if _HAS_PMTVS:
        try:
            return float(_pmtvs_perm_entropy(x, order=order))
        except Exception:
            pass
    n = len(x)
    if n < order + 1:
        return np.nan
    from math import factorial
    max_perms = factorial(order)
    counts = {}
    for i in range(n - order):
        pattern = tuple(np.argsort(x[i:i + order]))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    h = -np.sum(probs * np.log2(probs + 1e-12))
    return float(h / np.log2(max_perms))


def _sample_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    if _HAS_PMTVS:
        try:
            return float(_pmtvs_sample_entropy(x, m=m))
        except Exception:
            pass
    n = len(x)
    if n < m + 2:
        return np.nan
    r = r_mult * np.std(x, ddof=1)
    if r < 1e-10:
        return np.nan

    def _count_matches(length):
        templates = np.array([x[i:i + length] for i in range(n - length)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    a = _count_matches(m)
    b = _count_matches(m + 1)
    if a == 0:
        return np.nan
    return float(-np.log(b / a)) if b > 0 else np.nan


def _lyapunov_proxy(x: np.ndarray) -> float:
    if _HAS_PMTVS:
        try:
            result = _pmtvs_lyapunov(x)
            # lyapunov_rosenstein returns (exponent, divergence_curve, lags) tuple
            if isinstance(result, tuple):
                return float(result[0])
            return float(result)
        except Exception:
            pass
    # Fallback: average log divergence of nearest neighbors
    n = len(x)
    if n < 50:
        return np.nan
    diffs = np.abs(np.diff(x))
    diffs = diffs[diffs > 1e-12]
    if len(diffs) < 10:
        return 0.0
    return float(np.mean(np.log(diffs + 1e-12)) - np.log(1e-12))


def _spectral_flatness(x: np.ndarray) -> float:
    ps = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    ps = ps[1:]  # drop DC
    if len(ps) == 0 or np.max(ps) < 1e-20:
        return 0.0
    geo = np.exp(np.mean(np.log(ps + 1e-20)))
    arith = np.mean(ps)
    if arith < 1e-20:
        return 0.0
    return float(np.clip(geo / arith, 0.0, 1.0))


def _kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    return float(sp_kurtosis(x, fisher=True))


def _cv(x: np.ndarray) -> float:
    mu = np.mean(x)
    if abs(mu) < 1e-10:
        return float(np.std(x, ddof=1))
    return float(np.std(x, ddof=1) / abs(mu))


def _range_norm(x: np.ndarray) -> float:
    r = np.max(x) - np.min(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(r / s)


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


# ── Per-window computation ──────────────────────────────────────

def compute_window_typology(values: np.ndarray) -> dict:
    """Compute all typology metrics for a single window of data."""
    result = {}
    for name, fn in METRICS:
        try:
            result[name] = fn(values)
        except Exception:
            result[name] = np.nan
    return result


def _compute_single_signal(
    signal_id: str,
    signal_data: np.ndarray,
    signal_0_data: np.ndarray,
    system_window: int,
    system_stride: int,
    cohort: str = '',
) -> List[Dict[str, Any]]:
    """Compute typology metrics for all windows of one signal.

    Returns list of row dicts, one per window.
    """
    n = len(signal_data)
    rows = []
    window_id = 0

    if n < system_window:
        # Signal shorter than one window: compute on entire signal
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


# ── Summary: per-signal aggregation ─────────────────────────────

def _summarize_windows(windows_df: pl.DataFrame) -> pl.DataFrame:
    """Summarize per-window metrics into per-signal typology_vector.

    For each metric: mean, std, cv, varies (bool: cv > threshold).
    Pure math on the window data. Not interpretation.

    Returns one row per (signal_id, cohort).
    """
    group_cols = ['signal_id']
    if 'cohort' in windows_df.columns:
        group_cols.append('cohort')

    # Available metrics (intersect with columns)
    metrics = [m for m in METRIC_NAMES if m in windows_df.columns]

    agg_exprs = [pl.len().alias('n_windows')]

    for m in metrics:
        col = pl.col(m)
        mean_expr = col.mean()
        std_expr = col.std()

        agg_exprs.append(mean_expr.alias(f'{m}_mean'))
        agg_exprs.append(std_expr.alias(f'{m}_std'))

    result = windows_df.group_by(group_cols).agg(agg_exprs)

    # Compute CV and varies as derived columns (can't do abs(mean) inside agg easily)
    for m in metrics:
        mean_col = f'{m}_mean'
        std_col = f'{m}_std'
        cv_col = f'{m}_cv'
        varies_col = f'{m}_varies'

        result = result.with_columns(
            pl.when(pl.col(mean_col).abs() > 1e-10)
            .then(pl.col(std_col) / pl.col(mean_col).abs())
            .otherwise(pl.col(std_col))  # absolute variation when mean is near zero
            .alias(cv_col)
        )
        result = result.with_columns(
            (pl.col(cv_col) > _CV_THRESHOLD).alias(varies_col)
        )

    return result


# ── Stage entry point ───────────────────────────────────────────

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
        print(f"  pmtvs: {'accelerated' if _HAS_PMTVS else 'fallback (numpy/scipy)'}")
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

    # Parallel computation
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
        _write_empty(data_path, verbose)
        return pl.DataFrame()

    # Build windows DataFrame
    windows_df = pl.DataFrame(all_rows, infer_schema_length=None)
    windows_df = windows_df.cast({'window_id': pl.UInt32, 'n_obs': pl.UInt32})

    # Summarize into per-signal vector
    vector_df = _summarize_windows(windows_df)

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
        print(f"  Vector:  {vector_df.height} signals, {n_varies} metric-signal pairs vary (CV > {_CV_THRESHOLD})")

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
