"""
Unified Fingerprint Feature Module
===================================

Extracts a fixed-width feature vector per engine from raw sensor data.
Identical computation for train and test — one function, same code,
same windows, both sides.

Architecture:
    raw sensor data [cohort, signal_id, signal_0, value]
    → per-window metrics (18 metrics per window)
    → per-metric curve features (25 features per metric)
    → 18 × 25 = 450 features per engine

Public API:
    compute_window_metrics(sensor_matrix)    — numpy array in, dict out
    extract_curve_features(series, name)     — numpy array in, dict out
    compute_engine_fingerprint(sensor_df)    — polars DataFrame in, dict out
    build_fingerprint_matrix(observations)   — polars DataFrame in, DataFrame out

Pure compute — no file I/O in compute functions.
"""

import numpy as np
import polars as pl
from scipy import stats


# ── Constants ──────────────────────────────────────────────────────

WINDOW_METRICS = [
    # Geometry (7) — eigendecomposition of covariance
    'effective_dim', 'condition_number', 'eigenvalue_3', 'total_variance',
    'eigenvalue_entropy', 'ratio_2_1', 'ratio_3_1',
    # Typology on centroid (5)
    'hurst', 'perm_entropy', 'kurtosis', 'trend_strength', 'zero_crossing_rate',
    # Signal geometry (2)
    'mean_dist_to_centroid', 'mean_abs_correlation',
    # Centroid extras (4)
    'centroid_kurtosis', 'centroid_perm_entropy',
    'centroid_spectral_flatness', 'algebraic_connectivity',
]

CURVE_FEATURE_SUFFIXES = [
    # 0th order (15)
    'mean', 'std', 'min', 'max', 'first', 'last', 'delta', 'range',
    'spike_ratio', 'slope', 'r2', 'early_mean', 'mid_mean', 'late_mean',
    'early_late_delta',
    # 1st order (5)
    'vel_mean', 'vel_std', 'vel_max', 'vel_min', 'vel_late_mean',
    # 2nd order (5)
    'acc_mean', 'acc_std', 'acc_max', 'acc_min', 'curvature_mean',
]


# ── Core compute functions ─────────────────────────────────────────

def compute_window_metrics(sensor_matrix: np.ndarray) -> dict:
    """18 metrics from one window's sensor matrix (n_signals × window_size).

    This is THE function that ensures train/test identity. Both sides
    call this with the same shaped input → same metrics.

    Args:
        sensor_matrix: 2D array (n_signals, window_size). Z-score normalized.

    Returns:
        dict of 18 metric_name → float
    """
    n_signals, n_obs = sensor_matrix.shape
    result = {}

    # ── Geometry: eigendecomposition of covariance ──
    valid = sensor_matrix[np.all(np.isfinite(sensor_matrix), axis=1)]
    if len(valid) >= 2:
        cov = np.cov(valid)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_var = eigenvalues.sum()

        if total_var > 0:
            p = eigenvalues / total_var
            result['effective_dim'] = float(total_var ** 2 / np.sum(eigenvalues ** 2))
            result['condition_number'] = float(
                eigenvalues[0] / max(eigenvalues[-1], 1e-15)
            )
            result['total_variance'] = float(total_var)
            result['eigenvalue_3'] = (
                float(eigenvalues[2]) if len(eigenvalues) > 2 else np.nan
            )
            p_pos = p[p > 0]
            result['eigenvalue_entropy'] = float(-np.sum(p_pos * np.log(p_pos)))
            result['ratio_2_1'] = (
                float(eigenvalues[1] / eigenvalues[0])
                if len(eigenvalues) > 1 and eigenvalues[0] > 0
                else np.nan
            )
            result['ratio_3_1'] = (
                float(eigenvalues[2] / eigenvalues[0])
                if len(eigenvalues) > 2 and eigenvalues[0] > 0
                else np.nan
            )
        else:
            for k in ['effective_dim', 'condition_number', 'total_variance',
                       'eigenvalue_3', 'eigenvalue_entropy', 'ratio_2_1', 'ratio_3_1']:
                result[k] = np.nan
    else:
        for k in ['effective_dim', 'condition_number', 'total_variance',
                   'eigenvalue_3', 'eigenvalue_entropy', 'ratio_2_1', 'ratio_3_1']:
            result[k] = np.nan

    # ── Centroid + distances ──
    centroid = np.nanmean(sensor_matrix, axis=0)
    distances = np.sqrt(np.nansum((sensor_matrix - centroid) ** 2, axis=1))
    result['mean_dist_to_centroid'] = float(np.nanmean(distances))

    # ── Correlation matrix → mean |correlation| ──
    if len(valid) >= 2:
        corr = np.corrcoef(valid)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        if mask.sum() > 0:
            result['mean_abs_correlation'] = float(np.nanmean(np.abs(corr[mask])))
        else:
            result['mean_abs_correlation'] = np.nan
    else:
        corr = np.array([[1.0]])
        result['mean_abs_correlation'] = np.nan

    # ── Typology on centroid ──
    from manifold.core.typology import compute_window_typology

    typo = compute_window_typology(centroid)
    for key in ['hurst', 'perm_entropy', 'kurtosis', 'trend_strength',
                'zero_crossing_rate']:
        result[key] = typo.get(key, np.nan)

    # ── Centroid spectral features ──
    result['centroid_kurtosis'] = typo.get('kurtosis', np.nan)
    result['centroid_perm_entropy'] = typo.get('perm_entropy', np.nan)
    result['centroid_spectral_flatness'] = typo.get('spectral_flatness', np.nan)

    # ── Algebraic connectivity (graph Laplacian) ──
    if len(valid) >= 3:
        adj = np.abs(corr)
        np.fill_diagonal(adj, 0.0)
        try:
            from manifold.core._compat import graph_laplacian
            L = graph_laplacian(adj, normalized=True)
            eigs = np.sort(np.real(np.linalg.eigvalsh(L)))
            result['algebraic_connectivity'] = (
                float(eigs[1]) if len(eigs) >= 2 else np.nan
            )
        except Exception:
            result['algebraic_connectivity'] = np.nan
    else:
        result['algebraic_connectivity'] = np.nan

    return result


def extract_curve_features(series: np.ndarray, name: str) -> dict:
    """25 features from a per-window metric time series.

    Args:
        series: 1D array of metric values across windows.
        name: metric name prefix for feature column names.

    Returns:
        dict of 25 features: {name}_{suffix} → float
    """
    result = {}
    p = f"{name}_"
    n = len(series)
    valid = series[~np.isnan(series)] if n > 0 else np.array([])

    # ── 0th order (15) ──
    result[f'{p}mean'] = float(np.nanmean(series)) if len(valid) > 0 else np.nan
    result[f'{p}std'] = float(np.nanstd(series)) if len(valid) > 0 else np.nan
    result[f'{p}min'] = float(np.nanmin(series)) if len(valid) > 0 else np.nan
    result[f'{p}max'] = float(np.nanmax(series)) if len(valid) > 0 else np.nan
    result[f'{p}first'] = float(series[0]) if n > 0 else np.nan
    result[f'{p}last'] = float(series[-1]) if n > 0 else np.nan
    result[f'{p}delta'] = float(series[-1] - series[0]) if n > 1 else np.nan
    result[f'{p}range'] = (
        float(np.nanmax(series) - np.nanmin(series)) if len(valid) > 0 else np.nan
    )

    # Spike ratio: max / early_mean
    if len(valid) > 0:
        early_mean = np.nanmean(valid[:max(1, len(valid) // 3)])
        result[f'{p}spike_ratio'] = (
            float(np.nanmax(series) / early_mean)
            if abs(early_mean) > 1e-10 else np.nan
        )
    else:
        result[f'{p}spike_ratio'] = np.nan

    # Linear fit
    if len(valid) >= 3:
        x = np.arange(n)[~np.isnan(series)]
        slope, intercept, r, _, _ = stats.linregress(x, valid)
        result[f'{p}slope'] = float(slope)
        result[f'{p}r2'] = float(r ** 2)
    else:
        result[f'{p}slope'] = np.nan
        result[f'{p}r2'] = np.nan

    # Thirds
    third = max(1, n // 3)
    early = series[:third]
    mid = series[third:2 * third]
    late = series[-third:]
    result[f'{p}early_mean'] = (
        float(np.nanmean(early)) if len(early) > 0 and np.any(np.isfinite(early))
        else np.nan
    )
    result[f'{p}mid_mean'] = (
        float(np.nanmean(mid)) if len(mid) > 0 and np.any(np.isfinite(mid))
        else np.nan
    )
    result[f'{p}late_mean'] = (
        float(np.nanmean(late)) if len(late) > 0 and np.any(np.isfinite(late))
        else np.nan
    )
    result[f'{p}early_late_delta'] = (
        result[f'{p}late_mean'] - result[f'{p}early_mean']
        if not (np.isnan(result[f'{p}late_mean']) or np.isnan(result[f'{p}early_mean']))
        else np.nan
    )

    # ── 1st order: velocity (5) ──
    if n >= 3:
        vel = np.diff(series)
        vel_clean = vel[~np.isnan(vel)]
        if len(vel_clean) > 0:
            result[f'{p}vel_mean'] = float(np.mean(vel_clean))
            result[f'{p}vel_std'] = float(np.std(vel_clean))
            result[f'{p}vel_max'] = float(np.max(vel_clean))
            result[f'{p}vel_min'] = float(np.min(vel_clean))
            late_vel = vel_clean[-max(1, len(vel_clean) // 3):]
            result[f'{p}vel_late_mean'] = float(np.mean(late_vel))
        else:
            for s in ['vel_mean', 'vel_std', 'vel_max', 'vel_min', 'vel_late_mean']:
                result[f'{p}{s}'] = np.nan
    else:
        for s in ['vel_mean', 'vel_std', 'vel_max', 'vel_min', 'vel_late_mean']:
            result[f'{p}{s}'] = np.nan

    # ── 2nd order: acceleration + curvature (5) ──
    if n >= 4:
        acc = np.diff(series, n=2)
        acc_clean = acc[~np.isnan(acc)]
        if len(acc_clean) > 0:
            result[f'{p}acc_mean'] = float(np.mean(acc_clean))
            result[f'{p}acc_std'] = float(np.std(acc_clean))
            result[f'{p}acc_max'] = float(np.max(acc_clean))
            result[f'{p}acc_min'] = float(np.min(acc_clean))
        else:
            for s in ['acc_mean', 'acc_std', 'acc_max', 'acc_min']:
                result[f'{p}{s}'] = np.nan

        # Discrete curvature: |acc| / (1 + vel²)^1.5
        if n >= 3:
            vel_full = np.diff(series)
            acc_full = np.diff(series, n=2)
            # Align: vel_full[1:] corresponds to acc_full[:]
            vel_aligned = vel_full[1:]
            min_len = min(len(vel_aligned), len(acc_full))
            if min_len > 0:
                v = vel_aligned[:min_len]
                a = acc_full[:min_len]
                # Filter NaN from both
                mask = ~(np.isnan(v) | np.isnan(a))
                if mask.sum() > 0:
                    kappa = np.abs(a[mask]) / (1 + v[mask] ** 2) ** 1.5
                    result[f'{p}curvature_mean'] = float(np.mean(kappa))
                else:
                    result[f'{p}curvature_mean'] = np.nan
            else:
                result[f'{p}curvature_mean'] = np.nan
        else:
            result[f'{p}curvature_mean'] = np.nan
    else:
        for s in ['acc_mean', 'acc_std', 'acc_max', 'acc_min', 'curvature_mean']:
            result[f'{p}{s}'] = np.nan

    return result


def compute_engine_fingerprint(
    sensor_df: pl.DataFrame,
    window: int = 50,
    stride: int = 24,
) -> dict:
    """Full pipeline: sensor data → 450 features for one engine.

    Args:
        sensor_df: DataFrame with columns [signal_id, signal_0, value]
                   for a single engine. Values should already be z-score
                   normalized using TRAINING statistics.
        window: window size in cycles (signal_0 units).
        stride: stride between windows.

    Returns:
        dict of ~450 feature_name → float
    """
    # Pivot: each signal → one row of the matrix
    signal_ids = sorted(sensor_df['signal_id'].unique().to_list())
    rows = []
    for sid in signal_ids:
        sig_data = sensor_df.filter(pl.col('signal_id') == sid).sort('signal_0')
        rows.append(sig_data['value'].to_numpy().astype(float))

    # Truncate to shortest signal length
    min_len = min(len(r) for r in rows) if rows else 0
    if min_len == 0:
        return {}
    rows = [r[:min_len] for r in rows]
    matrix = np.array(rows)  # (n_signals, n_cycles)

    n_signals, n_total = matrix.shape
    if n_total < window:
        # Not enough data for even one window — compute from full data
        window = n_total

    # Slide windows
    starts = list(range(0, n_total - window + 1, stride))
    if not starts:
        starts = [0]

    # Compute per-window metrics
    metric_arrays = {m: [] for m in WINDOW_METRICS}
    for start in starts:
        win_matrix = matrix[:, start:start + window]
        metrics = compute_window_metrics(win_matrix)
        for m in WINDOW_METRICS:
            metric_arrays[m].append(metrics.get(m, np.nan))

    # Convert to numpy arrays
    for m in WINDOW_METRICS:
        metric_arrays[m] = np.array(metric_arrays[m], dtype=float)

    # Extract curve features for each metric
    features = {}
    for m in WINDOW_METRICS:
        features.update(extract_curve_features(metric_arrays[m], m))

    return features


def build_fingerprint_matrix(
    observations: pl.DataFrame,
    window: int = 50,
    stride: int = 24,
    verbose: bool = True,
) -> pl.DataFrame:
    """Build feature matrix for all engines.

    Args:
        observations: DataFrame with [cohort, signal_id, signal_0, value].
                      Already normalized (caller handles z-score with training stats).
        window: window size.
        stride: stride between windows.
        verbose: print progress.

    Returns:
        DataFrame: one row per cohort, ~450 feature columns.
    """
    engines = sorted(observations['cohort'].unique().to_list())
    if verbose:
        print(f"Building fingerprint matrix: {len(engines)} engines, "
              f"window={window}, stride={stride}")

    rows = []
    for i, eng in enumerate(engines):
        eng_data = observations.filter(pl.col('cohort') == eng)
        features = compute_engine_fingerprint(eng_data, window=window, stride=stride)
        features['cohort'] = eng
        rows.append(features)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(engines)} engines processed")

    if verbose:
        print(f"  {len(engines)}/{len(engines)} engines processed")

    result = pl.DataFrame(rows)

    # Move cohort to first column
    cols = ['cohort'] + [c for c in result.columns if c != 'cohort']
    result = result.select(cols)

    if verbose:
        n_feats = len(result.columns) - 1
        print(f"Fingerprint matrix: {len(result)} engines × {n_feats} features")

    return result


# ── CMAPSS helpers ─────────────────────────────────────────────────

def _read_cmapss_txt(path: str) -> pl.DataFrame:
    """Read a CMAPSS raw text file into a DataFrame.

    Standard CMAPSS format: space-separated, no header.
    Columns: unit, cycle, op1, op2, op3, s1..s21
    """
    col_names = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [
        f's{i}' for i in range(1, 22)
    ]
    df = pl.read_csv(
        path,
        separator=' ',
        has_header=False,
        new_columns=col_names,
        truncate_ragged_lines=True,
    )
    # Drop any trailing null columns from extra whitespace
    df = df.select([c for c in df.columns if c in col_names])
    return df


def _cmapss_to_manifold(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot CMAPSS wide format to manifold long format.

    Returns DataFrame with [cohort, signal_id, signal_0, value].
    """
    sensor_cols = [f's{i}' for i in range(1, 22)]
    rows = []
    for col in sensor_cols:
        subset = df.select([
            pl.col('unit').cast(pl.Utf8).alias('cohort'),
            pl.lit(col).alias('signal_id'),
            (pl.col('cycle') - 1).alias('signal_0'),  # 0-indexed
            pl.col(col).cast(pl.Float64).alias('value'),
        ])
        rows.append(subset)
    return pl.concat(rows)


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Unified fingerprint feature extraction',
    )
    parser.add_argument('--observations', help='observations.parquet path')
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--stride', type=int, default=24)
    parser.add_argument('-o', '--output', default='fingerprint_features.parquet')
    parser.add_argument('-q', '--quiet', action='store_true')

    # CMAPSS convenience
    parser.add_argument('--cmapss-train', help='CMAPSS train_FDXXX.txt')
    parser.add_argument('--cmapss-test', help='CMAPSS test_FDXXX.txt')
    parser.add_argument('--cmapss-rul', help='CMAPSS RUL_FDXXX.txt')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.cmapss_train:
        # ── CMAPSS mode ──
        if verbose:
            print("CMAPSS mode: reading raw text files...")

        train_raw = _read_cmapss_txt(args.cmapss_train)
        train_obs = _cmapss_to_manifold(train_raw)

        # Compute training stats ONCE
        train_stats = train_obs.group_by('signal_id').agg([
            pl.col('value').mean().alias('mu'),
            pl.col('value').std().alias('sigma'),
        ])
        # Replace zero sigma with 1.0 to avoid division by zero
        train_stats = train_stats.with_columns(
            pl.when(pl.col('sigma').abs() < 1e-10)
            .then(1.0)
            .otherwise(pl.col('sigma'))
            .alias('sigma')
        )

        def normalize(obs, stat):
            return (
                obs.join(stat, on='signal_id')
                .with_columns(
                    ((pl.col('value') - pl.col('mu')) / pl.col('sigma'))
                    .alias('value')
                )
                .drop(['mu', 'sigma'])
            )

        train_norm = normalize(train_obs, train_stats)
        train_features = build_fingerprint_matrix(
            train_norm, window=args.window, stride=args.stride, verbose=verbose,
        )
        train_features = train_features.with_columns(pl.lit('train').alias('split'))

        all_features = train_features

        if args.cmapss_test:
            test_raw = _read_cmapss_txt(args.cmapss_test)
            test_obs = _cmapss_to_manifold(test_raw)
            test_norm = normalize(test_obs, train_stats)  # SAME stats
            test_features = build_fingerprint_matrix(
                test_norm, window=args.window, stride=args.stride, verbose=verbose,
            )
            test_features = test_features.with_columns(pl.lit('test').alias('split'))

            # Attach RUL if provided
            if args.cmapss_rul:
                rul_values = []
                with open(args.cmapss_rul) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rul_values.append(int(line))
                rul_df = pl.DataFrame({
                    'cohort': [str(i + 1) for i in range(len(rul_values))],
                    'rul': rul_values,
                })
                test_features = test_features.join(rul_df, on='cohort', how='left')

            all_features = pl.concat(
                [train_features, test_features], how='diagonal_relaxed',
            )

        # Attach train RUL (max cycle per engine)
        if 'rul' not in all_features.columns or all_features.filter(
            pl.col('split') == 'train'
        )['rul'].null_count() == all_features.filter(
            pl.col('split') == 'train'
        ).height:
            max_cycles = (
                train_raw.group_by('unit')
                .agg(pl.col('cycle').max().alias('rul'))
                .with_columns(pl.col('unit').cast(pl.Utf8).alias('cohort'))
                .select(['cohort', 'rul'])
            )
            if 'rul' in all_features.columns:
                # Fill in train RUL only
                all_features = all_features.join(
                    max_cycles.rename({'rul': 'rul_train'}),
                    on='cohort', how='left',
                ).with_columns(
                    pl.when(pl.col('split') == 'train')
                    .then(pl.col('rul_train'))
                    .otherwise(pl.col('rul'))
                    .alias('rul')
                ).drop('rul_train')
            else:
                all_features = all_features.join(max_cycles, on='cohort', how='left')

        all_features.write_parquet(args.output)
        if verbose:
            print(f"Saved: {args.output} ({all_features.height} rows)")

    elif args.observations:
        # ── Generic mode ──
        obs = pl.read_parquet(args.observations)
        features = build_fingerprint_matrix(
            obs, window=args.window, stride=args.stride, verbose=verbose,
        )
        features.write_parquet(args.output)
        if verbose:
            print(f"Saved: {args.output}")

    else:
        parser.error("Provide --observations or --cmapss-train")
