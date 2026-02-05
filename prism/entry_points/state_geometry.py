"""
PRISM State Geometry Engine

State geometry computes the SHAPE of the signal distribution around each state.
This is where eigenvalues live - they describe relationships between signals,
not the state position itself.

Computes per engine, per index:
- Eigenvalues (via SVD)
- effective_dim (from eigenvalues)
- Total variance, condition number
- Eigenvalue entropy

REQUIRES: signal_vector.parquet + state_vector.parquet

Python first (SVD for eigenvalues), then SQL for aggregations.

Pipeline:
    signal_vector.parquet + state_vector.parquet → state_geometry.parquet

Normalization Options (v2.5):
- zscore: Standard (x-mean)/std - sensitive to outliers (default for backward compat)
- robust: (x-median)/IQR - moderate robustness to outliers
- mad: (x-median)/MAD - most robust, recommended for industrial/financial data
- none: Raw covariance (preserves actual variance dynamics)
"""

import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Literal

# Import normalization engine
from prism.engines.normalization import normalize, recommend_method, NormMethod


# ============================================================
# DEFAULT ENGINE FEATURE GROUPS - imported from config
# ============================================================

try:
    from prism.engines.geometry.config import DEFAULT_FEATURE_GROUPS
except ImportError:
    # Fallback if config not available
    DEFAULT_FEATURE_GROUPS = {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    }


# ============================================================
# EIGENVALUE COMPUTATION (Python - can't do SVD in SQL)
# ============================================================

def compute_eigenvalues(
    signal_matrix: np.ndarray,
    centroid: np.ndarray,
    min_signals: int = 3,
    norm_method: str = "zscore"
) -> Dict[str, Any]:
    """
    Compute eigenvalues of signal distribution around centroid.

    This is the SHAPE of the signal cloud - how signals spread
    around the state (centroid).

    Steps:
        1. Center signals around centroid
        2. Normalize (optional, so features are comparable)
        3. SVD to get eigenvalues

    Args:
        signal_matrix: N_signals × D_features
        centroid: D_features centroid from state_vector
        min_signals: Minimum signals for reliable eigenvalues
        norm_method: Normalization method (see prism.engines.normalization):
            - "zscore": Standard z-score (sensitive to outliers)
            - "robust": IQR-based (moderate robustness)
            - "mad": MAD-based (most robust, recommended for industrial data)
            - "none": Raw covariance (preserves actual variance dynamics)

    Returns:
        Eigenvalue metrics
    """
    N, D = signal_matrix.shape

    if N < min_signals:
        return _empty_eigenvalues(D)

    # Remove NaN/Inf
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    if valid_mask.sum() < min_signals:
        return _empty_eigenvalues(D)

    signal_matrix = signal_matrix[valid_mask]
    N = len(signal_matrix)

    # ─────────────────────────────────────────────────
    # CENTER SIGNALS AROUND CENTROID
    # ─────────────────────────────────────────────────
    centered = signal_matrix - centroid

    # ─────────────────────────────────────────────────
    # NORMALIZE BEFORE SVD (configurable method)
    # Without normalization, features with large variance
    # dominate eigenvalues, making them incomparable across time.
    #
    # Method selection guidance:
    # - zscore: Default, assumes Gaussian, sensitive to outliers
    # - robust: IQR-based, handles moderate outliers
    # - mad: MAD-based, most robust for industrial/financial data
    # - none: Raw covariance (preserves actual variance dynamics)
    # ─────────────────────────────────────────────────
    if norm_method.lower() == "none":
        normalized = centered
    else:
        # Use normalization engine
        normalized, _ = normalize(centered, method=norm_method, axis=0)

    # ─────────────────────────────────────────────────
    # SVD FOR EIGENVALUES
    # ─────────────────────────────────────────────────
    try:
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)

        # Eigenvalues of covariance = S² / (N-1)
        eigenvalues = (S ** 2) / max(N - 1, 1)

        # Principal components (rows of Vt)
        principal_components = Vt

    except np.linalg.LinAlgError:
        return _empty_eigenvalues(D)

    # ─────────────────────────────────────────────────
    # DERIVED METRICS
    # ─────────────────────────────────────────────────
    total_variance = eigenvalues.sum()

    if total_variance > 1e-10:
        # Effective dimension (participation ratio)
        effective_dim = (total_variance ** 2) / (eigenvalues ** 2).sum()

        # Explained variance ratios
        explained_ratios = eigenvalues / total_variance

        # Eigenvalue entropy
        nonzero = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero) > 1:
            p = nonzero / nonzero.sum()
            eigenvalue_entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(nonzero))
            eigenvalue_entropy_normalized = eigenvalue_entropy / max_entropy if max_entropy > 0 else 0
        else:
            eigenvalue_entropy = 0
            eigenvalue_entropy_normalized = 0

        # Condition number
        nonzero = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero) > 1:
            condition_number = nonzero[0] / nonzero[-1]
        else:
            condition_number = 1.0

        # Eigenvalue ratios (for multi-mode)
        if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-10:
            ratio_2_1 = eigenvalues[1] / eigenvalues[0]
        else:
            ratio_2_1 = 0

        if len(eigenvalues) >= 3 and eigenvalues[0] > 1e-10:
            ratio_3_1 = eigenvalues[2] / eigenvalues[0]
        else:
            ratio_3_1 = 0

    else:
        effective_dim = 0
        explained_ratios = np.zeros_like(eigenvalues)
        eigenvalue_entropy = 0
        eigenvalue_entropy_normalized = 0
        condition_number = 1.0
        ratio_2_1 = 0
        ratio_3_1 = 0

    return {
        'eigenvalues': eigenvalues,
        'explained_ratios': explained_ratios,
        'total_variance': total_variance,
        'effective_dim': effective_dim,
        'eigenvalue_entropy': eigenvalue_entropy,
        'eigenvalue_entropy_normalized': eigenvalue_entropy_normalized,
        'condition_number': condition_number,
        'ratio_2_1': ratio_2_1,
        'ratio_3_1': ratio_3_1,
        'principal_components': principal_components,  # Vt - feature loadings
        'signal_loadings': U,  # U - signal loadings on PCs
        'n_signals': N,
        'n_features': D,
    }


def _empty_eigenvalues(D: int, N: int = 4) -> Dict[str, Any]:
    """Return empty eigenvalue result for edge cases."""
    return {
        'eigenvalues': np.zeros(D),
        'explained_ratios': np.zeros(D),
        'total_variance': 0.0,
        'effective_dim': 0.0,
        'eigenvalue_entropy': 0.0,
        'eigenvalue_entropy_normalized': 0.0,
        'condition_number': 1.0,
        'ratio_2_1': 0.0,
        'ratio_3_1': 0.0,
        'principal_components': np.eye(D),
        'signal_loadings': np.eye(N, D),  # N signals × D components
        'n_signals': 0,
        'n_features': D,
    }


# ============================================================
# STATE GEOMETRY COMPUTATION
# ============================================================

def compute_state_geometry(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "state_geometry.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    max_eigenvalues: int = 5,
    norm_method: str = "zscore",
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state geometry (eigenvalues per engine per index).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path
        feature_groups: Dict mapping engine names to feature lists
        max_eigenvalues: Maximum eigenvalues to store
        norm_method: Normalization method before SVD:
            - "zscore": Standard (sensitive to outliers) - default
            - "robust": IQR-based (moderate robustness)
            - "mad": MAD-based (most robust for industrial data)
            - "none": Raw covariance (preserves variance dynamics)
        verbose: Print progress

    Returns:
        State geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STATE GEOMETRY ENGINE")
        print("Eigenvalues and shape metrics per engine")
        print(f"Normalization: {norm_method}")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

    # Identify features
    meta_cols = ['unit_id', 'I', 'signal_id']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    # Determine feature groups
    if feature_groups is None:
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups and len(all_features) >= 2:
            feature_groups['full'] = all_features[:3]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print()

    # Determine grouping columns - include cohort if present for per-unit analysis
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'I'] if has_cohort else ['I']

    # Process each group (cohort, I) or just (I)
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    # Track previous PC1 loadings for alignment computation (per cohort if available)
    # Key: (cohort, engine) or just engine if no cohort
    prev_pc1_by_key: Dict[Any, np.ndarray] = {}

    if verbose:
        if has_cohort:
            n_cohorts = signal_vector['cohort'].n_unique()
            print(f"Processing {n_groups} (cohort, I) groups across {n_cohorts} cohorts...")
        else:
            print(f"Processing {n_groups} time points...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, I = group_key if isinstance(group_key, tuple) else (group_key, None)
        else:
            cohort = None
            I = group_key[0] if isinstance(group_key, tuple) else group_key

        # Get state vector for this (cohort, I) or just I
        if has_cohort and 'cohort' in state_vector.columns:
            state_row = state_vector.filter(
                (pl.col('I') == I) & (pl.col('cohort') == cohort)
            )
        else:
            state_row = state_vector.filter(pl.col('I') == I)

        if len(state_row) == 0:
            continue

        # Compute eigenvalues for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get centroid from state_vector
            centroid_cols = [f'state_{engine_name}_{f}' for f in available]
            centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(centroid_available) != len(available):
                # Centroid not computed for this engine, compute from data
                matrix = group.select(available).to_numpy()
                valid_mask = np.isfinite(matrix).all(axis=1)
                if valid_mask.sum() > 0:
                    centroid = np.mean(matrix[valid_mask], axis=0)
                else:
                    continue
            else:
                centroid = state_row.select(centroid_available).to_numpy().flatten()

            # Get signal matrix
            matrix = group.select(available).to_numpy()

            # Compute eigenvalues with specified normalization
            eigen_result = compute_eigenvalues(matrix, centroid, norm_method=norm_method)

            # Build result row
            unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
            row = {
                'I': I,
                'engine': engine_name,
                'n_signals': eigen_result['n_signals'],
                'n_features': eigen_result['n_features'],
            }
            # Include cohort if available (for per-unit analysis)
            if cohort:
                row['cohort'] = cohort
            if unit_id:
                row['unit_id'] = unit_id

            # Eigenvalues
            for j in range(min(max_eigenvalues, len(eigen_result['eigenvalues']))):
                row[f'eigenvalue_{j+1}'] = float(eigen_result['eigenvalues'][j])
            for j in range(len(eigen_result['eigenvalues']), max_eigenvalues):
                row[f'eigenvalue_{j+1}'] = 0.0

            # Explained ratios
            for j in range(min(max_eigenvalues, len(eigen_result['explained_ratios']))):
                row[f'explained_{j+1}'] = float(eigen_result['explained_ratios'][j])
            for j in range(len(eigen_result['explained_ratios']), max_eigenvalues):
                row[f'explained_{j+1}'] = 0.0

            # Derived metrics
            row['effective_dim'] = eigen_result['effective_dim']
            row['total_variance'] = eigen_result['total_variance']
            row['eigenvalue_entropy'] = eigen_result['eigenvalue_entropy']
            row['eigenvalue_entropy_norm'] = eigen_result['eigenvalue_entropy_normalized']
            row['condition_number'] = eigen_result['condition_number']
            row['ratio_2_1'] = eigen_result['ratio_2_1']
            row['ratio_3_1'] = eigen_result['ratio_3_1']

            # PC1 loadings on FEATURES (first row of Vt)
            pc = eigen_result['principal_components']
            pc1 = None
            if pc is not None and len(pc) > 0:
                pc1 = pc[0] if len(pc.shape) > 1 else pc
                for j, feat in enumerate(available):
                    if j < len(pc1):
                        row[f'pc1_feat_{feat}'] = float(pc1[j])

            # PC1 ALIGNMENT with previous window (regime detection)
            # Correlation of current PC1 loadings with previous window's PC1
            # Use (cohort, engine) key to track per-unit alignment when cohort exists
            pc1_key = (cohort, engine_name) if cohort else engine_name

            if pc1 is not None and pc1_key in prev_pc1_by_key:
                prev_pc1 = prev_pc1_by_key[pc1_key]
                if len(pc1) == len(prev_pc1) and len(pc1) > 1:
                    # Safeguard against degenerate cases (constant PC1)
                    if np.std(pc1) > 1e-10 and np.std(prev_pc1) > 1e-10:
                        # Absolute correlation (sign can flip between windows)
                        alignment = abs(np.corrcoef(pc1, prev_pc1)[0, 1])
                        row['pc1_alignment'] = float(alignment) if np.isfinite(alignment) else 1.0
                    else:
                        # Constant PC = no rotation (same as previous)
                        row['pc1_alignment'] = 1.0
                else:
                    row['pc1_alignment'] = np.nan
            else:
                row['pc1_alignment'] = np.nan  # First window has no previous

            # Store current PC1 for next iteration
            if pc1 is not None:
                prev_pc1_by_key[pc1_key] = pc1.copy()

            # PC1 loadings on SIGNALS (first column of U)
            # This tells us how much each signal contributes to PC1
            U = eigen_result.get('signal_loadings')
            signal_ids = group['signal_id'].to_list() if 'signal_id' in group.columns else []
            if U is not None and len(U) > 0 and len(signal_ids) > 0:
                pc1_signal = U[:, 0] if len(U.shape) > 1 else U
                for j, sig_id in enumerate(signal_ids):
                    if j < len(pc1_signal):
                        row[f'pc1_signal_{sig_id}'] = float(pc1_signal[j])

            # Store signal names for this window (for alignment tracking)
            row['signal_ids'] = ','.join(signal_ids[:8])  # Comma-separated list

            results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary per engine
        for engine_name in feature_groups.keys():
            engine_data = result.filter(pl.col('engine') == engine_name)
            if len(engine_data) > 0:
                print(f"\n{engine_name} engine:")
                print(f"  effective_dim: mean={engine_data['effective_dim'].mean():.2f}, "
                      f"std={engine_data['effective_dim'].std():.2f}")
                print(f"  eigenvalue_1: mean={engine_data['eigenvalue_1'].mean():.4f}")

    return result


# ============================================================
# SQL AGGREGATIONS (after Python eigenvalue computation)
# ============================================================

STATE_GEOMETRY_SQL_AGGREGATIONS = """
-- Aggregate state geometry metrics
-- Run AFTER eigenvalue computation

-- Summary by unit
CREATE OR REPLACE VIEW v_state_geometry_by_unit AS
SELECT
    unit_id,
    engine,
    COUNT(*) AS n_indices,

    AVG(effective_dim) AS mean_effective_dim,
    STDDEV(effective_dim) AS std_effective_dim,
    MIN(effective_dim) AS min_effective_dim,
    MAX(effective_dim) AS max_effective_dim,

    AVG(eigenvalue_1) AS mean_eigenvalue_1,
    AVG(total_variance) AS mean_total_variance,
    AVG(condition_number) AS mean_condition_number,

    -- Detect dimensional collapse
    CORR(I, effective_dim) AS effective_dim_trend,

    -- Count high ratio_2_1 (multimode indicators)
    SUM(CASE WHEN ratio_2_1 > 0.5 THEN 1 ELSE 0 END) AS multimode_count

FROM state_geometry
GROUP BY unit_id, engine;


-- Cross-engine comparison
CREATE OR REPLACE VIEW v_engine_comparison AS
SELECT
    unit_id,
    I,
    MAX(CASE WHEN engine = 'shape' THEN effective_dim END) AS effective_dim_shape,
    MAX(CASE WHEN engine = 'complexity' THEN effective_dim END) AS effective_dim_complexity,
    MAX(CASE WHEN engine = 'spectral' THEN effective_dim END) AS effective_dim_spectral,

    -- Engine disagreement
    MAX(CASE WHEN engine = 'shape' THEN effective_dim END) -
    MIN(CASE WHEN engine = 'complexity' THEN effective_dim END) AS dim_disagreement_shape_complexity

FROM state_geometry
GROUP BY unit_id, I;
"""


def run_sql_aggregations(
    state_geometry_path: str,
    output_dir: str = ".",
    verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Run SQL aggregations on state geometry.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dict of aggregation DataFrames
    """
    if verbose:
        print("\nRunning SQL aggregations...")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE state_geometry AS SELECT * FROM read_parquet('{state_geometry_path}')")

    # Run aggregation views
    for statement in STATE_GEOMETRY_SQL_AGGREGATIONS.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                con.execute(statement)
            except Exception as e:
                if verbose:
                    print(f"  Warning: {e}")

    results = {}

    # Export aggregations
    output_dir = Path(output_dir)

    try:
        by_unit = con.execute("SELECT * FROM v_state_geometry_by_unit").pl()
        by_unit.write_parquet(output_dir / "state_geometry_by_unit.parquet")
        results['by_unit'] = by_unit
        if verbose:
            print(f"  Saved: state_geometry_by_unit.parquet")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not export by_unit: {e}")

    try:
        engine_comp = con.execute("SELECT * FROM v_engine_comparison").pl()
        engine_comp.write_parquet(output_dir / "engine_comparison.parquet")
        results['engine_comparison'] = engine_comp
        if verbose:
            print(f"  Saved: engine_comparison.parquet")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not export engine_comparison: {e}")

    con.close()

    return results


# ============================================================
# CLI
# ============================================================

def main():
    """Safe CLI for state_geometry."""
    from prism.cli import SafeCLI

    cli = SafeCLI(
        description="""
State Geometry Engine - Eigenvalues and shape metrics

Computes per engine, per index:
  - Eigenvalues (via SVD)
  - effective_dim (from eigenvalues)
  - Total variance, condition number
  - Eigenvalue entropy

This is the SHAPE of signal distribution around each state.

Normalization methods (--norm):
  zscore  - Standard (x-mean)/std - sensitive to outliers (default)
  robust  - (x-median)/IQR - moderate robustness
  mad     - (x-median)/MAD - most robust (recommended for industrial)
  none    - Raw covariance (preserves variance dynamics)

Examples:
  python state_geometry.py -s signal_vector.parquet -t state_vector.parquet
  python state_geometry.py -s signal_vector.parquet -t state_vector.parquet --norm mad
  python state_geometry.py -s signal_vector.parquet -t state_vector.parquet -o output.parquet
"""
    )
    cli.add_input('signal_vector', '-s', help='Path to signal_vector.parquet')
    cli.add_input('state_vector', '-t', help='Path to state_vector.parquet')
    cli.add_output('output', default='state_geometry.parquet')
    cli.parser.add_argument(
        '--norm', '-n',
        choices=['zscore', 'robust', 'mad', 'none'],
        default='zscore',
        help='Normalization method before SVD (default: zscore)'
    )

    args = cli.parse()

    compute_state_geometry(
        args.signal_vector,
        args.state_vector,
        args.output,
        norm_method=args.norm,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
