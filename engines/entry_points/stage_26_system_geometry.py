"""
Stage 26: System Geometry Entry Point
=====================================

Eigendecomposition of cohort_vector matrix (cohorts x features) per I window.
This is Scale 2: same eigendecomp engine, but applied to cohorts instead of signals.

Inputs:
    - cohort_vector.parquet

Outputs:
    - system_geometry.parquet (one row per I: eigenvalues, effective_dim, etc.)
    - system_geometry_loadings.parquet (one row per cohort per I: pc1..3 loadings)
"""

import numpy as np
import polars as pl
from pathlib import Path

from engines.manifold.state.eigendecomp import compute as compute_eigendecomp


def run(
    cohort_vector_path: str,
    output_path: str = "system_geometry.parquet",
    max_eigenvalues: int = 5,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute system geometry via eigendecomposition of cohort feature matrix.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        output_path: Output path for system_geometry.parquet
        max_eigenvalues: Maximum eigenvalues to store
        verbose: Print progress

    Returns:
        System geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 26: SYSTEM GEOMETRY")
        print("Eigendecomposition of cohort feature matrix per I window")
        print("=" * 70)

    cv = pl.read_parquet(cohort_vector_path)

    if verbose:
        print(f"Loaded cohort_vector: {cv.shape}")

    if len(cv) == 0:
        if verbose:
            print("  Empty cohort_vector — skipping")
        pl.DataFrame().write_parquet(output_path)
        return pl.DataFrame()

    # Identify feature columns (everything except cohort, I)
    feature_cols = [c for c in cv.columns if c not in ['cohort', 'I']]

    if verbose:
        print(f"Feature columns: {len(feature_cols)}")

    i_values = sorted(cv['I'].unique().to_list())
    results = []
    loading_rows = []

    for I in i_values:
        window = cv.filter(pl.col('I') == I)
        cohorts = window['cohort'].to_list()
        n_cohorts = len(cohorts)

        # Build matrix: rows = cohorts, columns = features
        matrix = window.select(feature_cols).to_numpy().astype(float)

        # Filter NaN rows
        valid_mask = np.isfinite(matrix).all(axis=1)
        matrix_clean = matrix[valid_mask]
        valid_cohorts = [c for c, v in zip(cohorts, valid_mask) if v]

        N, D = matrix_clean.shape

        # Need more rows than features for meaningful eigendecomposition
        if N < 2 or N < D + 1:
            continue

        # Skip if any column is constant (zero variance)
        col_std = np.std(matrix_clean, axis=0)
        if np.any(col_std < 1e-12):
            # Drop constant columns for this window
            varying = col_std >= 1e-12
            if varying.sum() < 2:
                continue
            matrix_clean = matrix_clean[:, varying]

        # Compute centroid
        centroid = np.mean(matrix_clean, axis=0)

        # Eigendecomposition — same engine as stage_03
        eigen = compute_eigendecomp(matrix_clean, centroid=centroid, norm_method="zscore")

        if eigen['n_signals'] == 0:
            continue

        # Build result row
        row = {
            'I': I,
            'n_cohorts': n_cohorts,
            'n_features': len(feature_cols),
        }

        # Eigenvalues
        for j in range(min(max_eigenvalues, len(eigen['eigenvalues']))):
            row[f'eigenvalue_{j+1}'] = float(eigen['eigenvalues'][j])

        # Explained ratios
        for j in range(min(max_eigenvalues, len(eigen['explained_ratio']))):
            row[f'explained_{j+1}'] = float(eigen['explained_ratio'][j])

        row['effective_dim'] = eigen['effective_dim']
        row['total_variance'] = eigen['total_variance']
        row['condition_number'] = eigen['condition_number']
        row['ratio_2_1'] = eigen['ratio_2_1']
        row['ratio_3_1'] = eigen['ratio_3_1']
        row['eigenvalue_entropy_norm'] = eigen['eigenvalue_entropy_normalized']

        results.append(row)

        # Cohort loadings on PCs (signal_loadings in eigendecomp = cohort loadings here)
        signal_loadings = eigen.get('signal_loadings')
        if signal_loadings is not None:
            n_pcs = min(3, signal_loadings.shape[1])
            for idx, cohort_name in enumerate(valid_cohorts[:len(signal_loadings)]):
                lr = {
                    'I': I,
                    'cohort': cohort_name,
                    'pc1_loading': float(signal_loadings[idx, 0]),
                }
                if n_pcs > 1:
                    lr['pc2_loading'] = float(signal_loadings[idx, 1])
                if n_pcs > 2:
                    lr['pc3_loading'] = float(signal_loadings[idx, 2])
                loading_rows.append(lr)

    # Build output
    result = pl.DataFrame(results) if results else pl.DataFrame()

    result.write_parquet(output_path)

    # Write loadings sidecar
    if loading_rows:
        loadings_df = pl.DataFrame(loading_rows)
        loadings_path = str(Path(output_path).parent / 'system_geometry_loadings.parquet')
        loadings_df.write_parquet(loadings_path)
        if verbose:
            print(f"Loadings sidecar: {loadings_df.shape} -> {loadings_path}")

    if verbose:
        print(f"\nShape: {result.shape}")
        if len(result) > 0 and 'effective_dim' in result.columns:
            print(f"  effective_dim: mean={result['effective_dim'].mean():.2f}")
        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 26: System Geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Eigendecomposition of cohort feature matrix per I window.
Scale 2: same engine as state_geometry, applied to cohorts.

Example:
  python -m engines.entry_points.stage_26_system_geometry \\
      cohort_vector.parquet -o system_geometry.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-o', '--output', default='system_geometry.parquet',
                        help='Output path (default: system_geometry.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
