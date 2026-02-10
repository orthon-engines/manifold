"""
Stage 29: Cohort Topology Entry Point
=====================================

Graph topology from cohort correlation matrix per I window.
Same topology computation as stage_11, applied at the cohort level.

Inputs:
    - cohort_vector.parquet

Output:
    - cohort_topology.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path


def _compute_basic_topology(matrix: np.ndarray, threshold: float = None) -> dict:
    """
    Compute basic topological features from a cohort feature matrix.

    Same logic as stage_11_topology.compute_basic_topology but
    inlined to avoid import coupling.

    Args:
        matrix: N_cohorts x D_features matrix
        threshold: Correlation threshold for adjacency. If None, uses
                   90th percentile of abs(correlation).

    Returns:
        Dict with topological metrics
    """
    N, D = matrix.shape

    if N < 3 or D < 2:
        return {'topology_computed': False}

    # Correlation-based adjacency (cohorts x cohorts from feature vectors)
    matrix_filled = np.where(np.isfinite(matrix), matrix, 0.0)
    corr = np.corrcoef(matrix_filled)
    np.fill_diagonal(corr, 0)

    # Adaptive threshold: 90th percentile of abs(correlation)
    if threshold is None:
        upper_tri = np.abs(corr[np.triu_indices_from(corr, k=1)])
        finite_vals = upper_tri[np.isfinite(upper_tri)]
        if len(finite_vals) > 0:
            threshold = float(np.percentile(finite_vals, 90))
        else:
            threshold = 0.5

    corr_clean = np.where(np.isfinite(corr), corr, 0.0)
    adjacency = np.abs(corr_clean) > threshold

    n_edges = int(np.sum(adjacency) // 2)
    density = n_edges / (N * (N - 1) / 2) if N > 1 else 0.0

    degrees = np.sum(adjacency, axis=0)
    mean_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))

    return {
        'topology_computed': True,
        'n_cohorts': N,
        'n_edges': n_edges,
        'density': float(density),
        'mean_degree': mean_degree,
        'max_degree': max_degree,
        'threshold': float(threshold),
    }


def run(
    cohort_vector_path: str,
    output_path: str = "cohort_topology.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute cohort topology per I window.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        output_path: Output path for cohort_topology.parquet
        verbose: Print progress

    Returns:
        Cohort topology DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 29: COHORT TOPOLOGY")
        print("Graph topology from cohort correlation matrix per I")
        print("=" * 70)

    cv = pl.read_parquet(cohort_vector_path)

    if verbose:
        print(f"Loaded cohort_vector: {cv.shape}")

    if len(cv) == 0:
        if verbose:
            print("  Empty cohort_vector — skipping")
        pl.DataFrame().write_parquet(output_path)
        return pl.DataFrame()

    feature_cols = [c for c in cv.columns if c not in ['cohort', 'I']]

    # Compute per-cohort threshold across all I windows (like stage_11 does per-cohort)
    # For system topology, use a single global threshold
    all_matrix = cv.select(feature_cols).to_numpy().astype(float)
    all_matrix_filled = np.where(np.isfinite(all_matrix), all_matrix, 0.0)
    # rows = cohort-I observations; correlate across cohort feature vectors
    # For global threshold, compute correlation of all rows
    if all_matrix_filled.shape[0] > 2:
        corr_global = np.corrcoef(all_matrix_filled)
        np.fill_diagonal(corr_global, 0)
        upper_tri = np.abs(corr_global[np.triu_indices_from(corr_global, k=1)])
        finite_vals = upper_tri[np.isfinite(upper_tri)]
        global_threshold = float(np.percentile(finite_vals, 90)) if len(finite_vals) > 0 else 0.5
    else:
        global_threshold = 0.5

    i_values = sorted(cv['I'].unique().to_list())
    results = []

    for I in i_values:
        window = cv.filter(pl.col('I') == I)

        if len(window) < 3:
            continue

        matrix = window.select(feature_cols).to_numpy().astype(float)
        topo = _compute_basic_topology(matrix, threshold=global_threshold)

        if topo.get('topology_computed', False):
            topo['I'] = I
            results.append(topo)

    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0 and 'topology_computed' in result.columns:
        result = result.drop('topology_computed')

    result.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {result.shape}")
        if len(result) > 0 and 'n_edges' in result.columns:
            print(f"  Mean edges: {result['n_edges'].mean():.1f}")
            print(f"  Mean density: {result['density'].mean():.4f}")
        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 29: Cohort Topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Graph topology from cohort correlation matrix per I window.

Example:
  python -m engines.entry_points.stage_29_cohort_topology \\
      cohort_vector.parquet -o cohort_topology.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-o', '--output', default='cohort_topology.parquet',
                        help='Output path (default: cohort_topology.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
