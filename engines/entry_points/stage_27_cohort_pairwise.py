"""
Stage 27: Cohort Pairwise Entry Point
=====================================

Pairwise metrics between cohorts at each I window.
Same distance/correlation/cosine primitives as signal_pairwise,
applied at the cohort level.

Inputs:
    - cohort_vector.parquet
    - system_geometry_loadings.parquet (optional, for pc1_coloading)

Output:
    - cohort_pairwise.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path
from itertools import combinations
from typing import Optional


def run(
    cohort_vector_path: str,
    output_path: str = "cohort_pairwise.parquet",
    system_geometry_loadings_path: Optional[str] = None,
    pc_coloading_threshold: float = 0.3,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute pairwise metrics between cohorts at each I window.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        output_path: Output path for cohort_pairwise.parquet
        system_geometry_loadings_path: Optional path to system_geometry_loadings.parquet
        pc_coloading_threshold: Threshold for flagging needs_granger
        verbose: Print progress

    Returns:
        Cohort pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 27: COHORT PAIRWISE")
        print("Pairwise distance, correlation, cosine between cohorts per I")
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

    # Load system_geometry_loadings if available
    loadings = None
    if system_geometry_loadings_path:
        loadings_file = Path(system_geometry_loadings_path)
        if loadings_file.exists():
            loadings = pl.read_parquet(str(loadings_file))
            if verbose:
                print(f"Loaded system_geometry_loadings: {loadings.shape}")

    i_values = sorted(cv['I'].unique().to_list())
    results = []

    for I in i_values:
        window = cv.filter(pl.col('I') == I)
        cohorts = window['cohort'].to_list()

        if len(cohorts) < 2:
            continue

        # Build feature matrix
        matrix = window.select(feature_cols).to_numpy().astype(float)

        # Build cohort-indexed lookup
        cohort_vectors = {}
        for idx, cohort in enumerate(cohorts):
            vec = matrix[idx]
            if np.isfinite(vec).all():
                cohort_vectors[cohort] = vec

        # Get loadings for this I window if available
        window_loadings = {}
        if loadings is not None and 'pc1_loading' in loadings.columns:
            wl = loadings.filter(pl.col('I') == I)
            for row in wl.iter_rows(named=True):
                window_loadings[row['cohort']] = row.get('pc1_loading')

        # Compute all C(N,2) pairs
        for cohort_a, cohort_b in combinations(sorted(cohort_vectors.keys()), 2):
            va = cohort_vectors[cohort_a]
            vb = cohort_vectors[cohort_b]

            # Euclidean distance
            distance = float(np.linalg.norm(va - vb))

            # Cosine similarity
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a > 1e-12 and norm_b > 1e-12:
                cosine_similarity = float(np.dot(va, vb) / (norm_a * norm_b))
            else:
                cosine_similarity = None

            # Pearson correlation
            if len(va) > 1:
                va_centered = va - np.mean(va)
                vb_centered = vb - np.mean(vb)
                denom = np.linalg.norm(va_centered) * np.linalg.norm(vb_centered)
                if denom > 1e-12:
                    correlation = float(np.dot(va_centered, vb_centered) / denom)
                else:
                    correlation = None
            else:
                correlation = None

            row = {
                'I': I,
                'cohort_a': cohort_a,
                'cohort_b': cohort_b,
                'distance': distance,
                'cosine_similarity': cosine_similarity,
                'correlation': correlation,
            }

            # PC1 co-loading and Granger gating
            if cohort_a in window_loadings and cohort_b in window_loadings:
                la = window_loadings[cohort_a]
                lb = window_loadings[cohort_b]
                if la is not None and lb is not None:
                    row['pc1_coloading'] = float(la * lb)
                    row['needs_granger'] = abs(la * lb) > pc_coloading_threshold
                else:
                    row['needs_granger'] = True  # Default to True if loadings unavailable
            else:
                row['needs_granger'] = True

            results.append(row)

    result = pl.DataFrame(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    result.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {result.shape}")
        if len(result) > 0:
            print(f"  Mean distance: {result['distance'].mean():.4f}")
            if 'needs_granger' in result.columns:
                n_granger = result.filter(pl.col('needs_granger') == True).height
                print(f"  Pairs needing Granger: {n_granger}/{len(result)}")
        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 27: Cohort Pairwise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise metrics between cohorts at each I window.

Example:
  python -m engines.entry_points.stage_27_cohort_pairwise \\
      cohort_vector.parquet -o cohort_pairwise.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-o', '--output', default='cohort_pairwise.parquet',
                        help='Output path (default: cohort_pairwise.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
