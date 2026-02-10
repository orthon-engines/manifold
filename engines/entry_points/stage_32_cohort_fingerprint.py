"""
Stage 32: Cohort Fingerprint Entry Point
========================================

Gaussian fingerprints per cohort + pairwise Bhattacharyya similarity.
Same SQL template pattern as stage_24_gaussian_fingerprint, applied to
cohort_vector features.

Inputs:
    - cohort_vector.parquet

Outputs:
    - cohort_fingerprint.parquet (per cohort: n_windows, mean/std per feature, volatility)
    - cohort_similarity.parquet (per pair: bhattacharyya_distance, similarity)
"""

import polars as pl
import duckdb
from pathlib import Path
from typing import List


def _build_fingerprint_sql(features: List[str]) -> str:
    """Generate fingerprint SQL for cohort features."""
    mean_lines = []
    std_lines = []
    for f in features:
        finite_filter = f"WHERE isfinite(\"{f}\")"
        mean_lines.append(f"        AVG(\"{f}\") FILTER ({finite_filter}) AS \"mean_{f}\"")
        std_lines.append(
            f"        CASE WHEN COUNT(\"{f}\") FILTER ({finite_filter}) > 1 "
            f"THEN STDDEV_SAMP(\"{f}\") FILTER ({finite_filter}) ELSE 0.0 END AS \"std_{f}\""
        )

    vol_parts = [f"COALESCE(\"std_{f}\", 0)" for f in features]
    vol_expr = " +\n            ".join(vol_parts)
    vol_line = f"    ({vol_expr}) / {len(features)}.0 AS volatility"

    select_cols = ["    cohort", "    n_windows"]
    select_cols.extend([f"    \"mean_{f}\"" for f in features])
    select_cols.extend([f"    \"std_{f}\"" for f in features])
    select_cols.append(vol_line)

    all_agg = ",\n".join(mean_lines + std_lines)

    sql = f"""
WITH cohort_stats AS (
    SELECT
        cohort,
        COUNT(*) AS n_windows,
{all_agg}
    FROM cohort_vector
    GROUP BY cohort
)
SELECT
{",\n".join(select_cols)}
FROM cohort_stats
ORDER BY cohort;
"""
    return sql


def _build_similarity_sql(features: List[str]) -> str:
    """Generate Bhattacharyya similarity SQL for cohort pairs."""
    db_lines = []
    for f in features:
        db_lines.append(f"""        CASE WHEN a."std_{f}" > 1e-10 AND b."std_{f}" > 1e-10 THEN
            0.25 * POWER(a."mean_{f}" - b."mean_{f}", 2)
                / (POWER(a."std_{f}", 2) + POWER(b."std_{f}", 2))
            + 0.5 * LN((POWER(a."std_{f}", 2) + POWER(b."std_{f}", 2))
                / (2.0 * a."std_{f}" * b."std_{f}"))
        ELSE NULL END AS "db_{f}" """)

    db_block = ",\n\n".join(db_lines)

    coalesce_parts = [f"COALESCE(\"db_{f}\", 0)" for f in features]
    sum_expr = " +\n        ".join(coalesce_parts)

    count_parts = [f"CASE WHEN \"db_{f}\" IS NOT NULL THEN 1 ELSE 0 END" for f in features]
    count_expr = " +\n        ".join(count_parts)

    sql = f"""
WITH feature_distance AS (
    SELECT
        a.cohort AS cohort_a,
        b.cohort AS cohort_b,

{db_block},

        ABS(a.volatility - b.volatility) AS volatility_diff

    FROM cohort_fingerprint a
    JOIN cohort_fingerprint b
        ON a.cohort < b.cohort
)
SELECT
    cohort_a,
    cohort_b,

    ({sum_expr}) AS bhattacharyya_distance,

    ({count_expr}) AS n_features,

    CASE WHEN ({count_expr}) > 0 THEN
        ({sum_expr}) / NULLIF(({count_expr}), 0)
    ELSE NULL END AS normalized_distance,

    EXP(-({sum_expr})) AS similarity,

    volatility_diff

FROM feature_distance
ORDER BY bhattacharyya_distance;
"""
    return sql


def run(
    cohort_vector_path: str,
    fingerprint_output_path: str = "cohort_fingerprint.parquet",
    similarity_output_path: str = "cohort_similarity.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute Gaussian fingerprints and pairwise Bhattacharyya similarity for cohorts.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        fingerprint_output_path: Output path for cohort_fingerprint.parquet
        similarity_output_path: Output path for cohort_similarity.parquet
        verbose: Print progress

    Returns:
        Fingerprint DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 32: COHORT FINGERPRINT")
        print("Gaussian fingerprints + pairwise Bhattacharyya similarity")
        print("=" * 70)

    con = duckdb.connect()
    con.execute(f"CREATE TABLE cohort_vector AS SELECT * FROM read_parquet('{cohort_vector_path}')")

    # Check row count
    n_rows = con.execute("SELECT COUNT(*) FROM cohort_vector").fetchone()[0]
    if n_rows == 0:
        if verbose:
            print("  Empty cohort_vector — skipping")
        con.close()
        pl.DataFrame().write_parquet(fingerprint_output_path)
        pl.DataFrame().write_parquet(similarity_output_path)
        return pl.DataFrame()

    n_cohorts = con.execute("SELECT COUNT(DISTINCT cohort) FROM cohort_vector").fetchone()[0]

    if verbose:
        print(f"Loaded: {n_rows:,} rows, {n_cohorts} cohorts")

    # Discover feature columns (exclude cohort and I)
    all_cols = [row[0] for row in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'cohort_vector'"
    ).fetchall()]
    feature_cols = [c for c in all_cols if c not in ['cohort', 'I']]

    if verbose:
        print(f"Feature columns: {len(feature_cols)}")

    if len(feature_cols) == 0:
        if verbose:
            print("  No feature columns — skipping")
        con.close()
        pl.DataFrame().write_parquet(fingerprint_output_path)
        pl.DataFrame().write_parquet(similarity_output_path)
        return pl.DataFrame()

    # Step 1: Compute fingerprints
    fingerprint_sql = _build_fingerprint_sql(feature_cols)

    if verbose:
        print("\nComputing Gaussian fingerprints...")

    fingerprint = con.execute(fingerprint_sql).pl()
    fingerprint.write_parquet(fingerprint_output_path)

    if verbose:
        print(f"  Fingerprints: {fingerprint.shape}")

    # Step 2: Compute similarity
    if n_cohorts >= 2:
        if verbose:
            print("Computing pairwise similarity...")

        con.execute(f"CREATE TABLE cohort_fingerprint AS SELECT * FROM read_parquet('{fingerprint_output_path}')")
        similarity_sql = _build_similarity_sql(feature_cols)
        similarity = con.execute(similarity_sql).pl()
        similarity.write_parquet(similarity_output_path)

        if verbose:
            print(f"  Similarity pairs: {similarity.shape}")
    else:
        if verbose:
            print("  < 2 cohorts — skipping similarity")
        pl.DataFrame().write_parquet(similarity_output_path)

    con.close()

    if verbose:
        print()
        print("=" * 50)
        print(f"  {Path(fingerprint_output_path).absolute()}")
        print(f"  {Path(similarity_output_path).absolute()}")
        print("=" * 50)

    return fingerprint


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 32: Cohort Fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Builds Gaussian fingerprints from cohort_vector.parquet,
then computes pairwise Bhattacharyya similarity between cohorts.

Example:
  python -m engines.entry_points.stage_32_cohort_fingerprint \\
      cohort_vector.parquet \\
      -o cohort_fingerprint.parquet \\
      --similarity cohort_similarity.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-o', '--output', default='cohort_fingerprint.parquet',
                        help='Output path for fingerprints (default: cohort_fingerprint.parquet)')
    parser.add_argument('--similarity', default='cohort_similarity.parquet',
                        help='Output path for similarity (default: cohort_similarity.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.output,
        args.similarity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
