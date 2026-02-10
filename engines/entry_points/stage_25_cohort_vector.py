"""
Stage 25: Cohort Vector Entry Point
====================================

Extracts per-cohort features from state_geometry by pivoting engine
rows to wide format. Each cohort at each I window becomes a single
feature vector with per-engine prefixed columns.

This is the foundation for Scale 2 (system_geometry) — same engines,
different input.

Inputs:
    - state_geometry.parquet

Output:
    - cohort_vector.parquet (one row per cohort per I, ~40-50 columns)
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional


# Columns to pivot from state_geometry per engine
PIVOT_COLS = [
    'effective_dim', 'eigenvalue_1', 'total_variance', 'condition_number',
    'ratio_2_1', 'ratio_3_1', 'eigenvalue_entropy_norm',
    'explained_1', 'explained_2', 'explained_3', 'n_signals',
]


def run(
    state_geometry_path: str,
    output_path: str = "cohort_vector.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Build cohort feature vectors by pivoting state_geometry engine rows to wide.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        output_path: Output path for cohort_vector.parquet
        verbose: Print progress

    Returns:
        Cohort vector DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 25: COHORT VECTOR")
        print("Pivot state_geometry engines to wide cohort feature vectors")
        print("=" * 70)

    sg = pl.read_parquet(state_geometry_path)

    if verbose:
        print(f"Loaded state_geometry: {sg.shape}")

    if len(sg) == 0:
        if verbose:
            print("  Empty state_geometry — skipping")
        pl.DataFrame().write_parquet(output_path)
        return pl.DataFrame()

    # Require cohort column
    if 'cohort' not in sg.columns:
        if verbose:
            print("  No cohort column — skipping (Scale 2 requires cohorts)")
        pl.DataFrame().write_parquet(output_path)
        return pl.DataFrame()

    engines = sorted(sg['engine'].unique().to_list())
    if verbose:
        print(f"Engines: {engines}")

    # Determine which pivot columns actually exist
    available_pivot = [c for c in PIVOT_COLS if c in sg.columns]

    if verbose:
        print(f"Pivot columns: {len(available_pivot)}")

    # Pivot each engine into prefixed columns
    pivoted_frames = []

    for engine in engines:
        engine_data = sg.filter(pl.col('engine') == engine).select(
            ['cohort', 'I'] + available_pivot
        )

        # Rename columns with engine prefix
        rename_map = {c: f'{engine}_{c}' for c in available_pivot}
        engine_data = engine_data.rename(rename_map)
        pivoted_frames.append(engine_data)

    if not pivoted_frames:
        if verbose:
            print("  No engine data to pivot")
        pl.DataFrame().write_parquet(output_path)
        return pl.DataFrame()

    # Join all engine frames on (cohort, I)
    result = pivoted_frames[0]
    for frame in pivoted_frames[1:]:
        result = result.join(frame, on=['cohort', 'I'], how='outer_coalesce')

    # Sort for deterministic output
    result = result.sort(['cohort', 'I'])

    # Add derived features per engine
    for engine in engines:
        eff_dim_col = f'{engine}_effective_dim'
        explained_col = f'{engine}_explained_1'

        # Effective dim rate: finite difference over I (thermodynamic temperature)
        if eff_dim_col in result.columns:
            result = result.with_columns(
                pl.col(eff_dim_col)
                .diff()
                .over('cohort')
                .alias(f'{engine}_eff_dim_rate')
            )

        # Variance concentration: explained_1 (energy in dominant mode)
        if explained_col in result.columns:
            result = result.with_columns(
                pl.col(explained_col).alias(f'{engine}_variance_concentration')
            )

    if verbose:
        print(f"\nShape: {result.shape}")
        print(f"Columns: {len(result.columns)}")
        feature_cols = [c for c in result.columns if c not in ['cohort', 'I']]
        print(f"Feature columns: {len(feature_cols)}")

    result.write_parquet(output_path)

    if verbose:
        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 25: Cohort Vector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pivots state_geometry engine rows into wide cohort feature vectors.
Foundation for Scale 2 system geometry.

Example:
  python -m engines.entry_points.stage_25_cohort_vector \\
      state_geometry.parquet -o cohort_vector.parquet
"""
    )
    parser.add_argument('state_geometry', help='Path to state_geometry.parquet')
    parser.add_argument('-o', '--output', default='cohort_vector.parquet',
                        help='Output path (default: cohort_vector.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_geometry,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
