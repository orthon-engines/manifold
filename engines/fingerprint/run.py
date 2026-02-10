"""
Fingerprint operation -- Gaussian fingerprints + pairwise similarity.

Takes ANY set of vectors over time.
Produces one fingerprint per entity.
Produces one similarity score per entity pair.

This module does NOT know what scale it's operating at.
It delegates to the existing stage entry points which already handle
I/O, DuckDB SQL generation, and verbose output.

For direct (non-pipeline) use, call compute_fingerprint() and
compute_similarity() from the engines sub-modules.
"""

import polars as pl
from typing import List, Optional


def run(
    signal_vector_path: str,
    fingerprint_output: str = "gaussian_fingerprint.parquet",
    similarity_output: str = "gaussian_similarity.parquet",
    **kwargs,
):
    """Run fingerprint + similarity at signal scale. Delegates to stage_24.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        fingerprint_output: Output path for gaussian_fingerprint.parquet
        similarity_output:  Output path for gaussian_similarity.parquet
        **kwargs:           Forwarded (verbose, etc.)

    Returns:
        polars.DataFrame -- fingerprint result
    """
    from engines.entry_points.stage_24_gaussian_fingerprint import run as _run

    return _run(
        signal_vector_path,
        fingerprint_output,
        similarity_output,
        **kwargs,
    )


def run_system(
    cohort_vector_path: str,
    fingerprint_output: str = "cohort_fingerprint.parquet",
    similarity_output: str = "cohort_similarity.parquet",
    **kwargs,
):
    """Run fingerprint + similarity at cohort scale. Delegates to stage_32.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        fingerprint_output: Output path for cohort_fingerprint.parquet
        similarity_output:  Output path for cohort_similarity.parquet
        **kwargs:           Forwarded (verbose, etc.)

    Returns:
        polars.DataFrame -- fingerprint result
    """
    from engines.entry_points.stage_32_cohort_fingerprint import run as _run

    return _run(
        cohort_vector_path,
        fingerprint_output,
        similarity_output,
        **kwargs,
    )


def run_generic(
    input_path: str,
    fingerprint_output: str,
    similarity_output: str,
    feature_columns: Optional[List[str]] = None,
    entity_col: str = 'cohort',
    **kwargs,
):
    """Run fingerprint + similarity on any input data (no stage delegation).

    This is the scale-agnostic entry point for cases where the input
    is not a standard signal_vector or cohort_vector, or when the caller
    wants direct control over entity_col and feature_columns.

    Args:
        input_path:       Path to input parquet file.
        fingerprint_output: Output path for fingerprint parquet.
        similarity_output:  Output path for similarity parquet.
        feature_columns:  Which columns to fingerprint. If None, auto-detected
                          (all numeric columns except entity_col and I).
        entity_col:       Column identifying entities (default: 'cohort').
        **kwargs:         Reserved for future use.

    Returns:
        Tuple of (fingerprint_df, similarity_df)
    """
    from engines.fingerprint.engines.gaussian import compute as compute_fingerprint
    from engines.fingerprint.engines.similarity import compute as compute_similarity

    data = pl.read_parquet(input_path)

    if feature_columns is None:
        meta_cols = {'I', 'cohort', 'signal_id', 'unit_id', entity_col}
        feature_columns = [
            c for c in data.columns
            if c not in meta_cols
            and data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    if len(feature_columns) == 0:
        pl.DataFrame().write_parquet(fingerprint_output)
        pl.DataFrame().write_parquet(similarity_output)
        return pl.DataFrame(), pl.DataFrame()

    fp = compute_fingerprint(data, feature_columns, entity_col=entity_col)
    fp.write_parquet(fingerprint_output)

    sim = compute_similarity(fp, feature_columns, entity_col=entity_col)
    sim.write_parquet(similarity_output)

    return fp, sim
