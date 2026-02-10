"""
Input validation — guards, min-sample checks, NaN/inf handling.

Centralizes defensive checks so engines can assume clean input.
"""

import numpy as np
import polars as pl
from typing import Optional


def validate_min_samples(y: np.ndarray, min_samples: int, engine_name: str = "engine") -> None:
    """Raise ValueError if array has fewer than min_samples valid values.

    Args:
        y: input array
        min_samples: minimum required
        engine_name: for error message

    Raises:
        ValueError: if insufficient samples
    """
    n_valid = np.sum(np.isfinite(y))
    if n_valid < min_samples:
        raise ValueError(
            f"{engine_name} requires {min_samples} samples, got {n_valid} valid"
        )


def clean_infinities(df: pl.DataFrame) -> pl.DataFrame:
    """Replace infinities with null in all float columns.

    Args:
        df: input DataFrame

    Returns:
        DataFrame with infinities replaced by null
    """
    float_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32]]
    for col in float_cols:
        df = df.with_columns(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df


def clean_array(y: np.ndarray) -> np.ndarray:
    """Remove NaN/inf from 1D array, preserving order.

    Args:
        y: input array (may contain NaN/inf)

    Returns:
        Clean array with only finite values
    """
    y = np.asarray(y, dtype=float).ravel()
    return y[np.isfinite(y)]


def validate_parquet_nonempty(path: str, name: str = "file") -> pl.DataFrame:
    """Read parquet and raise if empty.

    Args:
        path: path to parquet file
        name: for error message

    Returns:
        Non-empty DataFrame

    Raises:
        ValueError: if file is empty
        FileNotFoundError: if file doesn't exist
    """
    from engines.io.read import read_parquet
    df = read_parquet(path)
    if len(df) == 0:
        raise ValueError(f"{name} is empty: {path}")
    return df
