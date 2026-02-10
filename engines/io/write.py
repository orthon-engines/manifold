"""
Parquet writing with schema enforcement and empty-file handling.

Centralizes all output I/O. Engines produce DataFrames, this module persists them.
"""

import polars as pl
from pathlib import Path
from typing import Optional


def write_parquet(df: pl.DataFrame, path: str, verbose: bool = False) -> None:
    """Write a DataFrame to parquet.

    Args:
        df: DataFrame to write
        path: Output path
        verbose: Print confirmation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))
    if verbose:
        print(f"  Saved: {path}")
        print(f"  Shape: {df.shape}")


def write_empty(path: str, verbose: bool = False) -> None:
    """Write an empty parquet file (for stages that produce no output).

    This ensures downstream stages can detect "ran but empty" vs "never ran".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame().write_parquet(str(path))
    if verbose:
        print(f"  Saved (empty): {path}")


def safe_write(df: Optional[pl.DataFrame], path: str, verbose: bool = False) -> None:
    """Write DataFrame if non-empty, otherwise write empty marker."""
    if df is not None and len(df) > 0:
        write_parquet(df, path, verbose=verbose)
    else:
        write_empty(path, verbose=verbose)
