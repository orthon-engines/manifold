"""
Parquet reading and schema validation.

Centralizes all file I/O so engines never touch the filesystem directly.
"""

import polars as pl
from pathlib import Path
from typing import Optional, List, Set


def read_parquet(path: str, required_columns: Optional[Set[str]] = None) -> pl.DataFrame:
    """Read a parquet file with optional schema validation.

    Args:
        path: Path to parquet file
        required_columns: If provided, raises ValueError if any are missing

    Returns:
        Polars DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pl.read_parquet(str(path))

    if required_columns:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {sorted(missing)}. "
                f"Available: {sorted(df.columns)}"
            )

    return df


def validate_schema(df: pl.DataFrame, required: Set[str], name: str = "DataFrame") -> None:
    """Validate that a DataFrame has required columns.

    Raises ValueError with clear message if columns are missing.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def read_if_exists(path: str) -> Optional[pl.DataFrame]:
    """Read parquet if it exists, return None otherwise."""
    p = Path(path)
    if p.exists():
        df = pl.read_parquet(str(p))
        if len(df) > 0:
            return df
    return None
