"""Utility helpers for Manifold."""


def safe_fmt(value, fmt=".2f", default="N/A"):
    """Format a value that might be None from polars aggregation."""
    if value is None:
        return default
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return default


def safe_float(value, default=float('nan')):
    """Convert polars aggregation result to float, handling None."""
    if value is None:
        return default
    return float(value)
