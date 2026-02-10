"""Gaussian fingerprint -- per-entity mean, std, correlation of features over time.

Computes a probabilistic summary of each entity's feature distribution across
time windows. Scale-agnostic: works on signals, cohorts, or any entity type.

For each entity and each feature column, computes:
    - mean_{feature}: average value across windows
    - std_{feature}:  standard deviation across windows (sample std, 0 if n<2)

Aggregate:
    - n_windows:   number of time windows for the entity
    - volatility:  average of all std columns (overall behavioral variability)
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any


def compute(
    entity_data: pl.DataFrame,
    feature_columns: List[str],
    entity_col: str = 'cohort',
    **params,
) -> pl.DataFrame:
    """Compute Gaussian fingerprint for each entity. Scale-agnostic.

    Args:
        entity_data:     DataFrame with entity_col, I, and feature columns.
        feature_columns: Which columns to fingerprint.
        entity_col:      Column identifying entities (signal_id or cohort).
        **params:        Reserved for future use.

    Returns:
        DataFrame with one row per entity containing:
            entity_col, n_windows, mean_*, std_*, volatility
    """
    if entity_col not in entity_data.columns:
        raise ValueError(
            f"Entity column '{entity_col}' not found in data. "
            f"Available columns: {entity_data.columns}"
        )

    present_features = [f for f in feature_columns if f in entity_data.columns]
    if len(present_features) == 0:
        raise ValueError(
            f"None of the requested feature columns found in data. "
            f"Requested: {feature_columns}, Available: {entity_data.columns}"
        )

    rows = []
    for entity in entity_data[entity_col].unique().sort().to_list():
        subset = entity_data.filter(pl.col(entity_col) == entity)
        row: Dict[str, Any] = {entity_col: entity, 'n_windows': len(subset)}

        stds = []
        for feat in present_features:
            vals = subset[feat].drop_nulls().to_numpy()
            finite_vals = vals[np.isfinite(vals)] if len(vals) > 0 else vals

            if len(finite_vals) > 0:
                row[f'mean_{feat}'] = float(np.mean(finite_vals))
                # Use sample std (ddof=1) when n>1, else 0.0 (matches SQL STDDEV_SAMP)
                if len(finite_vals) > 1:
                    row[f'std_{feat}'] = float(np.std(finite_vals, ddof=1))
                else:
                    row[f'std_{feat}'] = 0.0
                stds.append(row[f'std_{feat}'])
            else:
                row[f'mean_{feat}'] = float('nan')
                row[f'std_{feat}'] = float('nan')

        row['volatility'] = float(np.mean(stds)) if stds else float('nan')
        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()
