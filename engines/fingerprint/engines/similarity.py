"""Similarity engine -- Bhattacharyya distance between entity fingerprints.

Computes pairwise similarity between Gaussian fingerprints using the
Bhattacharyya distance. For two univariate Gaussians N(mu_a, sigma_a^2) and
N(mu_b, sigma_b^2), the Bhattacharyya distance is:

    D_B = 0.25 * (mu_a - mu_b)^2 / (var_a + var_b)
        + 0.5 * ln((var_a + var_b) / (2 * sigma_a * sigma_b))

The per-feature distances are summed, and the total is exponentiated to get
a similarity score in (0, 1].

Scale-agnostic: works on signal, cohort, or any entity fingerprints.
"""

import numpy as np
import polars as pl
from typing import List
from itertools import combinations


def compute(
    fingerprint_df: pl.DataFrame,
    feature_columns: List[str],
    entity_col: str = 'cohort',
    **params,
) -> pl.DataFrame:
    """Compute pairwise Bhattacharyya similarity between entity fingerprints.

    Args:
        fingerprint_df:  Output of gaussian.compute() -- one row per entity
                         with mean_* and std_* columns.
        feature_columns: Original feature names (used to locate mean_/std_ cols).
        entity_col:      Column identifying entities.
        **params:        Reserved for future use.

    Returns:
        DataFrame with one row per entity pair containing:
            entity_a, entity_b, bhattacharyya_distance, n_features,
            normalized_distance, similarity, volatility_diff
    """
    entities = fingerprint_df[entity_col].unique().sort().to_list()

    if len(entities) < 2:
        return pl.DataFrame()

    rows = []
    for a, b in combinations(entities, 2):
        row_a = fingerprint_df.filter(pl.col(entity_col) == a)
        row_b = fingerprint_df.filter(pl.col(entity_col) == b)

        distances = []
        for feat in feature_columns:
            mean_col = f'mean_{feat}'
            std_col = f'std_{feat}'

            if mean_col not in fingerprint_df.columns or std_col not in fingerprint_df.columns:
                continue

            mu_a = row_a[mean_col][0]
            mu_b = row_b[mean_col][0]
            s_a = row_a[std_col][0]
            s_b = row_b[std_col][0]

            if (
                np.isfinite(mu_a) and np.isfinite(mu_b)
                and np.isfinite(s_a) and np.isfinite(s_b)
                and s_a > 1e-10 and s_b > 1e-10
            ):
                var_a, var_b = s_a ** 2, s_b ** 2
                sum_var = var_a + var_b
                d = (
                    0.25 * (mu_a - mu_b) ** 2 / sum_var
                    + 0.5 * np.log(sum_var / (2.0 * s_a * s_b))
                )
                distances.append(max(0.0, d))

        n_features = len(distances)
        bhatt = float(np.sum(distances)) if distances else float('nan')
        norm_dist = float(bhatt / n_features) if n_features > 0 else float('nan')

        # Volatility difference
        vol_a = row_a['volatility'][0] if 'volatility' in row_a.columns else float('nan')
        vol_b = row_b['volatility'][0] if 'volatility' in row_b.columns else float('nan')
        vol_diff = float(abs(vol_a - vol_b)) if np.isfinite(vol_a) and np.isfinite(vol_b) else float('nan')

        rows.append({
            'entity_a': a,
            'entity_b': b,
            'bhattacharyya_distance': bhatt,
            'n_features': n_features,
            'normalized_distance': norm_dist,
            'similarity': float(np.exp(-bhatt)) if np.isfinite(bhatt) else float('nan'),
            'volatility_diff': vol_diff,
        })

    return pl.DataFrame(rows) if rows else pl.DataFrame()
