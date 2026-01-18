"""
PRISM Cohort Aggregator Engine
==============================

Aggregates indicator-level geometry metrics to cohort level.

This engine transforms indicator-level barycenters, dispersions, and alignments
into cohort-level summaries for cross-cohort analysis.

Input: geometry.indicators data grouped by cohort
Output: Cohort-level aggregate metrics

Key Aggregations:
    - Mean/median dispersion per cohort
    - Mean/median alignment per cohort
    - Cohort barycenter (centroid of indicator barycenters)
    - Internal cohort coherence
    - Leading/lagging indicators within cohort

Usage:
    from prism.engines.cohort_aggregator import CohortAggregatorEngine

    engine = CohortAggregatorEngine()
    result = engine.run(indicators_df, cohort_members)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, euclidean


@dataclass
class CohortAggregateResult:
    """Result from cohort aggregation."""
    cohort_id: str
    n_indicators: int
    barycenter: np.ndarray
    dispersion_mean: float
    dispersion_std: float
    alignment_mean: float
    alignment_std: float
    internal_coherence: float
    spread: float  # How spread out are indicators?
    metrics: Dict[str, float]


class CohortAggregatorEngine:
    """
    Aggregate indicator-level metrics to cohort level.

    Takes indicator barycenters and metrics from geometry.indicators
    and produces cohort-level summaries for cross-cohort analysis.
    """

    def __init__(self):
        pass

    def run(
        self,
        indicator_data: pd.DataFrame,
        cohort_indicators: List[str]
    ) -> CohortAggregateResult:
        """
        Aggregate metrics for a single cohort.

        Args:
            indicator_data: DataFrame with columns:
                - indicator_id
                - barycenter (as array)
                - timescale_dispersion
                - timescale_alignment
            cohort_indicators: List of indicator IDs in this cohort

        Returns:
            CohortAggregateResult
        """
        # Filter to cohort indicators
        cohort_data = indicator_data[
            indicator_data['indicator_id'].isin(cohort_indicators)
        ]

        if cohort_data.empty or len(cohort_data) < 2:
            return CohortAggregateResult(
                cohort_id='',
                n_indicators=len(cohort_data),
                barycenter=np.array([]),
                dispersion_mean=0.0,
                dispersion_std=0.0,
                alignment_mean=0.0,
                alignment_std=0.0,
                internal_coherence=0.0,
                spread=0.0,
                metrics={}
            )

        n_indicators = len(cohort_data)

        # Aggregate dispersions
        dispersions = cohort_data['timescale_dispersion'].values
        dispersion_mean = float(np.mean(dispersions))
        dispersion_std = float(np.std(dispersions))

        # Aggregate alignments
        alignments = cohort_data['timescale_alignment'].values
        alignment_mean = float(np.mean(alignments))
        alignment_std = float(np.std(alignments))

        # Compute cohort barycenter (centroid of indicator barycenters)
        barycenters = []
        for _, row in cohort_data.iterrows():
            bc = row['barycenter']
            if bc is not None and len(bc) > 0:
                barycenters.append(np.array(bc))

        if len(barycenters) >= 2:
            cohort_barycenter = np.mean(barycenters, axis=0)

            # Internal coherence: inverse of average pairwise distance
            pairwise_dists = pdist(np.array(barycenters))
            internal_coherence = 1.0 / (1.0 + np.mean(pairwise_dists))

            # Spread: max distance from cohort centroid
            distances_to_center = [euclidean(bc, cohort_barycenter) for bc in barycenters]
            spread = float(np.max(distances_to_center))
        else:
            cohort_barycenter = barycenters[0] if barycenters else np.array([])
            internal_coherence = 1.0
            spread = 0.0

        # Additional metrics
        metrics = {
            'dispersion_min': float(np.min(dispersions)),
            'dispersion_max': float(np.max(dispersions)),
            'dispersion_range': float(np.max(dispersions) - np.min(dispersions)),
            'alignment_min': float(np.min(alignments)),
            'alignment_max': float(np.max(alignments)),
            'alignment_range': float(np.max(alignments) - np.min(alignments)),
            'homogeneity': 1.0 / (1.0 + dispersion_std + alignment_std),
        }

        return CohortAggregateResult(
            cohort_id='',
            n_indicators=n_indicators,
            barycenter=cohort_barycenter,
            dispersion_mean=dispersion_mean,
            dispersion_std=dispersion_std,
            alignment_mean=alignment_mean,
            alignment_std=alignment_std,
            internal_coherence=internal_coherence,
            spread=spread,
            metrics=metrics
        )

    def aggregate_all_cohorts(
        self,
        indicator_data: pd.DataFrame,
        cohort_membership: Dict[str, List[str]]
    ) -> Dict[str, CohortAggregateResult]:
        """
        Aggregate metrics for all cohorts.

        Args:
            indicator_data: Full indicator data
            cohort_membership: Dict mapping cohort_id -> list of indicator_ids

        Returns:
            Dict mapping cohort_id -> CohortAggregateResult
        """
        results = {}

        for cohort_id, indicators in cohort_membership.items():
            result = self.run(indicator_data, indicators)
            result.cohort_id = cohort_id
            results[cohort_id] = result

        return results

    def compute_cross_cohort_distances(
        self,
        cohort_results: Dict[str, CohortAggregateResult]
    ) -> pd.DataFrame:
        """
        Compute pairwise distances between cohort barycenters.

        Returns DataFrame with cohort_a, cohort_b, distance
        """
        cohorts = list(cohort_results.keys())
        records = []

        for i, cohort_a in enumerate(cohorts):
            bc_a = cohort_results[cohort_a].barycenter
            if len(bc_a) == 0:
                continue

            for cohort_b in cohorts[i+1:]:
                bc_b = cohort_results[cohort_b].barycenter
                if len(bc_b) == 0:
                    continue

                distance = euclidean(bc_a, bc_b)
                records.append({
                    'cohort_a': cohort_a,
                    'cohort_b': cohort_b,
                    'distance': distance
                })

        return pd.DataFrame(records)


def aggregate_cohort(
    indicator_data: pd.DataFrame,
    cohort_indicators: List[str]
) -> Dict[str, Any]:
    """
    Functional interface for cohort aggregation.

    Returns dict with cohort aggregate metrics.
    """
    engine = CohortAggregatorEngine()
    result = engine.run(indicator_data, cohort_indicators)

    return {
        'n_indicators': result.n_indicators,
        'barycenter': result.barycenter.tolist() if len(result.barycenter) > 0 else [],
        'dispersion_mean': result.dispersion_mean,
        'dispersion_std': result.dispersion_std,
        'alignment_mean': result.alignment_mean,
        'alignment_std': result.alignment_std,
        'internal_coherence': result.internal_coherence,
        'spread': result.spread,
        **result.metrics
    }
