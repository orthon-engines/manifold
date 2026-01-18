"""
PRISM State Layer - Transition Detector
=======================================

Detects regime boundaries using Laplace field divergence spikes.
A transition is a statistically significant change in system dynamics.

Key insight: Transitions manifest as divergence singularities in the field.
When a regime change occurs, the total system divergence spikes as indicators
collectively respond to the structural shift.

Usage:
    from prism.state import detect_transitions, find_leading_indicators

    system_div, transitions = detect_transitions(field_df, zscore_threshold=3.0)
    leaders = find_leading_indicators(field_df, [t.window_end for t in transitions])
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Transition:
    """
    Detected regime transition.

    Attributes:
        window_start: Start of transition window
        window_end: End of transition window (detection point)
        divergence_zscore: Z-score of total divergence (significance measure)
        total_divergence: Sum of absolute indicator divergences
        avg_gradient_magnitude: Mean gradient magnitude across indicators
        leading_indicator: Indicator with highest gradient response
        response_order: Top 10 indicators by response magnitude
        n_affected_indicators: Count of indicators with above-median response
    """
    window_start: str
    window_end: str
    divergence_zscore: float
    total_divergence: float
    avg_gradient_magnitude: float
    leading_indicator: str
    response_order: List[str]
    n_affected_indicators: int

    def __eq__(self, other):
        if not isinstance(other, Transition):
            return False
        return self.window_end == other.window_end

    def __hash__(self):
        return hash(self.window_end)


def compute_system_divergence(
    field_df: pl.DataFrame,
    exclude_indicators: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Compute total system divergence per window.

    System divergence = sum of absolute indicator divergences.
    High values indicate regime change (field singularity).

    Args:
        field_df: Indicator field data with columns:
            - window_end: Date of window end
            - indicator_id: Indicator identifier
            - divergence: Laplace field divergence
            - gradient_magnitude: Magnitude of gradient vector
        exclude_indicators: List of indicator IDs to exclude (e.g., fault labels)

    Returns:
        DataFrame with columns:
            - window_end: Date
            - total_divergence: Sum of |divergence| across indicators
            - avg_gradient_mag: Mean gradient magnitude
            - n_indicators: Count of indicators in window
    """
    if exclude_indicators is None:
        exclude_indicators = []

    filtered = field_df
    if exclude_indicators:
        filtered = field_df.filter(
            ~pl.col('indicator_id').is_in(exclude_indicators)
        )

    return filtered.group_by('window_end').agg([
        pl.col('divergence').abs().sum().alias('total_divergence'),
        pl.col('gradient_magnitude').mean().alias('avg_gradient_mag'),
        pl.col('indicator_id').n_unique().alias('n_indicators'),
    ]).sort('window_end')


def detect_transitions(
    field_df: pl.DataFrame,
    zscore_threshold: float = 3.0,
    exclude_indicators: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, List[Transition]]:
    """
    Detect regime transitions using divergence z-scores.

    A transition is flagged when:
    - |divergence_zscore| > threshold (default 3.0)

    The z-score is computed relative to the median and standard deviation
    of total divergence across all windows.

    Args:
        field_df: indicator_field.parquet data with columns:
            - window_end, indicator_id, divergence, gradient_magnitude
        zscore_threshold: Standard deviations for significance (default 3.0)
        exclude_indicators: Indicators to exclude (e.g., ['TEP_FAULT'])

    Returns:
        Tuple of:
            - DataFrame of all windows with z-scores (for analysis)
            - List of detected Transition objects (significant transitions)

    Example:
        >>> system_div, transitions = detect_transitions(field_df, zscore_threshold=3.0)
        >>> print(f"Found {len(transitions)} transitions")
        >>> for t in transitions[:5]:
        ...     print(f"  {t.window_end}: z={t.divergence_zscore:.2f}, leader={t.leading_indicator}")
    """
    # Compute system divergence
    system_div = compute_system_divergence(field_df, exclude_indicators)

    if len(system_div) == 0:
        return system_div, []

    # Compute z-scores using median (robust to outliers)
    median_div = system_div['total_divergence'].median()
    std_div = system_div['total_divergence'].std()

    if std_div is None or std_div == 0:
        system_div = system_div.with_columns([
            pl.lit(0.0).alias('divergence_zscore')
        ])
        return system_div, []

    system_div = system_div.with_columns([
        ((pl.col('total_divergence') - median_div) / std_div).alias('divergence_zscore')
    ])

    # Find transition windows
    transition_windows = system_div.filter(
        pl.col('divergence_zscore').abs() > zscore_threshold
    ).sort('divergence_zscore', descending=True)

    # Build Transition objects with response analysis
    transitions = []

    for row in transition_windows.iter_rows(named=True):
        window_end = row['window_end']

        # Find leading indicators for this window
        window_data = field_df.filter(
            pl.col('window_end') == window_end
        )

        # Exclude specified indicators from leadership analysis
        if exclude_indicators:
            window_data = window_data.filter(
                ~pl.col('indicator_id').is_in(exclude_indicators)
            )

        window_data = window_data.sort('gradient_magnitude', descending=True)

        if len(window_data) > 0:
            leading = window_data['indicator_id'].head(1)[0]
            response_order = window_data['indicator_id'].head(10).to_list()

            # Count affected indicators (above median gradient)
            median_grad = window_data['gradient_magnitude'].median()
            if median_grad is not None:
                n_affected = len(window_data.filter(
                    pl.col('gradient_magnitude') > median_grad
                ))
            else:
                n_affected = len(window_data) // 2
        else:
            leading = 'unknown'
            response_order = []
            n_affected = 0

        transitions.append(Transition(
            window_start=str(window_end),
            window_end=str(window_end),
            divergence_zscore=row['divergence_zscore'],
            total_divergence=row['total_divergence'],
            avg_gradient_magnitude=row['avg_gradient_mag'],
            leading_indicator=leading,
            response_order=response_order,
            n_affected_indicators=n_affected,
        ))

    return system_div, transitions


def find_leading_indicators(
    field_df: pl.DataFrame,
    transition_windows: List,
    top_n: int = 5,
    exclude_indicators: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    For each transition window, identify which indicators responded first/strongest.

    Args:
        field_df: Indicator field data
        transition_windows: List of window_end dates (str or date objects)
        top_n: Number of top responders per window (default 5)
        exclude_indicators: Indicators to exclude from analysis

    Returns:
        DataFrame with columns:
            - window_end: Transition window date
            - rank: Response rank (1 = strongest)
            - indicator_id: Indicator identifier
            - gradient_magnitude: Response magnitude
            - divergence: Indicator divergence at transition

    Example:
        >>> leaders = find_leading_indicators(field_df, ['2000-07-28', '2000-02-27'])
        >>> print(leaders.filter(pl.col('rank') == 1))  # Show only top responders
    """
    if exclude_indicators is None:
        exclude_indicators = []

    results = []

    for window in transition_windows:
        # Convert to string if needed for comparison
        window_str = str(window)

        window_data = field_df.filter(
            pl.col('window_end').cast(pl.Utf8) == window_str
        )

        if exclude_indicators:
            window_data = window_data.filter(
                ~pl.col('indicator_id').is_in(exclude_indicators)
            )

        window_data = window_data.sort('gradient_magnitude', descending=True).head(top_n)

        for rank, row in enumerate(window_data.iter_rows(named=True), 1):
            results.append({
                'window_end': window_str,
                'rank': rank,
                'indicator_id': row['indicator_id'],
                'gradient_magnitude': row['gradient_magnitude'],
                'divergence': row['divergence'],
            })

    if not results:
        return pl.DataFrame({
            'window_end': [],
            'rank': [],
            'indicator_id': [],
            'gradient_magnitude': [],
            'divergence': [],
        })

    return pl.DataFrame(results)


def summarize_leading_indicators(
    transitions: List[Transition],
) -> pl.DataFrame:
    """
    Summarize which indicators lead transitions most frequently.

    Args:
        transitions: List of detected Transition objects

    Returns:
        DataFrame with columns:
            - indicator_id: Indicator identifier
            - lead_count: Number of times this indicator led a transition
            - lead_ratio: Proportion of transitions led
            - avg_zscore: Average z-score when leading

    Example:
        >>> _, transitions = detect_transitions(field_df)
        >>> summary = summarize_leading_indicators(transitions)
        >>> print(summary.head(5))  # Top 5 leading indicators
    """
    from collections import Counter

    if not transitions:
        return pl.DataFrame({
            'indicator_id': [],
            'lead_count': [],
            'lead_ratio': [],
            'avg_zscore': [],
        })

    # Count leadership frequency
    leaders = [t.leading_indicator for t in transitions]
    counts = Counter(leaders)

    # Compute average z-score when leading
    leader_zscores = {}
    for t in transitions:
        leader = t.leading_indicator
        if leader not in leader_zscores:
            leader_zscores[leader] = []
        leader_zscores[leader].append(abs(t.divergence_zscore))

    records = []
    total = len(transitions)
    for indicator, count in counts.most_common():
        records.append({
            'indicator_id': indicator,
            'lead_count': count,
            'lead_ratio': count / total,
            'avg_zscore': np.mean(leader_zscores[indicator]),
        })

    return pl.DataFrame(records)
