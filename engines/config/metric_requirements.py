"""
Per-metric minimum observation requirements.

These are not arbitrary. They are based on the mathematical
requirements of each computation.

DO NOT reduce these without understanding the math.

Tier 1: Basic Statistics (30+)
  - Central Limit Theorem threshold
  - Simple aggregations

Tier 2: Distribution Shape (50-100)
  - Higher moments need more samples for stability
  - Kurtosis especially sensitive to outliers

Tier 3: Information Theory (100-200)
  - Probability estimation from histograms
  - Entropy requires sufficient bin counts

Tier 4: Spectral (100+)
  - FFT resolution depends on sample count
  - Need cycles to estimate frequency content

Tier 5: Dynamics (500+) CRITICAL
  - Phase space reconstruction requires filled space
  - Neighbor finding needs nearby trajectories
  - Lyapunov divergence tracking needs time
  - RQA recurrence matrix needs density

Tier 6: Topology (300+) CRITICAL
  - Persistent homology needs point cloud density
  - Betti numbers meaningless with sparse data

Tier 7: Pair Metrics (50-200)
  - Cross-signal relationships
  - Conditional probability estimation
"""

from typing import Tuple, Dict, List


# =============================================================================
# METRIC MINIMUMS BY TIER
# =============================================================================

TIER_1_BASIC = {
    'mean': 30,
    'std': 30,
    'min': 30,
    'max': 30,
    'rms': 30,
    'peak': 30,
    'peak_to_peak': 30,
    'crest_factor': 30,
    'variance': 30,
    'median': 30,
    'range': 30,
}

TIER_2_DISTRIBUTION = {
    'skewness': 50,
    'kurtosis': 100,
    'percentile_1': 50,
    'percentile_5': 50,
    'percentile_25': 50,
    'percentile_50': 50,
    'percentile_75': 50,
    'percentile_95': 50,
    'percentile_99': 50,
    'iqr': 50,
    'cv': 50,  # coefficient of variation
    'mode': 100,
}

TIER_3_INFORMATION = {
    'sample_entropy': 100,
    'permutation_entropy': 100,
    'approximate_entropy': 150,
    'spectral_entropy': 100,
    'entropy': 100,
    'entropy_rate': 150,
    'hurst': 100,
    'hurst_dfa': 100,
    'hurst_rs': 100,
    'transfer_entropy': 200,
    'mutual_info': 100,
    'conditional_entropy': 150,
}

TIER_4_SPECTRAL = {
    'spectral_centroid': 100,
    'spectral_slope': 100,
    'spectral_bandwidth': 100,
    'spectral_rolloff': 100,
    'spectral_flatness': 100,
    'spectral_flux': 100,
    'dominant_frequency': 100,
    'frequency_bands': 100,
    'harmonics': 200,
    'thd': 200,  # total harmonic distortion
}

TIER_5_DYNAMICS = {
    # CRITICAL - DO NOT REDUCE
    # Phase space reconstruction requires sufficient points
    'lyapunov': 500,
    'lyapunov_exponent': 500,
    'lyapunov_spectrum': 1000,
    'max_lyapunov': 500,
    'rqa_recurrence_rate': 500,
    'rqa_determinism': 500,
    'rqa_laminarity': 500,
    'rqa_entropy': 500,
    'rqa_trapping_time': 500,
    'rqa_longest_diagonal': 500,
    'rqa_longest_vertical': 500,
    'attractor_dimension': 500,
    'correlation_dimension': 500,
    'embedding_dimension': 500,
    'dmd': 300,  # dynamic mode decomposition
}

TIER_6_TOPOLOGY = {
    # CRITICAL - DO NOT REDUCE
    # Persistent homology needs point cloud density
    'betti_0': 300,
    'betti_1': 300,
    'betti_2': 500,
    'persistence_entropy': 300,
    'total_persistence': 300,
    'persistence_landscape': 300,
    'wasserstein_distance': 300,
    'bottleneck_distance': 300,
}

TIER_7_PAIRS = {
    'correlation': 50,
    'covariance': 50,
    'cross_correlation': 100,
    'coherence': 100,
    'cointegration': 100,
    'granger': 100,
    'granger_causality': 100,
    'transfer_entropy_pair': 200,
    'mutual_info_pair': 100,
    'dtw': 100,  # dynamic time warping
    'phase_sync': 200,
}

# Combine all tiers
METRIC_MINIMUMS: Dict[str, int] = {}
METRIC_MINIMUMS.update(TIER_1_BASIC)
METRIC_MINIMUMS.update(TIER_2_DISTRIBUTION)
METRIC_MINIMUMS.update(TIER_3_INFORMATION)
METRIC_MINIMUMS.update(TIER_4_SPECTRAL)
METRIC_MINIMUMS.update(TIER_5_DYNAMICS)
METRIC_MINIMUMS.update(TIER_6_TOPOLOGY)
METRIC_MINIMUMS.update(TIER_7_PAIRS)

# Default for unknown metrics (conservative)
DEFAULT_MINIMUM = 100

# Marginal threshold multiplier (1.5x minimum = marginal)
MARGINAL_MULTIPLIER = 1.5


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_minimum(metric_name: str) -> int:
    """
    Get minimum observations required for a metric.

    Args:
        metric_name: Name of the metric (case-insensitive, underscores normalized)

    Returns:
        Minimum observation count required
    """
    # Normalize metric name
    normalized = metric_name.lower().replace('-', '_')

    # Direct lookup
    if normalized in METRIC_MINIMUMS:
        return METRIC_MINIMUMS[normalized]

    # Try partial match (e.g., 'rolling_entropy' -> 'entropy')
    for key, value in METRIC_MINIMUMS.items():
        if key in normalized or normalized in key:
            return value

    return DEFAULT_MINIMUM


def get_tier(metric_name: str) -> int:
    """
    Get the tier number for a metric.

    Returns:
        1-7 for known metrics, 0 for unknown
    """
    normalized = metric_name.lower().replace('-', '_')

    if normalized in TIER_1_BASIC:
        return 1
    elif normalized in TIER_2_DISTRIBUTION:
        return 2
    elif normalized in TIER_3_INFORMATION:
        return 3
    elif normalized in TIER_4_SPECTRAL:
        return 4
    elif normalized in TIER_5_DYNAMICS:
        return 5
    elif normalized in TIER_6_TOPOLOGY:
        return 6
    elif normalized in TIER_7_PAIRS:
        return 7
    else:
        return 0


def can_compute(metric_name: str, n_observations: int) -> Tuple[bool, str]:
    """
    Check if we have enough data to compute this metric.

    Args:
        metric_name: Name of the metric
        n_observations: Number of observations available

    Returns:
        Tuple of (can_compute: bool, message: str)

    Messages:
        - "OK" if sufficient data
        - "MARGINAL: ..." if borderline (1.0x to 1.5x minimum)
        - "REFUSED: ..." if insufficient
    """
    minimum = get_minimum(metric_name)

    if n_observations < minimum:
        return False, f"REFUSED: {metric_name} requires {minimum} observations, have {n_observations}"

    if n_observations < minimum * MARGINAL_MULTIPLIER:
        return True, f"MARGINAL: {metric_name} has borderline data ({n_observations}/{minimum}, recommend {int(minimum * MARGINAL_MULTIPLIER)}+)"

    return True, "OK"


def get_computable_metrics(n_observations: int, requested_metrics: List[str] = None) -> Dict[str, List[str]]:
    """
    Given observation count, return which metrics can be computed.

    Args:
        n_observations: Number of observations available
        requested_metrics: Optional list of specific metrics to check.
                          If None, checks all known metrics.

    Returns:
        Dict with keys:
            'can_compute': List of metrics with sufficient data
            'marginal': List of metrics with borderline data
            'refused': List of metrics with insufficient data
    """
    result = {
        'can_compute': [],
        'marginal': [],
        'refused': []
    }

    metrics_to_check = requested_metrics if requested_metrics else list(METRIC_MINIMUMS.keys())

    for metric in metrics_to_check:
        minimum = get_minimum(metric)

        if n_observations >= minimum * MARGINAL_MULTIPLIER:
            result['can_compute'].append(metric)
        elif n_observations >= minimum:
            result['marginal'].append(metric)
        else:
            result['refused'].append(metric)

    return result


def get_tier_summary(n_observations: int) -> Dict[str, str]:
    """
    Get summary of which tiers are available given observation count.

    Returns:
        Dict mapping tier name to status ('OK', 'PARTIAL', 'REFUSED')
    """
    tiers = {
        'Tier 1 (Basic Stats)': (TIER_1_BASIC, 30),
        'Tier 2 (Distribution)': (TIER_2_DISTRIBUTION, 100),
        'Tier 3 (Information)': (TIER_3_INFORMATION, 200),
        'Tier 4 (Spectral)': (TIER_4_SPECTRAL, 100),
        'Tier 5 (Dynamics)': (TIER_5_DYNAMICS, 500),
        'Tier 6 (Topology)': (TIER_6_TOPOLOGY, 300),
        'Tier 7 (Pairs)': (TIER_7_PAIRS, 200),
    }

    result = {}
    for tier_name, (metrics, typical_min) in tiers.items():
        min_required = min(metrics.values())
        max_required = max(metrics.values())

        if n_observations >= max_required * MARGINAL_MULTIPLIER:
            result[tier_name] = 'OK'
        elif n_observations >= min_required:
            result[tier_name] = 'PARTIAL'
        else:
            result[tier_name] = 'REFUSED'

    return result


def validate_and_report(n_observations: int, requested_metrics: List[str] = None) -> str:
    """
    Generate human-readable validation report.

    Args:
        n_observations: Number of observations available
        requested_metrics: Optional specific metrics to check

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "METRIC REQUIREMENTS VALIDATION",
        "=" * 60,
        f"Observations available: {n_observations}",
        ""
    ]

    # Tier summary
    lines.append("TIER STATUS:")
    lines.append("-" * 40)
    tier_summary = get_tier_summary(n_observations)
    for tier_name, status in tier_summary.items():
        icon = '✓' if status == 'OK' else '⚠' if status == 'PARTIAL' else '✗'
        lines.append(f"  {icon} {tier_name}: {status}")

    lines.append("")

    # Specific metrics
    computable = get_computable_metrics(n_observations, requested_metrics)

    if computable['can_compute']:
        lines.append(f"CAN COMPUTE ({len(computable['can_compute'])} metrics):")
        for m in sorted(computable['can_compute'])[:10]:  # Show first 10
            lines.append(f"  ✓ {m}")
        if len(computable['can_compute']) > 10:
            lines.append(f"  ... and {len(computable['can_compute']) - 10} more")

    if computable['marginal']:
        lines.append(f"\nMARGINAL ({len(computable['marginal'])} metrics):")
        for m in sorted(computable['marginal']):
            minimum = get_minimum(m)
            lines.append(f"  ⚠ {m} (need {minimum}, have {n_observations})")

    if computable['refused']:
        lines.append(f"\nREFUSED ({len(computable['refused'])} metrics):")
        for m in sorted(computable['refused']):
            minimum = get_minimum(m)
            lines.append(f"  ✗ {m} (need {minimum}, have {n_observations})")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# CRITICAL METRICS (DO NOT REDUCE)
# =============================================================================

CRITICAL_METRICS = {
    'lyapunov': 500,
    'lyapunov_exponent': 500,
    'rqa_determinism': 500,
    'rqa_recurrence_rate': 500,
    'betti_0': 300,
    'betti_1': 300,
    'attractor_dimension': 500,
}


def is_critical_metric(metric_name: str) -> bool:
    """Check if metric has non-negotiable minimum."""
    normalized = metric_name.lower().replace('-', '_')
    return normalized in CRITICAL_METRICS


def get_critical_warning(metric_name: str) -> str:
    """Get warning message for critical metrics."""
    if not is_critical_metric(metric_name):
        return ""

    minimum = CRITICAL_METRICS.get(metric_name.lower(), 500)

    return f"""
WARNING: {metric_name} is a CRITICAL metric.

The minimum of {minimum} observations is based on mathematical requirements:
- Phase space reconstruction needs sufficient points
- Neighbor finding requires nearby trajectories
- Statistical averaging needs many samples

Computing with fewer observations produces NOISE, not signal.
DO NOT reduce this minimum.
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'METRIC_MINIMUMS',
    'DEFAULT_MINIMUM',
    'get_minimum',
    'get_tier',
    'can_compute',
    'get_computable_metrics',
    'get_tier_summary',
    'validate_and_report',
    'is_critical_metric',
    'get_critical_warning',
    'TIER_1_BASIC',
    'TIER_2_DISTRIBUTION',
    'TIER_3_INFORMATION',
    'TIER_4_SPECTRAL',
    'TIER_5_DYNAMICS',
    'TIER_6_TOPOLOGY',
    'TIER_7_PAIRS',
]
