"""
Dynamical Systems Engine Mapping
================================

Maps dynamical state to recommended analysis engines and next-layer escalation.

The Four Dimensions:
    1. Regime     - Coupled / Decoupled / Transitioning
    2. Stability  - Stable / Evolving / Unstable / Critical
    3. Trajectory - Converging / Diverging / Oscillating / Wandering
    4. Attractor  - Fixed point / Limit cycle / Strange / None

Engine Selection Philosophy:
    - STABLE regimes → confirmatory engines, monitoring
    - TRANSITIONING regimes → detection engines, early warning
    - CRITICAL states → prediction engines, intervention timing
    - STRANGE attractors → chaos engines, embedding analysis

Usage:
    from prism.dynamical_systems.engine_mapping import select_engines

    state = {
        'regime': 'COUPLED',
        'stability': 'EVOLVING',
        'correlation_change': 0.15,
        'trajectory': 'CONVERGING',
    }

    engines = select_engines(state)
    # ['trend_extrapolation', 'regime_persistence', ...]
"""

from typing import Dict, List, Optional


# =============================================================================
# THRESHOLDS
# =============================================================================

STABILITY_THRESHOLDS = {
    'stable_change': 0.05,     # Max change for STABLE
    'evolving_change': 0.15,   # Max change for EVOLVING
    'unstable_change': 0.30,   # Max change for UNSTABLE
    # Above 0.30 → CRITICAL
}

REGIME_THRESHOLDS = {
    'coupled': 0.7,            # Correlation above this → COUPLED
    'decoupled': 0.3,          # Correlation below this → DECOUPLED
    'transitioning': 0.2,      # Change rate above this → TRANSITIONING
}

TRAJECTORY_THRESHOLDS = {
    'convergence_trend': 0.01,   # Positive trend → CONVERGING
    'divergence_trend': -0.01,   # Negative trend → DIVERGING
    'oscillation_ratio': 0.5,    # Sign changes above this → OSCILLATING
}


# =============================================================================
# ENGINE RECOMMENDATIONS
# =============================================================================

ENGINE_MAP = {
    # Regime-based engines
    'COUPLED': [
        'correlation_tracking',
        'vector_autoregression',
        'cointegration_test',
        'granger_causality',
        'dynamic_factor_model',
    ],
    'DECOUPLED': [
        'independence_test',
        'structural_break_detection',
        'regime_shift_detector',
        'isolation_analysis',
    ],
    'TRANSITIONING': [
        'changepoint_detection',
        'regime_hmm',
        'early_warning_indicators',
        'tipping_point_analysis',
        'critical_slowing_down',
    ],

    # Stability-based engines
    'STABLE': [
        'equilibrium_monitoring',
        'variance_tracking',
        'steady_state_analysis',
    ],
    'EVOLVING': [
        'trend_estimation',
        'drift_detection',
        'adaptive_filtering',
        'kalman_smoothing',
    ],
    'UNSTABLE': [
        'volatility_modeling',
        'jump_detection',
        'shock_propagation',
        'cascade_analysis',
    ],
    'CRITICAL': [
        'bifurcation_analysis',
        'catastrophe_theory',
        'phase_transition_detector',
        'collapse_warning',
    ],

    # Trajectory-based engines
    'CONVERGING': [
        'equilibrium_estimation',
        'attractor_reconstruction',
        'convergence_rate',
        'basin_analysis',
    ],
    'DIVERGING': [
        'instability_analysis',
        'lyapunov_exponent',
        'escape_rate',
        'repeller_detection',
    ],
    'OSCILLATING': [
        'limit_cycle_detection',
        'period_estimation',
        'amplitude_tracking',
        'phase_analysis',
    ],
    'WANDERING': [
        'random_walk_test',
        'diffusion_analysis',
        'ergodicity_test',
    ],

    # Attractor-based engines
    'FIXED_POINT': [
        'fixed_point_stability',
        'linear_stability_analysis',
        'eigenvalue_analysis',
    ],
    'LIMIT_CYCLE': [
        'floquet_analysis',
        'poincare_section',
        'return_map',
    ],
    'STRANGE': [
        'correlation_dimension',
        'recurrence_quantification',
        'embedding_dimension',
        'information_dimension',
    ],
}

# Compound engines for specific state combinations
COMPOUND_ENGINES = {
    ('TRANSITIONING', 'CRITICAL'): [
        'impending_bifurcation',
        'tipping_cascade',
        'critical_transition_indicator',
    ],
    ('COUPLED', 'UNSTABLE'): [
        'synchronized_instability',
        'contagion_analysis',
        'cascade_propagation',
    ],
    ('DECOUPLED', 'CONVERGING'): [
        'independent_equilibration',
        'parallel_relaxation',
    ],
    ('OSCILLATING', 'LIMIT_CYCLE'): [
        'phase_locking',
        'synchronization_analysis',
        'entrainment_detection',
    ],
}


# =============================================================================
# ESCALATION CRITERIA
# =============================================================================

# When to escalate to Causal Mechanics (Layer 4)
ESCALATION_CRITERIA = {
    'always_escalate': ['CRITICAL', 'STRANGE'],
    'escalate_if_unstable': ['TRANSITIONING', 'DIVERGING'],
    'escalate_threshold': 0.25,  # Change rate threshold
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def select_engines(state: Dict) -> List[str]:
    """
    Select recommended engines based on dynamical state.

    Args:
        state: Dict with regime, stability, trajectory, attractor keys

    Returns:
        Prioritized list of recommended engines
    """
    engines = []

    # Get state classifications
    regime = state.get('regime', 'MODERATE').upper()
    stability = state.get('stability', 'STABLE').upper()
    trajectory = state.get('trajectory', 'WANDERING').upper()
    attractor = state.get('attractor', 'NONE').upper()

    # Add regime engines
    if regime in ENGINE_MAP:
        engines.extend(ENGINE_MAP[regime])

    # Add stability engines
    if stability in ENGINE_MAP:
        engines.extend(ENGINE_MAP[stability])

    # Add trajectory engines
    if trajectory in ENGINE_MAP:
        engines.extend(ENGINE_MAP[trajectory])

    # Add attractor engines
    if attractor in ENGINE_MAP:
        engines.extend(ENGINE_MAP[attractor])

    # Add compound engines for special combinations
    for combo, combo_engines in COMPOUND_ENGINES.items():
        if all(c in [regime, stability, trajectory, attractor] for c in combo):
            engines.extend(combo_engines)

    # Deduplicate while preserving order
    seen = set()
    prioritized = []
    for e in engines:
        if e not in seen:
            seen.add(e)
            prioritized.append(e)

    return prioritized


def get_regime_classification(
    correlation: float,
    correlation_change: float,
) -> str:
    """
    Classify regime from correlation metrics.

    Args:
        correlation: Current correlation level [0, 1]
        correlation_change: Rate of correlation change

    Returns:
        Regime classification string
    """
    if abs(correlation_change) > REGIME_THRESHOLDS['transitioning']:
        return 'TRANSITIONING'
    elif correlation > REGIME_THRESHOLDS['coupled']:
        return 'COUPLED'
    elif correlation < REGIME_THRESHOLDS['decoupled']:
        return 'DECOUPLED'
    else:
        return 'MODERATE'


def get_stability_classification(
    correlation_change: float,
    density_change: float,
) -> str:
    """
    Classify stability from change metrics.

    Args:
        correlation_change: Change in correlation
        density_change: Change in network density

    Returns:
        Stability classification string
    """
    total_change = abs(correlation_change) + abs(density_change)

    if total_change < STABILITY_THRESHOLDS['stable_change']:
        return 'STABLE'
    elif total_change < STABILITY_THRESHOLDS['evolving_change']:
        return 'EVOLVING'
    elif total_change < STABILITY_THRESHOLDS['unstable_change']:
        return 'UNSTABLE'
    else:
        return 'CRITICAL'


def get_trajectory_classification(
    correlation_series: List[float],
    density_series: List[float],
) -> str:
    """
    Classify trajectory from time series history.

    Args:
        correlation_series: Recent correlation values
        density_series: Recent density values

    Returns:
        Trajectory classification string
    """
    import numpy as np

    if len(correlation_series) < 3:
        return 'WANDERING'

    # Compute trend
    x = np.arange(len(correlation_series))
    trend = np.polyfit(x, correlation_series, 1)[0]

    # Compute oscillation
    diff = np.diff(correlation_series)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    oscillation_ratio = sign_changes / len(diff) if len(diff) > 0 else 0

    if oscillation_ratio > TRAJECTORY_THRESHOLDS['oscillation_ratio']:
        return 'OSCILLATING'
    elif trend > TRAJECTORY_THRESHOLDS['convergence_trend']:
        return 'CONVERGING'
    elif trend < TRAJECTORY_THRESHOLDS['divergence_trend']:
        return 'DIVERGING'
    else:
        return 'WANDERING'


def should_escalate_to_mechanics(state: Dict) -> bool:
    """
    Determine if analysis should escalate to Causal Mechanics layer.

    Args:
        state: Current dynamical state

    Returns:
        True if escalation is recommended
    """
    regime = state.get('regime', '').upper()
    stability = state.get('stability', '').upper()
    trajectory = state.get('trajectory', '').upper()
    attractor = state.get('attractor', '').upper()

    # Always escalate for critical states
    for key in [regime, stability, trajectory, attractor]:
        if key in ESCALATION_CRITERIA['always_escalate']:
            return True

    # Escalate if unstable in certain regimes
    if stability == 'UNSTABLE':
        for key in [regime, trajectory]:
            if key in ESCALATION_CRITERIA['escalate_if_unstable']:
                return True

    # Escalate based on change rate
    change_rate = abs(state.get('correlation_change', 0)) + abs(state.get('density_change', 0))
    if change_rate > ESCALATION_CRITERIA['escalate_threshold']:
        return True

    return False
