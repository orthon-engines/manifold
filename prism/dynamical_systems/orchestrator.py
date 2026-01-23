"""
Dynamical Systems Orchestrator
==============================

Layer 3 of ORTHON framework: Analyzes temporal evolution of behavioral geometry.

The Four Dimensions:
    1. Regime     - Coupled / Decoupled / Transitioning
    2. Stability  - Stable / Evolving / Unstable / Critical
    3. Trajectory - Converging / Diverging / Oscillating / Wandering
    4. Attractor  - Fixed point / Limit cycle / Strange / None

Architecture:
    run.py (entry point)
        │
        ▼
    orchestrator.py (this file - routes + formats)
        │
        ▼
    layers/dynamical_systems.py (DynamicalSystemsLayer)
        │
        ▼
    engines/dynamics/* (computations)
        │
        ▼
    engine_mapping.py (selects next-phase engines)

Usage:
    from prism.dynamical_systems.orchestrator import run_dynamical_systems

    results = run_dynamical_systems(geometry_history, timestamps)
    print(results['regime'])
    print(results['engine_recommendations'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .engine_mapping import (
    select_engines,
    get_regime_classification,
    get_stability_classification,
    get_trajectory_classification,
    should_escalate_to_mechanics,
)


# Dimension names (canonical order)
DIMENSION_NAMES = [
    'regime',
    'stability',
    'trajectory',
    'attractor',
]


def run_dynamical_systems(
    geometry_history: List[Dict[str, float]],
    timestamps: Optional[List[datetime]] = None,
    entity_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Dynamical Systems analysis.

    Args:
        geometry_history: List of geometry measurements over time
            Each dict should contain: mean_correlation, network_density, n_clusters, etc.
        timestamps: Optional timestamps for each measurement
        entity_id: Entity identifier
        config: Optional configuration overrides

    Returns:
        {
            'regime': current regime classification,
            'stability': current stability classification,
            'trajectory': trajectory classification,
            'attractor': attractor classification,
            'vector': DynamicsVector as dict,
            'typology': DynamicsTypology as dict,
            'engine_recommendations': [engines],
            'escalate_to_mechanics': bool,
            'transition_history': [...],
            'metadata': {...}
        }
    """
    # Import layer for delegated computation
    from ..layers.dynamical_systems import DynamicalSystemsLayer

    config = config or {}

    if len(geometry_history) < 2:
        return _empty_result(entity_id, "Insufficient data (need >= 2 geometry snapshots)")

    # Use layer for core analysis
    layer = DynamicalSystemsLayer(entity_id=entity_id, config=config)
    output = layer.analyze(geometry_history, timestamps)

    # Extract time series for detailed analysis
    correlations = [g.get('mean_correlation', 0.0) for g in geometry_history]
    densities = [g.get('network_density', 0.0) for g in geometry_history]

    # Compute changes
    corr_changes = np.diff(correlations)
    density_changes = np.diff(densities)

    # Current state
    current_corr = correlations[-1]
    current_corr_change = corr_changes[-1] if len(corr_changes) > 0 else 0.0
    current_density_change = density_changes[-1] if len(density_changes) > 0 else 0.0

    # Get classifications
    regime = get_regime_classification(current_corr, current_corr_change)
    stability = get_stability_classification(current_corr_change, current_density_change)
    trajectory = get_trajectory_classification(correlations, densities)

    # Build state for engine selection
    state = {
        'regime': regime,
        'stability': stability,
        'trajectory': trajectory,
        'attractor': output.typology.attractor_class.value.upper(),
        'correlation_change': current_corr_change,
        'density_change': current_density_change,
    }

    # Select engines
    engines = select_engines(state)

    # Check escalation
    escalate = should_escalate_to_mechanics(state)

    # Build transition history
    transition_history = _build_transition_history(
        correlations, densities, corr_changes, density_changes
    )

    return {
        'regime': regime,
        'stability': stability,
        'trajectory': trajectory,
        'attractor': output.typology.attractor_class.value.upper(),
        'vector': output.vector.to_dict(),
        'typology': output.typology.to_dict(),
        'engine_recommendations': engines,
        'escalate_to_mechanics': escalate,
        'transition_history': transition_history,
        'metadata': {
            'entity_id': entity_id,
            'n_observations': len(geometry_history),
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


def analyze_single_entity(
    geometry_df,
    entity_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single entity from a DataFrame.

    Args:
        geometry_df: DataFrame with geometry columns
        entity_id: Entity identifier
        config: Optional configuration

    Returns:
        Dict with dynamics analysis results
    """
    # Convert DataFrame to geometry_history format
    geometry_history = []

    for row in geometry_df.iter_rows(named=True):
        geometry_history.append({
            'mean_correlation': row.get('mean_correlation', 0.0),
            'network_density': row.get('network_density', 0.0),
            'n_clusters': row.get('n_clusters', 0),
            'n_signals': row.get('n_signals', 0),
        })

    timestamps = None
    if 'timestamp' in geometry_df.columns:
        timestamps = geometry_df['timestamp'].to_list()

    return run_dynamical_systems(geometry_history, timestamps, entity_id, config)


def get_trajectory_fingerprint(state: Dict[str, Any]) -> np.ndarray:
    """
    Convert dynamical state to a fingerprint vector.

    Args:
        state: Dict with regime, stability, trajectory, attractor

    Returns:
        numpy array encoding the state
    """
    # Encode each dimension as a value
    regime_map = {'COUPLED': 1.0, 'MODERATE': 0.5, 'DECOUPLED': 0.0, 'TRANSITIONING': 0.75}
    stability_map = {'STABLE': 1.0, 'EVOLVING': 0.66, 'UNSTABLE': 0.33, 'CRITICAL': 0.0}
    trajectory_map = {'CONVERGING': 1.0, 'OSCILLATING': 0.66, 'WANDERING': 0.33, 'DIVERGING': 0.0}
    attractor_map = {'FIXED_POINT': 1.0, 'LIMIT_CYCLE': 0.75, 'STRANGE': 0.25, 'NONE': 0.5}

    return np.array([
        regime_map.get(state.get('regime', '').upper(), 0.5),
        stability_map.get(state.get('stability', '').upper(), 0.5),
        trajectory_map.get(state.get('trajectory', '').upper(), 0.5),
        attractor_map.get(state.get('attractor', '').upper(), 0.5),
    ])


def trajectory_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute distance between two trajectory fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, 2 = maximally different)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_regime_transition(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect regime transition between two states.

    Args:
        previous_state: Previous dynamical state
        current_state: Current dynamical state
        threshold: Change threshold for flagging transition

    Returns:
        Dict with transition detection results
    """
    fp_prev = get_trajectory_fingerprint(previous_state)
    fp_curr = get_trajectory_fingerprint(current_state)

    distance = trajectory_distance(fp_prev, fp_curr)

    # Find what changed
    changes = {}
    for i, dim in enumerate(DIMENSION_NAMES):
        changes[dim] = {
            'previous': previous_state.get(dim, ''),
            'current': current_state.get(dim, ''),
            'changed': previous_state.get(dim, '') != current_state.get(dim, ''),
        }

    changed_dims = [d for d, c in changes.items() if c['changed']]

    # Classify transition type
    if not changed_dims:
        transition_type = 'NONE'
    elif 'regime' in changed_dims:
        transition_type = 'REGIME_SHIFT'
    elif 'stability' in changed_dims:
        transition_type = 'STABILITY_CHANGE'
    elif 'trajectory' in changed_dims:
        transition_type = 'TRAJECTORY_SHIFT'
    else:
        transition_type = 'ATTRACTOR_CHANGE'

    return {
        'transition_detected': distance >= threshold,
        'transition_type': transition_type,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'changed_dimensions': changed_dims,
    }


def _build_transition_history(
    correlations: List[float],
    densities: List[float],
    corr_changes: np.ndarray,
    density_changes: np.ndarray,
) -> List[Dict]:
    """Build history of state transitions."""
    history = []

    for i in range(len(corr_changes)):
        regime = get_regime_classification(correlations[i+1], corr_changes[i])
        stability = get_stability_classification(corr_changes[i], density_changes[i])

        history.append({
            'index': i + 1,
            'regime': regime,
            'stability': stability,
            'correlation': correlations[i+1],
            'correlation_change': float(corr_changes[i]),
            'density_change': float(density_changes[i]),
        })

    return history


def _empty_result(entity_id: str, error: str) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'regime': 'UNDETERMINED',
        'stability': 'UNDETERMINED',
        'trajectory': 'UNDETERMINED',
        'attractor': 'UNDETERMINED',
        'vector': {},
        'typology': {},
        'engine_recommendations': [],
        'escalate_to_mechanics': False,
        'transition_history': [],
        'metadata': {
            'entity_id': entity_id,
            'error': error,
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }
