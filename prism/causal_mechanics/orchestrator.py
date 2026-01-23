"""
Causal Mechanics Orchestrator
=============================

Layer 4 of ORTHON framework: Answers WHY the system changes using physics analogies.

The Five Physics Dimensions:
    1. Energy       - Is energy conserved? (Hamiltonian)
    2. Motion       - What are the equations of motion? (Lagrangian)
    3. Equilibrium  - Spontaneous or forced? (Gibbs free energy)
    4. Cycles       - What are the rotational dynamics? (Angular momentum)
    5. Flow         - How does momentum propagate? (Momentum flux)

Architecture:
    run.py (entry point)
        │
        ▼
    orchestrator.py (this file - routes + formats)
        │
        ▼
    layers/causal_mechanics.py (CausalMechanicsLayer)
        │
        ▼
    engines/physics/* (computations)
        │
        ▼
    engine_mapping.py (action recommendations)

Usage:
    from prism.causal_mechanics.orchestrator import run_causal_mechanics

    results = run_causal_mechanics(signal_data, entity_id='unit_1')
    print(results['energy_class'])
    print(results['intervention_recommendations'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .engine_mapping import (
    select_engines,
    get_energy_classification,
    get_equilibrium_classification,
    get_flow_classification,
    get_intervention_recommendations,
)


# Dimension names (canonical order)
DIMENSION_NAMES = [
    'energy',
    'motion',
    'equilibrium',
    'cycles',
    'flow',
]


def run_causal_mechanics(
    signal: np.ndarray,
    entity_id: str = "",
    signal_id: str = "",
    dynamics_state: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Causal Mechanics analysis.

    Args:
        signal: 1D numpy array of signal values
        entity_id: Entity identifier
        signal_id: Signal identifier
        dynamics_state: Optional state from Dynamical Systems layer
        config: Optional configuration overrides

    Returns:
        {
            'energy_class': energy regime classification,
            'equilibrium_class': equilibrium state classification,
            'flow_class': flow regime classification,
            'orbit_class': orbit classification,
            'dominant_energy': kinetic vs potential,
            'system_class': overall system classification,
            'vector': MechanicsVector as dict,
            'typology': MechanicsTypology as dict,
            'engine_recommendations': [engines],
            'intervention_recommendations': {...},
            'alerts': [...],
            'metadata': {...}
        }
    """
    # Import layer for delegated computation
    from ..layers.causal_mechanics import CausalMechanicsLayer

    config = config or {}

    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=float)

    if len(signal) < 30:
        return _empty_result(entity_id, signal_id, "Insufficient data (need >= 30 points)")

    # Use layer for core analysis
    layer = CausalMechanicsLayer(config=config)
    output = layer.analyze(signal, entity_id=entity_id, signal_id=signal_id)

    # Get vector and typology
    vector = output.vector
    typology = output.typology

    # Build state for engine selection
    state = {
        'energy_class': typology.energy_class.value.upper(),
        'equilibrium_class': typology.equilibrium_class.value.upper(),
        'flow_class': typology.flow_class.value.upper(),
        'orbit_class': typology.orbit_class.value.upper(),
        'motion_class': typology.motion_class.value.upper(),
        'H_cv': vector.H_cv,
        'H_trend': vector.H_trend,
        'delta_G': vector.delta_G,
        'temperature': vector.temperature_mean,
        'spontaneous': vector.spontaneous,
        'reynolds_proxy': vector.reynolds_proxy,
        'turbulence_intensity': vector.turbulence_intensity,
    }

    # Select engines
    engines = select_engines(state)

    # Get intervention recommendations
    interventions = get_intervention_recommendations(state)

    # Include dynamics context if provided
    if dynamics_state:
        state['dynamics_regime'] = dynamics_state.get('regime', '')
        state['dynamics_stability'] = dynamics_state.get('stability', '')

    return {
        'energy_class': typology.energy_class.value.upper(),
        'equilibrium_class': typology.equilibrium_class.value.upper(),
        'flow_class': typology.flow_class.value.upper(),
        'orbit_class': typology.orbit_class.value.upper(),
        'dominant_energy': typology.dominant_energy.value.upper(),
        'motion_class': typology.motion_class.value.upper(),
        'system_class': typology.system_class,
        'vector': vector.to_dict(),
        'typology': typology.to_dict(),
        'engine_recommendations': engines,
        'intervention_recommendations': interventions,
        'alerts': typology.alerts,
        'summary': typology.summary,
        'confidence': typology.confidence,
        'metadata': {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_observations': len(signal),
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


def analyze_single_signal(
    signal: np.ndarray,
    entity_id: str = "",
    signal_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single signal.

    Args:
        signal: 1D numpy array
        entity_id: Entity identifier
        signal_id: Signal identifier
        config: Optional configuration

    Returns:
        Dict with mechanics analysis results
    """
    return run_causal_mechanics(signal, entity_id, signal_id, config=config)


def get_mechanics_fingerprint(state: Dict[str, Any]) -> np.ndarray:
    """
    Convert mechanics state to a fingerprint vector.

    Args:
        state: Dict with energy_class, equilibrium_class, flow_class, etc.

    Returns:
        numpy array encoding the state
    """
    # Encode each dimension as a value
    energy_map = {'CONSERVATIVE': 1.0, 'FLUCTUATING': 0.66, 'DISSIPATIVE': 0.33, 'DRIVEN': 0.0}
    equilibrium_map = {'AT_EQUILIBRIUM': 1.0, 'APPROACHING': 0.75, 'FORCED': 0.25, 'DEPARTING': 0.0}
    flow_map = {'LAMINAR': 1.0, 'TRANSITIONAL': 0.5, 'TURBULENT': 0.0}
    orbit_map = {'CIRCULAR': 1.0, 'ELLIPTICAL': 0.75, 'LINEAR': 0.5, 'IRREGULAR': 0.0}
    motion_map = {'BALANCED': 0.5, 'KINETIC': 1.0, 'POTENTIAL': 0.0}

    return np.array([
        energy_map.get(state.get('energy_class', '').upper(), 0.5),
        equilibrium_map.get(state.get('equilibrium_class', '').upper(), 0.5),
        flow_map.get(state.get('flow_class', '').upper(), 0.5),
        orbit_map.get(state.get('orbit_class', '').upper(), 0.5),
        motion_map.get(state.get('motion_class', '').upper(), 0.5),
    ])


def mechanics_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute distance between two mechanics fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, ~2.2 = maximally different)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_energy_transition(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect energy/mechanics transition between two states.

    Args:
        previous_state: Previous mechanics state
        current_state: Current mechanics state
        threshold: Change threshold for flagging transition

    Returns:
        Dict with transition detection results
    """
    fp_prev = get_mechanics_fingerprint(previous_state)
    fp_curr = get_mechanics_fingerprint(current_state)

    distance = mechanics_distance(fp_prev, fp_curr)

    # Find what changed
    changes = {}
    dim_keys = ['energy_class', 'equilibrium_class', 'flow_class', 'orbit_class', 'motion_class']

    for dim in dim_keys:
        changes[dim] = {
            'previous': previous_state.get(dim, ''),
            'current': current_state.get(dim, ''),
            'changed': previous_state.get(dim, '') != current_state.get(dim, ''),
        }

    changed_dims = [d for d, c in changes.items() if c['changed']]

    # Classify transition type
    if not changed_dims:
        transition_type = 'NONE'
    elif 'energy_class' in changed_dims:
        transition_type = 'ENERGY_TRANSITION'
    elif 'equilibrium_class' in changed_dims:
        transition_type = 'EQUILIBRIUM_SHIFT'
    elif 'flow_class' in changed_dims:
        transition_type = 'FLOW_REGIME_CHANGE'
    else:
        transition_type = 'MINOR_CHANGE'

    return {
        'transition_detected': distance >= threshold,
        'transition_type': transition_type,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'changed_dimensions': changed_dims,
    }


def classify_system(state: Dict[str, Any]) -> str:
    """
    Derive overall system classification from mechanics state.

    Args:
        state: Dict with energy_class, equilibrium_class, flow_class, orbit_class

    Returns:
        System classification string
    """
    energy = state.get('energy_class', '').upper()
    equilibrium = state.get('equilibrium_class', '').upper()
    flow = state.get('flow_class', '').upper()
    orbit = state.get('orbit_class', '').upper()

    # Priority-based classification
    if energy == 'CONSERVATIVE':
        if orbit == 'CIRCULAR':
            return 'Stable Oscillator'
        elif orbit == 'ELLIPTICAL':
            return 'Quasi-Periodic'
        elif flow == 'LAMINAR':
            return 'Conservative Laminar'
        else:
            return 'Conservative'

    elif energy == 'DRIVEN':
        if flow == 'TURBULENT':
            return 'Turbulent Driven'
        elif equilibrium == 'DEPARTING':
            return 'Forced Unstable'
        else:
            return 'Driven System'

    elif energy == 'DISSIPATIVE':
        if equilibrium == 'APPROACHING':
            return 'Damped Equilibrating'
        else:
            return 'Dissipative'

    elif flow == 'TURBULENT':
        return 'Turbulent'

    elif equilibrium == 'APPROACHING':
        return 'Equilibrating'

    elif equilibrium == 'DEPARTING':
        return 'Destabilizing'

    else:
        return 'Transitional'


def _empty_result(entity_id: str, signal_id: str, error: str) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'energy_class': 'UNDETERMINED',
        'equilibrium_class': 'UNDETERMINED',
        'flow_class': 'UNDETERMINED',
        'orbit_class': 'UNDETERMINED',
        'dominant_energy': 'UNDETERMINED',
        'motion_class': 'UNDETERMINED',
        'system_class': 'Unknown',
        'vector': {},
        'typology': {},
        'engine_recommendations': [],
        'intervention_recommendations': {'recommended_actions': [], 'timing': 'MONITOR', 'methods': [], 'urgency': 'LOW'},
        'alerts': [],
        'summary': '',
        'confidence': 0.0,
        'metadata': {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'error': error,
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }
