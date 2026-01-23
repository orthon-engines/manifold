"""
Causal Mechanics Engine Mapping
===============================

Maps physics-based analysis to recommended engines and intervention strategies.

The Five Physics Dimensions:
    1. Energy       - Conservative / Driven / Dissipative
    2. Motion       - Kinetic-dominant / Potential-dominant / Balanced
    3. Equilibrium  - Approaching / At equilibrium / Departing / Forced
    4. Cycles       - Circular / Elliptical / Irregular / Linear
    5. Flow         - Laminar / Transitional / Turbulent

Engine Selection Philosophy:
    - CONSERVATIVE systems → monitoring, prediction
    - DRIVEN systems → source identification, load analysis
    - DISSIPATIVE systems → intervention timing, restoration
    - TURBULENT flow → chaos management, stabilization

Intervention Philosophy:
    Causal Mechanics is the FINAL layer - it produces actionable recommendations:
    - When to intervene (timing)
    - How to intervene (method)
    - What outcome to expect (prediction)

Usage:
    from prism.causal_mechanics.engine_mapping import select_engines

    state = {
        'energy_class': 'DRIVEN',
        'equilibrium_class': 'DEPARTING',
        'flow_class': 'TRANSITIONAL',
        'H_cv': 0.3,
        'spontaneous': False,
    }

    engines = select_engines(state)
    interventions = get_intervention_recommendations(state)
"""

from typing import Dict, List, Optional


# =============================================================================
# THRESHOLDS
# =============================================================================

ENERGY_THRESHOLDS = {
    'conservative_cv': 0.1,      # H_cv below this → CONSERVATIVE
    'fluctuating_cv': 0.3,       # H_cv above this → FLUCTUATING
    'driven_trend': 0.01,        # H_trend above this → DRIVEN
    'dissipative_trend': -0.01,  # H_trend below this → DISSIPATIVE
}

EQUILIBRIUM_THRESHOLDS = {
    'approaching_delta_G': -0.05,  # delta_G below this → APPROACHING
    'departing_delta_G': 0.05,     # delta_G above this → DEPARTING
    'forced_temperature': 2.0,     # Temperature above this → FORCED
}

FLOW_THRESHOLDS = {
    'laminar_reynolds': 10,        # Reynolds below this → LAMINAR
    'turbulent_reynolds': 100,     # Reynolds above this → TURBULENT
    'turbulent_intensity': 0.5,    # Turbulence intensity above this → TURBULENT
}


# =============================================================================
# ENGINE RECOMMENDATIONS
# =============================================================================

ENGINE_MAP = {
    # Energy-based engines
    'CONSERVATIVE': [
        'hamiltonian_tracking',
        'energy_conservation_test',
        'adiabatic_analysis',
        'integrable_motion',
    ],
    'DRIVEN': [
        'external_force_estimation',
        'energy_injection_rate',
        'driving_frequency_analysis',
        'resonance_detection',
    ],
    'DISSIPATIVE': [
        'damping_estimation',
        'energy_loss_rate',
        'viscous_analysis',
        'relaxation_time',
    ],
    'FLUCTUATING': [
        'stochastic_energy',
        'energy_variance_tracking',
        'intermittency_analysis',
    ],

    # Motion-based engines
    'KINETIC': [
        'momentum_analysis',
        'velocity_tracking',
        'inertial_behavior',
    ],
    'POTENTIAL': [
        'position_analysis',
        'spring_constant_estimation',
        'restoring_force',
    ],
    'BALANCED': [
        'phase_space_analysis',
        'virial_theorem',
        'equipartition_check',
    ],

    # Equilibrium-based engines
    'APPROACHING': [
        'relaxation_dynamics',
        'equilibration_rate',
        'spontaneous_process',
    ],
    'AT_EQUILIBRIUM': [
        'steady_state_monitoring',
        'fluctuation_analysis',
        'detailed_balance',
    ],
    'DEPARTING': [
        'instability_growth',
        'departure_rate',
        'bifurcation_approach',
    ],
    'FORCED': [
        'external_forcing_analysis',
        'steady_state_forcing',
        'non_equilibrium_steady_state',
    ],

    # Flow-based engines
    'LAMINAR': [
        'streamline_analysis',
        'viscous_flow',
        'predictable_dynamics',
    ],
    'TRANSITIONAL': [
        'intermittent_turbulence',
        'transition_monitoring',
        'critical_reynolds_tracking',
    ],
    'TURBULENT': [
        'cascade_analysis',
        'eddy_viscosity',
        'kolmogorov_scaling',
        'mixing_analysis',
    ],

    # Orbit-based engines
    'CIRCULAR': [
        'constant_frequency',
        'stable_oscillation',
        'phase_lock_analysis',
    ],
    'ELLIPTICAL': [
        'period_variation',
        'precession_analysis',
        'quasi_periodic',
    ],
    'IRREGULAR': [
        'chaos_detection',
        'strange_attractor',
        'sensitive_dependence',
    ],
    'LINEAR': [
        'unidirectional_flow',
        'drift_analysis',
    ],
}

# Compound engines for specific state combinations
COMPOUND_ENGINES = {
    ('DRIVEN', 'DEPARTING'): [
        'forced_instability',
        'runaway_detection',
        'critical_threshold',
    ],
    ('DISSIPATIVE', 'APPROACHING'): [
        'damped_oscillation',
        'exponential_decay',
        'return_to_equilibrium',
    ],
    ('TURBULENT', 'FLUCTUATING'): [
        'turbulent_energy_cascade',
        'intermittent_bursts',
        'extreme_events',
    ],
    ('CONSERVATIVE', 'CIRCULAR'): [
        'perfect_oscillator',
        'harmonic_motion',
        'phase_stability',
    ],
}


# =============================================================================
# INTERVENTION RECOMMENDATIONS
# =============================================================================

INTERVENTION_MAP = {
    # Energy interventions
    'DRIVEN': {
        'action': 'REDUCE_DRIVING',
        'timing': 'BEFORE_SATURATION',
        'methods': ['reduce_external_input', 'damping_increase', 'load_shedding'],
    },
    'DISSIPATIVE': {
        'action': 'RESTORE_ENERGY',
        'timing': 'BEFORE_EXHAUSTION',
        'methods': ['energy_injection', 'friction_reduction', 'maintenance'],
    },

    # Equilibrium interventions
    'DEPARTING': {
        'action': 'STABILIZE',
        'timing': 'IMMEDIATE',
        'methods': ['feedback_control', 'constraint_addition', 'anchor_point'],
    },
    'FORCED': {
        'action': 'REMOVE_FORCING',
        'timing': 'GRADUAL',
        'methods': ['force_reduction', 'natural_relaxation', 'soft_landing'],
    },

    # Flow interventions
    'TURBULENT': {
        'action': 'LAMINARIZE',
        'timing': 'WHEN_POSSIBLE',
        'methods': ['reynolds_reduction', 'stabilization', 'flow_control'],
    },
    'TRANSITIONAL': {
        'action': 'PREVENT_TURBULENCE',
        'timing': 'PROACTIVE',
        'methods': ['damping', 'boundary_control', 'disturbance_rejection'],
    },

    # Orbit interventions
    'IRREGULAR': {
        'action': 'REGULARIZE',
        'timing': 'CONTINUOUS',
        'methods': ['phase_lock', 'synchronization', 'chaos_control'],
    },
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def select_engines(state: Dict) -> List[str]:
    """
    Select recommended engines based on mechanics state.

    Args:
        state: Dict with energy_class, equilibrium_class, flow_class, etc.

    Returns:
        Prioritized list of recommended engines
    """
    engines = []

    # Get state classifications
    energy = state.get('energy_class', 'CONSERVATIVE').upper()
    equilibrium = state.get('equilibrium_class', 'AT_EQUILIBRIUM').upper()
    flow = state.get('flow_class', 'LAMINAR').upper()
    orbit = state.get('orbit_class', 'LINEAR').upper()
    motion = state.get('motion_class', 'BALANCED').upper()

    # Add energy engines
    if energy in ENGINE_MAP:
        engines.extend(ENGINE_MAP[energy])

    # Add motion engines
    if motion in ENGINE_MAP:
        engines.extend(ENGINE_MAP[motion])

    # Add equilibrium engines
    if equilibrium in ENGINE_MAP:
        engines.extend(ENGINE_MAP[equilibrium])

    # Add flow engines
    if flow in ENGINE_MAP:
        engines.extend(ENGINE_MAP[flow])

    # Add orbit engines
    if orbit in ENGINE_MAP:
        engines.extend(ENGINE_MAP[orbit])

    # Add compound engines for special combinations
    for combo, combo_engines in COMPOUND_ENGINES.items():
        if all(c in [energy, equilibrium, flow, orbit, motion] for c in combo):
            engines.extend(combo_engines)

    # Deduplicate while preserving order
    seen = set()
    prioritized = []
    for e in engines:
        if e not in seen:
            seen.add(e)
            prioritized.append(e)

    return prioritized


def get_energy_classification(H_cv: float, H_trend: float) -> str:
    """
    Classify energy regime from Hamiltonian metrics.

    Args:
        H_cv: Coefficient of variation of Hamiltonian
        H_trend: Trend in Hamiltonian (dH/dt)

    Returns:
        Energy classification string
    """
    if H_cv < ENERGY_THRESHOLDS['conservative_cv']:
        return 'CONSERVATIVE'
    elif H_trend > ENERGY_THRESHOLDS['driven_trend']:
        return 'DRIVEN'
    elif H_trend < ENERGY_THRESHOLDS['dissipative_trend']:
        return 'DISSIPATIVE'
    else:
        return 'FLUCTUATING'


def get_equilibrium_classification(
    delta_G: float,
    temperature: float,
    spontaneous: bool,
) -> str:
    """
    Classify equilibrium state from Gibbs metrics.

    Args:
        delta_G: Change in Gibbs free energy
        temperature: System temperature (volatility)
        spontaneous: Whether process is spontaneous

    Returns:
        Equilibrium classification string
    """
    if temperature > EQUILIBRIUM_THRESHOLDS['forced_temperature']:
        return 'FORCED'
    elif delta_G < EQUILIBRIUM_THRESHOLDS['approaching_delta_G']:
        return 'APPROACHING'
    elif delta_G > EQUILIBRIUM_THRESHOLDS['departing_delta_G']:
        return 'DEPARTING'
    else:
        return 'AT_EQUILIBRIUM'


def get_flow_classification(
    reynolds_proxy: float,
    turbulence_intensity: float,
) -> str:
    """
    Classify flow regime from momentum flux metrics.

    Args:
        reynolds_proxy: Inertial / viscous ratio
        turbulence_intensity: Flow variability measure

    Returns:
        Flow classification string
    """
    if turbulence_intensity > FLOW_THRESHOLDS['turbulent_intensity']:
        return 'TURBULENT'
    elif reynolds_proxy > FLOW_THRESHOLDS['turbulent_reynolds']:
        return 'TURBULENT'
    elif reynolds_proxy > FLOW_THRESHOLDS['laminar_reynolds']:
        return 'TRANSITIONAL'
    else:
        return 'LAMINAR'


def get_intervention_recommendations(state: Dict) -> Dict:
    """
    Get actionable intervention recommendations.

    Args:
        state: Current mechanics state

    Returns:
        Dict with recommended actions, timing, and methods
    """
    interventions = {
        'recommended_actions': [],
        'timing': 'MONITOR',
        'methods': [],
        'urgency': 'LOW',
    }

    energy = state.get('energy_class', '').upper()
    equilibrium = state.get('equilibrium_class', '').upper()
    flow = state.get('flow_class', '').upper()
    orbit = state.get('orbit_class', '').upper()

    # Collect all relevant interventions
    for key in [energy, equilibrium, flow, orbit]:
        if key in INTERVENTION_MAP:
            intervention = INTERVENTION_MAP[key]
            interventions['recommended_actions'].append(intervention['action'])
            interventions['methods'].extend(intervention['methods'])

            # Update timing to most urgent
            if intervention['timing'] == 'IMMEDIATE':
                interventions['timing'] = 'IMMEDIATE'
                interventions['urgency'] = 'HIGH'
            elif intervention['timing'] == 'PROACTIVE' and interventions['urgency'] != 'HIGH':
                interventions['timing'] = 'PROACTIVE'
                interventions['urgency'] = 'MEDIUM'

    # Deduplicate methods
    interventions['methods'] = list(dict.fromkeys(interventions['methods']))

    return interventions
