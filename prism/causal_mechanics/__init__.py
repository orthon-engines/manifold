"""
Causal Mechanics Layer
======================

Layer 4 of the ORTHON framework:

    Signal Typology     → WHAT is it?
    Behavioral Geometry → HOW does it behave?
    Dynamical Systems   → WHEN/HOW does it change?
    Causal Mechanics    → WHY does it change? (this package)

The Five Physics-Inspired Dimensions:
    1. Energy       - Is energy conserved? (Hamiltonian)
    2. Motion       - What are the equations of motion? (Lagrangian)
    3. Equilibrium  - Spontaneous or forced? (Gibbs free energy)
    4. Cycles       - What are the rotational dynamics? (Angular momentum)
    5. Flow         - How does momentum propagate? (Momentum flux)

Key Questions:
    - Is the system conservative, driven, or dissipative?
    - Is it spontaneously equilibrating or externally forced?
    - Is flow laminar, transitional, or turbulent?
    - What is the energy dominance (kinetic vs potential)?

Usage:
    >>> from prism.causal_mechanics import run_causal_mechanics, analyze_mechanics
    >>>
    >>> # Full analysis from signal
    >>> results = run_causal_mechanics(signal_data, entity_id='unit_1')
    >>> print(results['energy_class'])
    >>> print(results['system_class'])
    >>> print(results['engine_recommendations'])
    >>>
    >>> # Quick single analysis
    >>> result = analyze_mechanics(my_array)
    >>> print(result.typology.summary)

Architecture:
    run.py (entry point)
        │
        ▼
    orchestrator.py (routes + formats)
        │
        ▼
    engines/physics/* (computations)
        │
        ▼
    engine_mapping.py (action recommendations)
"""

__version__ = "1.0.0"
__author__ = "Ørthon Project"

# Orchestrator (main API)
from .orchestrator import (
    run_causal_mechanics,
    analyze_single_signal,
    detect_energy_transition,
    get_mechanics_fingerprint,
    mechanics_distance,
    classify_system,
    DIMENSION_NAMES,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_energy_classification,
    get_equilibrium_classification,
    get_flow_classification,
    get_intervention_recommendations,
    ENGINE_MAP,
    ENERGY_THRESHOLDS,
    INTERVENTION_MAP,
)

# Re-export layer classes from prism.layers
from ..layers.causal_mechanics import (
    CausalMechanicsLayer,
    CausalMechanicsOutput,
    MechanicsVector,
    MechanicsTypology,
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
    analyze_mechanics,
)

__all__ = [
    # Version
    "__version__",

    # Orchestrator API
    "run_causal_mechanics",
    "analyze_single_signal",
    "detect_energy_transition",
    "get_mechanics_fingerprint",
    "mechanics_distance",
    "classify_system",
    "DIMENSION_NAMES",

    # Engine mapping
    "select_engines",
    "get_energy_classification",
    "get_equilibrium_classification",
    "get_flow_classification",
    "get_intervention_recommendations",
    "ENGINE_MAP",
    "ENERGY_THRESHOLDS",
    "INTERVENTION_MAP",

    # Layer classes
    "CausalMechanicsLayer",
    "CausalMechanicsOutput",
    "MechanicsVector",
    "MechanicsTypology",
    "EnergyClass",
    "EquilibriumClass",
    "FlowClass",
    "OrbitClass",
    "DominanceClass",
    "analyze_mechanics",
]
