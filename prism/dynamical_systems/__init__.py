"""
Dynamical Systems Layer
=======================

Layer 3 of the ORTHON framework:

    Signal Typology     → WHAT is it?
    Behavioral Geometry → HOW does it behave?
    Dynamical Systems   → WHEN/HOW does it change? (this package)
    Causal Mechanics    → WHY does it change?

The Four Dimensions of Dynamical Analysis:
    1. Regime     - What dynamical state is the system in?
    2. Stability  - Is the system stable or transitioning?
    3. Trajectory - Where is the system heading?
    4. Attractor  - What states does it tend toward?

Key Questions:
    - Is the system in a coupled, decoupled, or transitioning regime?
    - Is the current state stable, evolving, or critical?
    - Is the trajectory converging, diverging, or oscillating?
    - Are there attractors (fixed points, limit cycles, strange attractors)?

Usage:
    >>> from prism.dynamical_systems import run_dynamical_systems, analyze_dynamics
    >>>
    >>> # Full analysis
    >>> results = run_dynamical_systems(geometry_history, timestamps)
    >>> print(results['regime'])
    >>> print(results['stability'])
    >>> print(results['engine_recommendations'])
    >>>
    >>> # Quick single analysis
    >>> result = analyze_dynamics(geometry_df)
    >>> print(result.typology.summary)

Architecture:
    run.py (entry point)
        │
        ▼
    orchestrator.py (routes + formats)
        │
        ▼
    engines/dynamics/* (computations)
        │
        ▼
    engine_mapping.py (selects next-phase engines)
"""

__version__ = "1.0.0"
__author__ = "Ørthon Project"

# Orchestrator (main API)
from .orchestrator import (
    run_dynamical_systems,
    analyze_single_entity,
    detect_regime_transition,
    get_trajectory_fingerprint,
    trajectory_distance,
    DIMENSION_NAMES,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_regime_classification,
    get_stability_classification,
    get_trajectory_classification,
    should_escalate_to_mechanics,
    ENGINE_MAP,
    STABILITY_THRESHOLDS,
)

# Re-export layer classes from prism.layers
from ..layers.dynamical_systems import (
    DynamicalSystemsLayer,
    DynamicalSystemsOutput,
    DynamicsVector,
    DynamicsTypology,
    RegimeClass,
    StabilityClass,
    TrajectoryClass,
    AttractorClass,
    analyze_dynamics,
)

__all__ = [
    # Version
    "__version__",

    # Orchestrator API
    "run_dynamical_systems",
    "analyze_single_entity",
    "detect_regime_transition",
    "get_trajectory_fingerprint",
    "trajectory_distance",
    "DIMENSION_NAMES",

    # Engine mapping
    "select_engines",
    "get_regime_classification",
    "get_stability_classification",
    "get_trajectory_classification",
    "should_escalate_to_mechanics",
    "ENGINE_MAP",
    "STABILITY_THRESHOLDS",

    # Layer classes
    "DynamicalSystemsLayer",
    "DynamicalSystemsOutput",
    "DynamicsVector",
    "DynamicsTypology",
    "RegimeClass",
    "StabilityClass",
    "TrajectoryClass",
    "AttractorClass",
    "analyze_dynamics",
]
