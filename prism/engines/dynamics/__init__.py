"""
Dynamics Engines.

Temporal dynamics and chaos analysis.
- lyapunov: stability/chaos indicator (Rosenstein algorithm)
- attractor: phase space reconstruction
- critical_slowing_down: early warning signals for B-tipping
- formal_definitions: classification framework for stability analysis
"""

from . import lyapunov
from . import attractor
from . import critical_slowing_down
from . import formal_definitions

# Convenience imports
from .lyapunov import compute as compute_lyapunov
from .critical_slowing_down import compute as compute_csd
from .formal_definitions import (
    AttractorType,
    StabilityType,
    FailureMode,
    TippingType,
    SystemTypology,
    GeometryMetrics,
    MassMetrics,
    EarlyWarningSignals,
    FormalAssessment,
    classify_failure_mode,
    classify_tipping_type,
    classify_stability,
)

__all__ = [
    # Modules
    'lyapunov',
    'attractor',
    'critical_slowing_down',
    'formal_definitions',
    # Functions
    'compute_lyapunov',
    'compute_csd',
    # Enums
    'AttractorType',
    'StabilityType',
    'FailureMode',
    'TippingType',
    'SystemTypology',
    # Dataclasses
    'GeometryMetrics',
    'MassMetrics',
    'EarlyWarningSignals',
    'FormalAssessment',
    # Classification
    'classify_failure_mode',
    'classify_tipping_type',
    'classify_stability',
]
