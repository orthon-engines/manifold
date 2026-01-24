"""
ORTHON Classification Thresholds
================================

Centralized configuration for all classification thresholds across
the four ORTHON analytical layers.

Principle: Thresholds define when states change, marking "relevant times".
Adjusting these parameters changes transition sensitivity without recomputation.

Usage:
    from prism.config.thresholds import (
        TYPOLOGY_THRESHOLDS,
        GEOMETRY_THRESHOLDS,
        DYNAMICS_THRESHOLDS,
        TRANSITION_THRESHOLDS,
    )

Modification:
    To change classification sensitivity, modify values here.
    Lower thresholds = more sensitive (more transitions detected).
    Higher thresholds = less sensitive (fewer transitions detected).
"""

# =============================================================================
# SIGNAL TYPOLOGY THRESHOLDS
# =============================================================================
# Used in: prism/signal_typology/classify.py
# Purpose: Map 0-1 normalized scores to 5-level labels

TYPOLOGY_CLASSIFICATION = {
    # Default thresholds: [strong_low, weak_low, weak_high, strong_high]
    # Scores map to: [0, t1) -> strong_low, [t1, t2) -> weak_low, etc.
    'default': [0.25, 0.40, 0.60, 0.75],

    # Alternative presets
    'strict': [0.20, 0.35, 0.65, 0.80],   # Wider indeterminate zone
    'loose': [0.30, 0.45, 0.55, 0.70],    # Narrower indeterminate zone
    'binary': [0.50, 0.50, 0.50, 0.50],   # Simple high/low split
}

# Per-axis threshold overrides (axis-specific tuning)
TYPOLOGY_AXIS_THRESHOLDS = {
    'memory': [0.30, 0.45, 0.55, 0.70],       # Hurst clusters around 0.5
    'information': [0.25, 0.40, 0.60, 0.75],  # Default
    'frequency': [0.25, 0.40, 0.60, 0.75],    # Default
    'volatility': [0.25, 0.40, 0.60, 0.75],   # Default
    'wavelet': [0.25, 0.40, 0.60, 0.75],      # Default
    'derivatives': [0.20, 0.35, 0.65, 0.80],  # Kurtosis can be extreme
    'recurrence': [0.25, 0.40, 0.60, 0.75],   # Default
    'discontinuity': [0.15, 0.30, 0.50, 0.70],# Level shifts are rare
    'momentum': [0.30, 0.45, 0.55, 0.70],     # Same as memory (Hurst-based)
}

# Signal typology regime change detection
TYPOLOGY_TRANSITION = {
    'distance_threshold': 0.3,  # Fingerprint distance for regime change
    'axis_threshold': 0.3,      # Per-axis change for "moving" classification
}


# =============================================================================
# STRUCTURAL GEOMETRY THRESHOLDS
# =============================================================================
# Used in: prism/structural_geometry/engine_mapping.py
# Purpose: Classify network topology, stability, and leadership

GEOMETRY_TOPOLOGY = {
    'highly_connected_density': 0.7,   # Density above this = HIGHLY_CONNECTED
    'modular_silhouette': 0.3,         # Silhouette above this + >2 clusters = MODULAR
    'hierarchical_density': 0.5,       # Density below this + hubs = HIERARCHICAL
}

GEOMETRY_STABILITY = {
    'breaking_severe': 1,              # Severe decouplings > this = BREAKING
    'weakening_pairs': 0.25,           # Decoupling rate above this = WEAKENING
}

GEOMETRY_LEADERSHIP = {
    'bidirectional_ratio': 0.5,        # If bidirectional > this * causal = BIDIRECTIONAL
    'contemporaneous_correlation': 0.5, # Correlation above this with no causality = CONTEMPORANEOUS
}

# Structure change detection
GEOMETRY_TRANSITION = {
    'distance_threshold': 0.3,         # Fingerprint distance for structure change
    'correlation_threshold': 0.5,      # Edge threshold for network construction
    'historical_window': 100,          # Window for baseline correlation
    'recent_window': 20,               # Window for recent correlation
}


# =============================================================================
# DYNAMICAL SYSTEMS THRESHOLDS
# =============================================================================
# Used in: prism/dynamical_systems/engine_mapping.py
# Purpose: Classify regime, stability, trajectory, attractor

DYNAMICS_REGIME = {
    'coupled': 0.7,                    # Correlation above this = COUPLED
    'decoupled': 0.3,                  # Correlation below this = DECOUPLED
    'transitioning': 0.2,              # Change rate above this = TRANSITIONING
}

DYNAMICS_STABILITY = {
    'stable_change': 0.05,             # Max change for STABLE
    'evolving_change': 0.15,           # Max change for EVOLVING
    'unstable_change': 0.30,           # Max change for UNSTABLE
    # Above 0.30 = CRITICAL
}

DYNAMICS_TRAJECTORY = {
    'convergence_trend': 0.01,         # Positive trend = CONVERGING
    'divergence_trend': -0.01,         # Negative trend = DIVERGING
    'oscillation_ratio': 0.5,          # Sign changes above this = OSCILLATING
}

# Regime transition detection
DYNAMICS_TRANSITION = {
    'distance_threshold': 0.3,         # Fingerprint distance for regime change
    'escalation_threshold': 0.25,      # Change rate to escalate to mechanics
}


# =============================================================================
# TRANSITION DETECTION THRESHOLDS (PR-aligned)
# =============================================================================
# Used in: prism/dynamical_systems/transitions.py (to be created)
# Purpose: Detect meaningful state changes between windows

TRANSITION_NUMERIC = {
    # Thresholds for "meaningful" change in numeric fields
    'stability': 0.2,                  # 20% of range
    'predictability': 0.15,
    'coupling': 0.15,
    'memory': 0.1,
}

TRANSITION_SEVERITY = {
    # Severity classification multipliers
    'mild_max': 2.0,                   # Delta > threshold but < 2x threshold
    'moderate_max': 3.0,               # Delta > 2x threshold OR sign change
    # Above 3x threshold = SEVERE
}

TRANSITION_TYPES = {
    # Conditions for transition type classification
    'bifurcation': 'stability crosses zero (stable → unstable)',
    'collapse': 'predictability or coupling drops > 2x threshold',
    'recovery': 'metrics improving after previous decline',
    'shift': 'categorical change (trajectory or attractor type)',
    'flip': 'memory crosses 0.5 (persistent ↔ anti-persistent)',
}


# =============================================================================
# STATE LAYER THRESHOLDS
# =============================================================================
# Used in: prism/engines/state/transition_detector.py
# Purpose: Detect regime transitions via divergence spikes

STATE_TRANSITION = {
    'zscore_threshold': 3.0,           # Z-score threshold for transition detection
    'stability_window': 5,             # Windows needed for full stability score
    'early_warning_ratio': 2.0,        # Gradient must be 2x baseline for warning
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_typology_thresholds(axis: str, preset: str = 'default') -> list:
    """Get thresholds for a specific axis."""
    if axis in TYPOLOGY_AXIS_THRESHOLDS:
        return TYPOLOGY_AXIS_THRESHOLDS[axis]
    return TYPOLOGY_CLASSIFICATION.get(preset, TYPOLOGY_CLASSIFICATION['default'])


def is_meaningful_change(field: str, delta: float) -> bool:
    """Check if a numeric change is meaningful."""
    threshold = TRANSITION_NUMERIC.get(field, 0.15)
    return abs(delta) > threshold


def classify_severity(field: str, delta: float) -> str:
    """Classify change severity."""
    threshold = TRANSITION_NUMERIC.get(field, 0.15)
    ratio = abs(delta) / threshold if threshold > 0 else 0

    if ratio <= 1.0:
        return 'none'
    elif ratio <= TRANSITION_SEVERITY['mild_max']:
        return 'mild'
    elif ratio <= TRANSITION_SEVERITY['moderate_max']:
        return 'moderate'
    else:
        return 'severe'


def get_transition_type(
    field: str,
    from_value: float,
    to_value: float,
) -> str:
    """Determine transition type based on field and values."""
    delta = to_value - from_value

    if field == 'stability':
        if (from_value > 0 and to_value < 0) or (from_value < 0 and to_value > 0):
            return 'bifurcation'

    if field == 'memory':
        if (from_value > 0.5 and to_value < 0.5) or (from_value < 0.5 and to_value > 0.5):
            return 'flip'

    if field in ('predictability', 'coupling'):
        if delta < -2 * TRANSITION_NUMERIC.get(field, 0.15):
            return 'collapse'
        elif delta > 2 * TRANSITION_NUMERIC.get(field, 0.15):
            return 'recovery'

    return 'shift'


# =============================================================================
# SUMMARY TABLE (for reference)
# =============================================================================
"""
Layer               | Parameter                  | Value  | Effect
--------------------|----------------------------|--------|----------------------------------
Signal Typology     | axis_threshold             | 0.30   | Change per axis for "moving"
Signal Typology     | distance_threshold         | 0.30   | Profile distance for regime change
Signal Typology     | classification[0]          | 0.25   | Score < this = strong_low
Signal Typology     | classification[3]          | 0.75   | Score > this = strong_high

Struct. Geometry    | highly_connected_density   | 0.70   | Above = HIGHLY_CONNECTED
Struct. Geometry    | modular_silhouette         | 0.30   | Above + clusters = MODULAR
Struct. Geometry    | breaking_severe            | 1      | Severe decouplings = BREAKING
Struct. Geometry    | weakening_pairs            | 0.25   | Decoupling rate = WEAKENING

Dynamical Systems   | coupled                    | 0.70   | Correlation = COUPLED
Dynamical Systems   | decoupled                  | 0.30   | Correlation = DECOUPLED
Dynamical Systems   | transitioning              | 0.20   | Change rate = TRANSITIONING
Dynamical Systems   | stable_change              | 0.05   | Below = STABLE
Dynamical Systems   | critical_change            | 0.30   | Above = CRITICAL

Transitions         | stability_threshold        | 0.20   | Meaningful stability change
Transitions         | memory_threshold           | 0.10   | Meaningful memory change
Transitions         | zscore_threshold           | 3.0    | State divergence spike
"""
