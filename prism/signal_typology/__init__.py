"""
Signal Typology
===============

One of the four ORTHON analytical frameworks:

    Signal Typology     → What IS this signal? (this framework)
    Structural Geometry → What is its STRUCTURE?
    Dynamical Systems   → How does the SYSTEM evolve?
    Causal Mechanics    → What DRIVES the system?

The Nine Axes (0-1 normalized):
    - Memory        - Forgetful (0) → Persistent (1)
    - Information   - Predictable (0) → Entropic (1)
    - Frequency     - Aperiodic (0) → Periodic (1)
    - Volatility    - Stable (0) → Clustered (1)
    - Wavelet       - Single-scale (0) → Multi-scale (1)
    - Derivatives   - Smooth (0) → Spiky (1)
    - Recurrence    - Wandering (0) → Returning (1)
    - Discontinuity - Continuous (0) → Step-like (1)
    - Momentum      - Reverting (0) → Trending (1)

Principle: Data = math. Labels = rendering.
    - Store normalized 0-1 scores
    - Classification computed at query/display time
    - Threshold changes don't require recomputation

Outputs:
    - signal_typology_metrics.parquet - Raw engine measurements
    - signal_typology_profile.parquet - Normalized 0-1 axis scores

Usage:
    >>> from prism.signal_typology import analyze_single, classify_profile
    >>>
    >>> # Analyze a signal
    >>> result = analyze_single(my_array, signal_id='sensor_1')
    >>> print(result['profile'])      # 0-1 scores
    >>> print(result['classification'])  # labels (computed at runtime)
    >>> print(result['summary'])      # one-line summary
    >>>
    >>> # Classify any profile at runtime
    >>> labels = classify_profile({'memory': 0.73, 'volatility': 0.45, ...})

Architecture:
    signal_typology/
        __init__.py         # This file
        orchestrator.py     # Computes metrics → profile
        normalize.py        # Engine outputs → 0-1 scores
        classify.py         # 0-1 scores → labels (runtime)
        engine_mapping.py   # Engine selection
"""

__version__ = "3.0.0"
__author__ = "Ørthon Project"

# Orchestrator (main API)
from .orchestrator import (
    run_signal_typology,
    analyze_single,
    compute_metrics,
    get_fingerprint,
    fingerprint_distance,
    detect_regime_change,
)

# Normalization
from .normalize import (
    normalize_all,
    metrics_to_profile,
    AXIS_NAMES,
    NORMALIZERS,
)

# Classification (runtime)
from .classify import (
    classify,
    classify_profile,
    summarize_profile,
    get_dominant_characteristics,
    format_profile_table,
    AXIS_LABELS,
    AXIS_POLES,
    AXIS_DESCRIPTIONS,
    DEFAULT_THRESHOLDS,
    THRESHOLD_PRESETS,
)

# Engine mapping (if available)
try:
    from .engine_mapping import (
        select_engines,
        get_primary_classification,
        ENGINE_MAP,
    )
except ImportError:
    select_engines = None
    get_primary_classification = None
    ENGINE_MAP = {}

# Paragraph Engine (dynamic narrative generation)
from .paragraph_engine import (
    ParagraphEngine,
    TemplateEngine,
    Classification,
    classify_value,
    classify_signal,
    generate_signal_report,
    generate_prism_report,
    metrics_to_paragraph_context,
    render_template,
    THRESHOLDS,
    SIGNAL_TYPES,
    TEMPLATES,
)

__all__ = [
    # Version
    "__version__",

    # Orchestrator API
    "run_signal_typology",
    "analyze_single",
    "compute_metrics",
    "get_fingerprint",
    "fingerprint_distance",
    "detect_regime_change",

    # Normalization
    "normalize_all",
    "metrics_to_profile",
    "AXIS_NAMES",
    "NORMALIZERS",

    # Classification
    "classify",
    "classify_profile",
    "summarize_profile",
    "get_dominant_characteristics",
    "format_profile_table",
    "AXIS_LABELS",
    "AXIS_POLES",
    "AXIS_DESCRIPTIONS",
    "DEFAULT_THRESHOLDS",
    "THRESHOLD_PRESETS",

    # Engine mapping
    "select_engines",
    "get_primary_classification",
    "ENGINE_MAP",

    # Paragraph Engine (dynamic narrative generation)
    "ParagraphEngine",
    "TemplateEngine",
    "Classification",
    "classify_value",
    "classify_signal",
    "generate_signal_report",
    "generate_prism_report",
    "metrics_to_paragraph_context",
    "render_template",
    "THRESHOLDS",
    "SIGNAL_TYPES",
    "TEMPLATES",
]
