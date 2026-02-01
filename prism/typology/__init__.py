"""
PRISM Typology Module

Typology is the foundation. It determines:
- Window size
- Engine selection
- Derivative depth
- Everything downstream

Order of Operations:
    0. Data Validation (FIRST GATE - before anything else)
    1. Bachelor: Stationarity (ADF + KPSS)
    2. Masters: Classification (periodic/trending/chaotic/random/stationary)
    3. PhD: Confidence scoring (Chaos Decision Tree)
    4. Associate Prof: Optimal representation (wavelets)
    5. Emeritus: Cross-signal relationships
    6. Full Prof: Adaptive, self-correcting

Build order: Each level must be VALIDATED before the next is BUILT.
Bad data is ORTHON's problem to fix, not PRISM's to accommodate.
"""

# Level 0: Data Validation (FIRST GATE)
from prism.typology.data_validation import (
    validate_observations,
    validate_benchmark_file,
    ValidationResult,
    ValidationStatus,
)

# Level 1: Stationarity
from prism.typology.level1_stationarity import (
    test_stationarity,
    StationarityType,
    StationarityResult,
    Confidence,
    validate_level1_benchmarks,
)

__all__ = [
    # Level 0: Data Validation
    'validate_observations',
    'validate_benchmark_file',
    'ValidationResult',
    'ValidationStatus',
    # Level 1: Stationarity
    'test_stationarity',
    'StationarityType',
    'StationarityResult',
    'Confidence',
    'validate_level1_benchmarks',
]
