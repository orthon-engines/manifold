"""
PRISM - Pure Calculation Engine

Computes math on sensor data. Nothing else.

Usage:
    from prism import analyze

    results = analyze("sensors.csv")
    results = analyze("data.parquet", config={'entity_column': 'unit_id'})

Outputs 5 parquet files:
    - data.parquet     : Original data + characterization
    - vector.parquet   : Per-signal metrics (62 engines)
    - geometry.parquet : Pairwise relationships
    - dynamics.parquet : States, transitions, trends
    - physics.parquet  : Domain-specific calculations
"""

__version__ = "0.1.0"

from prism.analyze import analyze, AnalysisResult

__all__ = [
    "__version__",
    "analyze",
    "AnalysisResult",
]
