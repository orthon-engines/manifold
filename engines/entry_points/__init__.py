"""
ENGINES Entry Points - Orchestration Layer
=========================================

Entry points are pure orchestration: read parquet → call engines → write parquet.
No computation logic lives here - that belongs in engines or primitives.

Stage Order (canonical dependency):
    stage_00_breaks             → breaks.parquet (runs first)
    stage_01_signal_vector      → signal_vector.parquet
    stage_02_state_vector       → state_vector.parquet
    stage_03_state_geometry     → state_geometry.parquet
    stage_04_cohorts            → cohorts.parquet
    stage_05_signal_geometry    → signal_geometry.parquet
    stage_06_signal_pairwise    → signal_pairwise.parquet
    stage_07_geometry_dynamics  → geometry_dynamics.parquet
    stage_08_lyapunov           → lyapunov.parquet
    stage_09_dynamics           → dynamics.parquet
    stage_10_information_flow   → information_flow.parquet
    stage_11_topology           → topology.parquet
    stage_12_zscore             → zscore.parquet
    stage_13_statistics         → statistics.parquet
    stage_14_correlation        → correlation.parquet

Substages:
    stage_09a_cohort_thermodynamics → cohort_thermodynamics.parquet

Usage:
    python -m engines.entry_points.run_pipeline manifest.yaml
    python -m engines.entry_points.stage_01_signal_vector observations.parquet manifest.yaml
"""

# Lazy imports to avoid circular dependencies
# Individual stages are imported on demand by run_pipeline.py

__all__ = [
    'run_pipeline',
    'stage_00_breaks',
    'stage_01_signal_vector',
    'stage_02_state_vector',
    'stage_03_state_geometry',
    'stage_04_cohorts',
    'stage_05_signal_geometry',
    'stage_06_signal_pairwise',
    'stage_07_geometry_dynamics',
    'stage_08_lyapunov',
    'stage_09_dynamics',
    'stage_09a_cohort_thermodynamics',
    'stage_10_information_flow',
    'stage_11_topology',
    'stage_12_zscore',
    'stage_13_statistics',
    'stage_14_correlation',
]
