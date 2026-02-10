"""
Pipeline module — orchestrates operations at different scales.

Two pipelines:
- signal_pipeline: Scale 1 — raw signals → signal_vector → pairwise → state_geometry
- cohort_pipeline: Scale 2 — state_geometry → cohort_vector → pairwise → system_geometry

The engines do math. The pipelines decide what math to do on what data.
"""

from engines.pipeline.signal_pipeline import run as run_signal_pipeline
from engines.pipeline.cohort_pipeline import run as run_cohort_pipeline
from engines.pipeline.manifest import load_manifest, validate_manifest

__all__ = [
    'run_signal_pipeline',
    'run_cohort_pipeline',
    'load_manifest',
    'validate_manifest',
]
