"""
Fingerprint operation module -- Gaussian fingerprints + pairwise similarity.

This module is scale-agnostic. It takes a set of entity vectors over time,
computes per-entity Gaussian fingerprints (mean, std, volatility), and then
computes pairwise Bhattacharyya similarity between all entity pairs.

It does NOT know whether entities are signals, cohorts, or anything else.
The same math applies at every scale.

Entry points:
    run()                -- signal-scale fingerprinting (delegates to stage_24)
    run_system()         -- cohort-scale fingerprinting (delegates to stage_32)

Compute engines:
    engines.gaussian     -- per-entity Gaussian fingerprint (mean, std, volatility)
    engines.similarity   -- pairwise Bhattacharyya distance between fingerprints
"""

from engines.fingerprint.run import run, run_system
from engines.fingerprint.engines.gaussian import compute as compute_fingerprint
from engines.fingerprint.engines.similarity import compute as compute_similarity

__all__ = ['run', 'run_system', 'compute_fingerprint', 'compute_similarity']
