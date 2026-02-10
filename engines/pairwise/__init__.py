"""
Pairwise operation module.

Computes relationship metrics between ANY two vectors.
Scale-agnostic: works at signal, cohort, or system level.

This module delegates all computation to existing engines:
    - engines.manifold.pairwise.correlation   (Pearson, Spearman, cross-correlation, MI)
    - engines.manifold.pairwise.causality     (Granger, transfer entropy)
    - engines.manifold.pairwise.cointegration (Engle-Granger, ADF)
    - engines.manifold.pairwise.copula        (Gaussian, Clayton, Gumbel, Frank)
    - engines.primitives.pairwise.distance    (Euclidean, DTW, cosine)
    - engines.primitives.information.divergence (KL, JS)
    - engines.entry_points.stage_11_topology  (graph topology)
"""

from .run import run, compute_pairwise

__all__ = ['run', 'compute_pairwise']
