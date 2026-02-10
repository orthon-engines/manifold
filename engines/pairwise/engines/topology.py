"""
Topology engine -- graph metrics from a pairwise matrix.

Delegates to engines.entry_points.stage_11_topology.compute_basic_topology
which computes:
    - Correlation-based adjacency (thresholded)
    - Network density, mean/max degree, edge count

This engine accepts a signal matrix (N_samples x D_signals) and
computes topology of the inter-signal correlation graph.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(signal_matrix: np.ndarray, **params) -> Dict[str, Any]:
    """
    Compute graph topology from a signal matrix.

    Unlike the other pairwise engines (which take two vectors), this
    engine takes an entire signal matrix and computes topology of
    the pairwise correlation graph.

    Args:
        signal_matrix: N_samples x D_signals matrix.
        **params:
            threshold: float or None -- Correlation threshold for adjacency.
                       If None, uses 90th percentile of abs(correlation).

    Returns:
        Dict with:
            topology_computed: bool
            n_signals: int
            n_edges: int
            density: float
            mean_degree: float
            max_degree: int
            threshold: float
    """
    from engines.entry_points.stage_11_topology import compute_basic_topology

    threshold = params.get('threshold', None)

    return compute_basic_topology(signal_matrix, threshold=threshold)


def compute_from_pair_vectors(
    x: np.ndarray, y: np.ndarray, **params
) -> Dict[str, Any]:
    """
    Compute topology from two vectors stacked as a 2-signal system.

    This is a convenience wrapper for the standard compute(x, y) interface.
    With only 2 signals, topology is trivial (one edge or none), but this
    keeps the interface consistent.

    Args:
        x, y: Input vectors (1D arrays, same length).
        **params:
            threshold: float or None

    Returns:
        Dict with topology metrics for the 2-signal graph.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Stack into N x 2 matrix
    signal_matrix = np.column_stack([x, y])

    return compute(signal_matrix, **params)
