"""
Information engines -- Granger causality, transfer entropy, mutual info, KL, JS.

Delegates to:
    - engines.manifold.pairwise.causality       (Granger, transfer entropy, bidirectional)
    - engines.primitives.information.divergence  (KL divergence, JS divergence)
    - engines.primitives.information.transfer    (transfer_entropy, net_transfer_entropy)
    - engines.primitives.pairwise.information    (mutual_information)
"""

import numpy as np
from typing import Dict, Any


def compute(x: np.ndarray, y: np.ndarray, **params) -> Dict[str, Any]:
    """
    Compute all information-theoretic and causal measures between two vectors.

    Args:
        x, y: Input vectors (1D arrays). For directional metrics, x=source, y=target.
        **params:
            max_lag: int -- Maximum lag for Granger (default 5).
            te_lag: int -- Lag for transfer entropy (default 1).
            n_bins: int -- Bins for discretization (default 10).

    Returns:
        Dict with:
            granger_f: Granger F-statistic (x -> y)
            granger_p: Granger p-value (x -> y)
            optimal_lag: Optimal Granger lag
            transfer_entropy: TE(x -> y)
            normalized_te: Normalized TE(x -> y)
            granger_f_ab: Granger F (x -> y)   [bidirectional]
            granger_p_ab: Granger p (x -> y)    [bidirectional]
            granger_f_ba: Granger F (y -> x)    [bidirectional]
            granger_p_ba: Granger p (y -> x)    [bidirectional]
            net_causality: F(x->y) - F(y->x)
            kl_divergence_xy: KL(x || y)
            kl_divergence_yx: KL(y || x)
            js_divergence: JS(x, y) -- symmetric
    """
    from engines.manifold.pairwise.causality import (
        compute_all as _causality_all,
        compute_bidirectional as _bidirectional,
    )
    from engines.primitives.information.divergence import (
        kl_divergence as _kl,
        js_divergence as _js,
    )

    max_lag = params.get('max_lag', 5)
    te_lag = params.get('te_lag', 1)

    results = {}

    # Directional causality: x -> y
    causality = _causality_all(x, y, max_lag=max_lag, te_lag=te_lag)
    results.update(causality)

    # Bidirectional causality
    bidir = _bidirectional(x, y, max_lag=max_lag)
    results['granger_f_ab'] = bidir['granger_f_ab']
    results['granger_p_ab'] = bidir['granger_p_ab']
    results['granger_f_ba'] = bidir['granger_f_ba']
    results['granger_p_ba'] = bidir['granger_p_ba']
    results['net_causality'] = bidir['net_causality']

    # Divergences
    try:
        results['kl_divergence_xy'] = float(_kl(x, y))
        results['kl_divergence_yx'] = float(_kl(y, x))
        results['js_divergence'] = float(_js(x, y))
    except Exception:
        results['kl_divergence_xy'] = float('nan')
        results['kl_divergence_yx'] = float('nan')
        results['js_divergence'] = float('nan')

    return results


def compute_granger(
    source: np.ndarray, target: np.ndarray, max_lag: int = 5
) -> Dict[str, float]:
    """Compute Granger causality (source -> target)."""
    from engines.manifold.pairwise.causality import compute_granger as _granger
    return _granger(source, target, max_lag=max_lag)


def compute_transfer_entropy(
    source: np.ndarray, target: np.ndarray, lag: int = 1, n_bins: int = 8
) -> Dict[str, float]:
    """Compute transfer entropy (source -> target)."""
    from engines.manifold.pairwise.causality import compute_transfer_entropy as _te
    return _te(source, target, lag=lag, n_bins=n_bins)


def compute_divergence(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute KL and JS divergence between two distributions/signals."""
    from engines.primitives.information.divergence import (
        kl_divergence as _kl,
        js_divergence as _js,
    )
    return {
        'kl_divergence_xy': float(_kl(x, y)),
        'kl_divergence_yx': float(_kl(y, x)),
        'js_divergence': float(_js(x, y)),
    }
