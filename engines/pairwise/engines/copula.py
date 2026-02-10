"""
Copula engine -- models dependency structure beyond linear correlation.

Delegates to engines.manifold.pairwise.copula which implements:
    - Gaussian, Clayton, Gumbel, Frank copula families
    - AIC-based family selection
    - Tail dependence coefficients (lower and upper)
    - Kendall tau and Spearman rho
"""

import numpy as np
from typing import Dict, Any, Optional, List


def compute(x: np.ndarray, y: np.ndarray, **params) -> Dict[str, Any]:
    """
    Fit copula models to two vectors and extract dependency structure.

    Args:
        x, y: Input vectors (1D arrays).
        **params:
            families: list -- Copula families to test
                      (default: ["gaussian", "clayton", "gumbel", "frank"]).

    Returns:
        Dict with:
            best_family: str (winning copula)
            best_aic: float
            best_param: float (copula parameter theta)
            kendall_tau: float
            spearman_rho: float
            lower_tail_dependence: float (lambda_L)
            upper_tail_dependence: float (lambda_U)
            tail_asymmetry: float (lambda_U - lambda_L)
            tail_dependence_ratio: float
            gaussian_param, gaussian_aic: float
            clayton_param, clayton_aic: float
            gumbel_param, gumbel_aic: float
            frank_param, frank_aic: float
            empirical_lower_tail: float
            empirical_upper_tail: float
            n_samples: int
    """
    from engines.manifold.pairwise.copula import compute as _compute

    families = params.get('families', None)

    return _compute(x, y, families=families)
