"""
Cointegration engine -- tests whether two time series share a long-run equilibrium.

Delegates to engines.manifold.pairwise.cointegration which implements:
    - Engle-Granger two-step procedure (OLS + ADF on residuals)
    - Hedge ratio, half-life, spread z-score
"""

import numpy as np
from typing import Dict, Any


def compute(x: np.ndarray, y: np.ndarray, **params) -> Dict[str, Any]:
    """
    Test for cointegration between two vectors.

    Args:
        x, y: Input vectors (1D arrays, time series).
        **params:
            max_lag: int -- Maximum lag for ADF test (default 4).
            significance: float -- Significance level (default 0.05).

    Returns:
        Dict with:
            is_cointegrated: bool
            adf_statistic: ADF test statistic on residuals
            adf_pvalue: p-value of ADF test
            hedge_ratio: beta from OLS (y1 = alpha + beta*y2)
            intercept: alpha from OLS
            residual_std: std of equilibrium residuals
            half_life: mean-reversion half-life in samples
            residual_adf_critical_1pct: float
            residual_adf_critical_5pct: float
            residual_adf_critical_10pct: float
            spread_current: current residual value
            spread_zscore: current residual in std units
            n_samples: int
    """
    from engines.manifold.pairwise.cointegration import compute as _compute

    max_lag = params.get('max_lag', 4)
    significance = params.get('significance', 0.05)

    return _compute(x, y, max_lag=max_lag, significance=significance)
