"""
Decompose operation module -- eigendecomposition of ANY feature matrix.

This module is scale-agnostic. It takes a matrix of (N entities x M features)
and performs eigendecomposition. It does NOT know whether entities are signals,
cohorts, or anything else.

Entry points:
    run()         -- signal-scale decomposition (delegates to stage_03)
    run_system()  -- cohort-scale decomposition (delegates to stage_26)

Compute engines:
    engines.eigen          -- full eigendecomposition via SVD
    engines.effective_dim  -- effective dimensionality from eigenvalue spectrum
    engines.condition      -- condition number, spectral gap, eigenvalue ratios
    engines.thermodynamics -- temperature, free energy, heat capacity from spectra
"""

from engines.decompose.run import run, run_system
from engines.decompose.engines.eigen import compute as compute_decomposition

__all__ = ['run', 'run_system', 'compute_decomposition']
