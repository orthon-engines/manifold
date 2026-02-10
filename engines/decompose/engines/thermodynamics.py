"""
Thermodynamic quantities from eigenvalue spectra over time.

Maps dynamical-systems geometry onto statistical-mechanics analogues:
    S  ~ effective_dim   (entropy proxy -- how many dimensions are active)
    E  ~ total_variance  (energy proxy  -- total spread of the system)
    T  = dS / dI         (temperature   -- rate of dimensional change)
    F  = E - T * S       (free energy   -- constrained energy)
    C  = dE / dT         (heat capacity -- energy response to temperature change)

Scale-agnostic: operates on pre-computed time-series arrays.
"""

import numpy as np
from typing import Dict


def compute(
    effective_dims: np.ndarray,
    total_variances: np.ndarray,
    indices: np.ndarray,
    **params,
) -> Dict[str, np.ndarray]:
    """Compute thermodynamic quantities from geometry time series. Scale-agnostic.

    Args:
        effective_dims:  (n,) entropy proxy (S) at each index.
        total_variances: (n,) energy proxy (E) at each index.
        indices:         (n,) window indices (I).
        **params: Reserved for future use.

    Returns:
        Dict with (n,)-length arrays:
            temperature   -- dS/dI
            free_energy   -- E - T*S
            heat_capacity -- dE/dT
    """
    effective_dims = np.asarray(effective_dims, dtype=float)
    total_variances = np.asarray(total_variances, dtype=float)
    indices = np.asarray(indices, dtype=float)

    n = len(indices)
    if n < 3:
        return {
            'temperature': np.full(n, np.nan),
            'free_energy': np.full(n, np.nan),
            'heat_capacity': np.full(n, np.nan),
        }

    # Temperature: rate of change of entropy proxy with respect to index
    temperature = np.gradient(effective_dims, indices)

    # Free energy: E - T * S
    free_energy = total_variances - temperature * effective_dims

    # Heat capacity: dE / dT  (guard against constant temperature)
    if np.any(temperature != 0):
        heat_capacity = np.gradient(total_variances, temperature)
    else:
        heat_capacity = np.full(n, np.nan)

    return {
        'temperature': temperature,
        'free_energy': free_energy,
        'heat_capacity': heat_capacity,
    }
