"""
Spectral engine — spectral slope, dominant freq, spectral entropy, centroid, bandwidth.

Wraps engines.manifold.signal.spectral which imports from
engines.primitives.individual.spectral (canonical).

Outputs:
    spectral_slope      - Log-log slope of PSD. Steeper = more low-freq energy.
    dominant_freq       - Frequency with most power.
    spectral_entropy    - Flatness of spectrum. 1.0 = white noise, 0.0 = pure tone.
    spectral_centroid   - Center of mass of spectrum.
    spectral_bandwidth  - Spread of spectral energy.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, **params) -> Dict[str, float]:
    """Compute spectral features from a 1D signal window.

    Delegates to engines.manifold.signal.spectral which imports from
    engines.primitives.individual.spectral (canonical).

    Args:
        y: 1D numpy array of signal values.
        **params: Optional. Pass sample_rate (float) to set sampling rate.
                  Default is 1.0 Hz.

    Returns:
        Dict with spectral feature keys.
        Values are np.nan when insufficient samples.
    """
    from engines.manifold.signal.spectral import compute as _compute_spectral

    sample_rate = params.get('sample_rate', 1.0)

    try:
        result = _compute_spectral(y, sample_rate=sample_rate)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {
        'spectral_slope': np.nan,
        'dominant_freq': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan,
    }
