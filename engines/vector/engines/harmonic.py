"""
Harmonic engines — harmonics, fundamental_freq, thd.

Wraps engines.manifold.signal.harmonics, fundamental_freq, and thd.
All three exist in the signal engine directory.

Outputs from harmonics:
    fundamental_freq       - Auto-detected fundamental frequency.
    fundamental_amplitude  - Amplitude at the fundamental.
    harmonic_2x            - Amplitude at 2x fundamental.
    harmonic_3x            - Amplitude at 3x fundamental.
    thd                    - Total harmonic distortion (%).

Outputs from fundamental_freq:
    fundamental_freq       - Lowest dominant frequency.
    fundamental_power      - Power at fundamental.
    fundamental_ratio      - Fraction of total power at fundamental.
    fundamental_confidence - Peak prominence confidence [0,1].

Outputs from thd:
    thd_percent            - THD as percentage.
    thd_db                 - THD in decibels.
    thd_fundamental_power  - Power at fundamental.
    thd_harmonic_power     - Summed harmonic power.
    n_harmonics_found      - Number of harmonics detected.

Note: harmonics.compute and fundamental_freq.compute both output a
'fundamental_freq' key. When both are run, the fundamental_freq engine
result overwrites the harmonics engine result for that key. This is
intentional — fundamental_freq is the more specialized detector.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, **params) -> Dict[str, Any]:
    """Compute harmonic features from a 1D signal window.

    Delegates to engines.manifold.signal.harmonics,
    engines.manifold.signal.fundamental_freq, and
    engines.manifold.signal.thd (all canonical).

    Args:
        y: 1D numpy array of signal values.
        **params: Optional. Pass sample_rate (float) for harmonics engine.
                  Default is 1.0 Hz.

    Returns:
        Dict with harmonic analysis features.
        Values are np.nan when insufficient samples.
    """
    from engines.manifold.signal.harmonics import compute as _compute_harmonics
    from engines.manifold.signal.fundamental_freq import compute as _compute_fundamental
    from engines.manifold.signal.thd import compute as _compute_thd

    sample_rate = params.get('sample_rate', 1.0)
    results = {}

    # Harmonics (fundamental_freq, fundamental_amplitude, harmonic_2x, harmonic_3x, thd)
    try:
        r = _compute_harmonics(y, sample_rate=sample_rate)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        pass

    # Fundamental frequency detection (more specialized than harmonics)
    try:
        r = _compute_fundamental(y)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        pass

    # THD (total harmonic distortion — detailed)
    try:
        r = _compute_thd(y)
        if isinstance(r, dict):
            results.update(r)
    except Exception:
        pass

    return results
