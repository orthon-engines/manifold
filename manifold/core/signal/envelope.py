"""
Envelope Engine.

Imports from primitives/individual/hilbert.py (canonical).
Primitives handle validation - no redundant checks here.
"""

import warnings

import numpy as np
from manifold.primitives.individual.hilbert import envelope
from manifold.primitives.individual.statistics import kurtosis, rms


def compute(y: np.ndarray) -> dict:
    """
    Compute Hilbert envelope metrics of signal.

    Args:
        y: Signal values

    Returns:
        dict with envelope_rms, envelope_peak, envelope_kurtosis
    """
    try:
        env = envelope(y)

        return {
            'envelope_rms': rms(env),
            'envelope_peak': float(np.max(env)),
            'envelope_kurtosis': kurtosis(env, fisher=True)
        }
    except ValueError:
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }
    except Exception as e:
        warnings.warn(f"envelope.compute: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }
