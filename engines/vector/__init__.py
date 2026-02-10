"""
Vector operation — extract per-signal features from time series windows.

This module is a clean facade over the existing signal engine infrastructure.
It groups engines by concern (shape, complexity, spectral, harmonic) and
provides a unified interface for feature extraction.

Usage:
    from engines.vector import run, compute_vector

    # Full pipeline step
    run(observations_path, output_path, manifest)

    # Single window
    features = compute_vector(y, engines=['statistics', 'spectral'])
"""

from engines.vector.run import run, compute_vector

__all__ = ['run', 'compute_vector']
