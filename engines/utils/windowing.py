"""
Window/stride logic driven by manifest and typology.

Centralizes all windowing decisions so engines never compute window boundaries.
"""

import numpy as np
from typing import List, Tuple


def generate_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
) -> List[Tuple[int, np.ndarray]]:
    """Generate (start_index, window_data) tuples from a 1D array.

    Args:
        data: 1D numpy array of signal values (sorted by I)
        window_size: samples per window
        stride: samples between window starts

    Returns:
        List of (start_I, window_array) tuples
    """
    n = len(data)
    if n < window_size:
        return []

    windows = []
    for start in range(0, n - window_size + 1, stride):
        window = data[start:start + window_size]
        windows.append((start, window))

    return windows


def window_count(n_samples: int, window_size: int, stride: int) -> int:
    """Compute number of windows for given parameters.

    Args:
        n_samples: total samples
        window_size: samples per window
        stride: samples between window starts

    Returns:
        Number of complete windows (0 if insufficient data)
    """
    if n_samples < window_size:
        return 0
    return (n_samples - window_size) // stride + 1


def effective_window(
    base_window: int,
    window_factor: float = 1.0,
    min_window: int = 4,
    max_window: int = 1024,
) -> int:
    """Compute effective window size for an engine.

    Args:
        base_window: engine's base window from config
        window_factor: per-signal scaling from typology
        min_window: hard minimum
        max_window: hard maximum

    Returns:
        Clamped effective window size
    """
    w = int(round(base_window * window_factor))
    return max(min_window, min(max_window, w))
