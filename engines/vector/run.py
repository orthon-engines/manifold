"""
Vector operation — extract features from signal windows.

Takes raw signal windows. Runs all specified engines.
Returns one row per entity per window.
Entity = signal (Scale 1) or cohort (Scale 2).

This module does NOT know what scale it's operating at.
It delegates to the existing signal vector infrastructure in
engines.entry_points.stage_01_signal_vector for the full pipeline step,
and provides a lightweight compute_vector() for single-window use.

Architecture:
    engines/vector/engines/shape.py       -> kurtosis, skewness, crest_factor
    engines/vector/engines/complexity.py  -> sample/perm entropy, hurst, acf
    engines/vector/engines/spectral.py    -> spectral slope, dominant freq, etc.
    engines/vector/engines/harmonic.py    -> harmonics, fundamental freq, THD
"""

import numpy as np
from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# Engine group registry — maps group names to their vector engine modules.
# Individual engine names are resolved via the stage_01 legacy registry.
# ---------------------------------------------------------------------------

_ENGINE_GROUPS = {
    'shape': 'engines.vector.engines.shape',
    'complexity': 'engines.vector.engines.complexity',
    'spectral': 'engines.vector.engines.spectral',
    'harmonic': 'engines.vector.engines.harmonic',
}


def _resolve_engine_func(engine_name: str):
    """
    Resolve an engine name to a compute function.

    Tries vector engine groups first, then falls back to the
    stage_01 legacy/registry lookup.

    Args:
        engine_name: Engine name (e.g. 'spectral', 'kurtosis', 'shape')

    Returns:
        Callable that accepts (y, **params) -> dict
    """
    # Check if it's a vector engine group name
    if engine_name in _ENGINE_GROUPS:
        import importlib
        module = importlib.import_module(_ENGINE_GROUPS[engine_name])
        return module.compute

    # Fall back to the stage_01 engine resolution (legacy + registry)
    from engines.entry_points.stage_01_signal_vector import get_engine_compute_func
    func = get_engine_compute_func(engine_name)
    if func is not None:
        return func

    return None


def compute_vector(
    y: np.ndarray,
    engines: Optional[List[str]] = None,
    **params,
) -> Dict[str, Any]:
    """
    Run specified engines on a single signal window.

    This is the lightweight, single-window interface. For full pipeline
    processing (windowing, parallelism, parquet I/O), use run().

    Args:
        y: 1D numpy array of signal values for one window.
        engines: List of engine names to run. If None, runs all four
                 vector engine groups (shape, complexity, spectral, harmonic).
        **params: Passed through to each engine (e.g. sample_rate=1.0).

    Returns:
        Dict mapping feature names to computed values.
        Failed engines contribute no keys (silent skip).

    Example:
        >>> import numpy as np
        >>> from engines.vector import compute_vector
        >>> y = np.random.randn(256)
        >>> features = compute_vector(y, engines=['shape', 'spectral'])
        >>> features.keys()
        dict_keys(['kurtosis', 'skewness', 'crest_factor', 'spectral_slope', ...])
    """
    if engines is None:
        engines = list(_ENGINE_GROUPS.keys())

    y = np.asarray(y).flatten()

    results = {}
    for engine_name in engines:
        func = _resolve_engine_func(engine_name)
        if func is None:
            continue
        try:
            r = func(y, **params)
            if isinstance(r, dict):
                results.update(r)
        except TypeError:
            # Engine doesn't accept **params — call without
            try:
                r = func(y)
                if isinstance(r, dict):
                    results.update(r)
            except Exception:
                pass
        except Exception:
            pass

    return results


def run(
    observations_path: str,
    output_path: str,
    manifest: Dict[str, Any],
    **kwargs,
):
    """
    Run the full vector pipeline step.

    This is a facade over engines.entry_points.stage_01_signal_vector.run().
    It processes all signals in the observations file according to the
    manifest, applying per-engine windowing, parallel dispatch, and
    parquet output.

    Args:
        observations_path: Path to observations.parquet.
        output_path: Path to write signal_vector.parquet.
        manifest: Manifest dict from ORTHON.
        **kwargs: Passed through to stage_01 run() (verbose, typology_path, etc.)

    Returns:
        polars.DataFrame with computed features per signal per window.
    """
    from engines.entry_points.stage_01_signal_vector import run as _run_stage01
    return _run_stage01(observations_path, output_path, manifest, **kwargs)
