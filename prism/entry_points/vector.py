#!/usr/bin/env python3
"""
PRISM Vector Entry Point
========================

Computes signal-level metrics using ALL vector engines with index-based windowing.

ORTHON Canonical Spec v1.0.0:
    Window k contains all rows where:
        x₀ + k*stride ≤ index < x₀ + k*stride + window_size

    CRITICAL: window_size and stride are in INDEX UNITS (seconds, meters, cycles),
              NOT row counts.

Engines (28 total):
    Memory (4): hurst_dfa, hurst_rs, acf_decay, spectral_slope
    Information (3): permutation_entropy, sample_entropy, entropy_rate
    Frequency (2): spectral, wavelet
    Volatility (4): garch, realized_vol, bipower_variation, hilbert_amplitude
    Recurrence (1): rqa
    Typology (8): cusum, derivative_stats, distribution, rolling_volatility,
                  seasonality, stationarity, takens, trend
    Pointwise (3): derivatives, hilbert, statistical
    Momentum (1): runs_test
    Discontinuity (3): dirac, heaviside, structural

Usage:
    python -m prism.entry_points.vector
    python -m prism.entry_points.vector --force
    python -m prism.entry_points.vector --window 100.0 --stride 50.0

Output:
    data/vector.parquet - One row per (entity, signal, window)
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic
from prism.config.validator import ConfigurationError

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS
# =============================================================================

# Available engines - must be explicitly enabled in config
AVAILABLE_ENGINES = {
    # Memory engines
    'hurst_dfa': 'prism.engines.memory:compute_hurst_dfa',
    'hurst_rs': 'prism.engines.memory:compute_hurst_rs',
    'acf_decay': 'prism.engines.memory:compute_acf_decay',
    'spectral_slope': 'prism.engines.memory:compute_spectral_slope',
    # Information engines
    'permutation_entropy': 'prism.engines.information:compute_permutation_entropy',
    'sample_entropy': 'prism.engines.information:compute_sample_entropy',
    'entropy_rate': 'prism.engines.information:compute_entropy_rate',
    # Frequency engines
    'spectral': 'prism.engines.frequency:compute_spectral',
    'wavelet': 'prism.engines.frequency:compute_wavelet',
    # Volatility engines
    'garch': 'prism.engines.volatility:compute_garch',
    'realized_vol': 'prism.engines.volatility:compute_realized_vol',
    'bipower_variation': 'prism.engines.volatility:compute_bipower_variation',
    'hilbert_amplitude': 'prism.engines.volatility:compute_hilbert_amplitude',
    # Recurrence engines
    'rqa': 'prism.engines.recurrence:compute_rqa',
    # Typology engines
    'cusum': 'prism.engines.typology:compute_cusum',
    'derivative_stats': 'prism.engines.typology:compute_derivative_stats',
    'distribution': 'prism.engines.typology:compute_distribution',
    'rolling_volatility': 'prism.engines.typology:compute_rolling_volatility',
    'seasonality': 'prism.engines.typology:compute_seasonality',
    'stationarity': 'prism.engines.typology:compute_stationarity',
    'takens': 'prism.engines.typology:compute_takens',
    'trend': 'prism.engines.typology:compute_trend',
    # Pointwise engines
    'derivatives': 'prism.engines.pointwise:compute_derivatives',
    'hilbert': 'prism.engines.pointwise:compute_hilbert',
    'statistical': 'prism.engines.pointwise:compute_statistical',
    # Momentum engines
    'runs_test': 'prism.engines.momentum:compute_runs_test',
    # Discontinuity engines
    'dirac': 'prism.engines.discontinuity:compute_dirac',
    'heaviside': 'prism.engines.discontinuity:compute_heaviside',
    'structural': 'prism.engines.discontinuity:compute_structural',
    # Laplace engines
    'laplace': None,  # Special handling
    # Dynamics engines
    'hd_slope': 'prism.engines.dynamics.hd_slope:compute_hd_slope',
}


def import_engines(config: Dict[str, Any]):
    """
    Import vector engines based on EXPLICIT config selection.

    ZERO DEFAULTS POLICY: Engines must be explicitly listed in config.

    Config structure (REQUIRED):
        engines:
          vector:
            enabled:
              - hurst_dfa
              - sample_entropy
              - rqa
            params:
              rqa:
                embedding_dim: 3
                time_delay: 1
                threshold_percentile: 10.0

    Raises:
        ConfigurationError: If engines.vector.enabled not specified
    """
    engines = {}

    if 'engines' not in config or 'vector' not in config.get('engines', {}):
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: engines.vector section missing\n"
            f"{'='*60}\n\n"
            f"PRISM requires explicit engine configuration.\n"
            f"Add to config.yaml:\n\n"
            f"  engines:\n"
            f"    vector:\n"
            f"      enabled:\n"
            f"        - hurst_dfa\n"
            f"        - sample_entropy\n"
            f"        - rqa\n\n"
            f"Available engines: {list(AVAILABLE_ENGINES.keys())}\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    engine_config = config['engines']['vector']

    if 'enabled' not in engine_config:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: engines.vector.enabled not specified\n"
            f"{'='*60}\n\n"
            f"List the engines to run explicitly:\n\n"
            f"  engines:\n"
            f"    vector:\n"
            f"      enabled:\n"
            f"        - hurst_dfa\n"
            f"        - sample_entropy\n"
            f"        - rqa\n\n"
            f"Available engines: {list(AVAILABLE_ENGINES.keys())}\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    enabled_engines = engine_config['enabled']
    engine_params = engine_config.get('params', {})

    for engine_name in enabled_engines:
        if engine_name not in AVAILABLE_ENGINES:
            logger.warning(f"Unknown engine: {engine_name}")
            continue

        module_path = AVAILABLE_ENGINES[engine_name]
        if module_path is None:
            continue

        try:
            module_name, func_name = module_path.rsplit(':', 1)
            module = __import__(module_name, fromlist=[func_name])
            compute_fn = getattr(module, func_name)

            # Wrap with params if specified
            params = engine_params.get(engine_name, {})
            if params:
                from functools import partial
                engines[engine_name] = partial(compute_fn, **params)
            else:
                engines[engine_name] = compute_fn

        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load engine {engine_name}: {e}")

    logger.info(f"  Loaded {len(engines)} engines from config: {list(engines.keys())}")
    return engines


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """
    Load config from data directory.

    ORTHON Canonical Spec v1.0.0:
        window.size  - Window width in INDEX UNITS (e.g., seconds, meters, cycles)
        window.stride - Step between windows in INDEX UNITS

    CRITICAL: size and stride are in INDEX UNITS, not row counts.
        Window k contains all rows where:
            x₀ + k*stride ≤ index < x₀ + k*stride + size

    Raises:
        ConfigurationError: If window.size or window.stride not set
    """
    config_path = data_path / 'config.yaml'

    if not config_path.exists():
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: config.yaml not found\n"
            f"{'='*60}\n"
            f"Location: {config_path}\n\n"
            f"PRISM requires explicit windowing configuration.\n"
            f"Create config.yaml with:\n\n"
            f"  window:\n"
            f"    size: 50.0      # Window width in INDEX UNITS (not row count)\n"
            f"    stride: 25.0    # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10  # Minimum rows per window\n\n"
            f"CRITICAL: size/stride are in INDEX UNITS (seconds, meters, cycles)\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    config = {
        'engines': user_config.get('engines', {}),
    }

    # REQUIRED: window section (ORTHON Canonical Spec v1.0.0)
    if 'window' not in user_config:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window section missing\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"PRISM requires explicit windowing configuration.\n"
            f"Add to config.yaml:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"CRITICAL: size/stride are in INDEX UNITS (not row counts)\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    window_cfg = user_config['window']

    if window_cfg.get('size') is None:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window.size not set\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"window.size is REQUIRED (in INDEX UNITS). Example:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    if window_cfg.get('stride') is None:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window.stride not set\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"window.stride is REQUIRED (in INDEX UNITS). Example:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    # Use float for index-based windowing
    config['window_size'] = float(window_cfg['size'])
    config['stride'] = float(window_cfg['stride'])
    # Default min_observations = 10 per ORTHON spec
    config['min_observations'] = window_cfg.get('min_observations', 10)

    logger.info(f"Loaded config: window_size={config['window_size']}, stride={config['stride']}, min_observations={config['min_observations']} (INDEX UNITS)")

    return config


# =============================================================================
# WINDOWING (ORTHON Canonical Spec v1.0.0)
# =============================================================================

def generate_windows(
    values: np.ndarray,
    indices: np.ndarray,
    window_size: float,
    stride: float,
    min_observations: int,
) -> List[Dict]:
    """
    Generate overlapping windows from a signal using INDEX-BASED windowing.

    ORTHON Canonical Spec v1.0.0:
        Window k contains all rows where:
            x₀ + k*S ≤ index < x₀ + k*S + W

        Where:
            x₀ = first index value
            W  = window_size (in INDEX UNITS, not row count)
            S  = stride (in INDEX UNITS, not row count)

    Args:
        values: Signal values
        indices: Corresponding sequence indices (time, depth, cycle, etc.)
        window_size: Window width in INDEX UNITS (not row count)
        stride: Step between windows in INDEX UNITS (not row count)
        min_observations: Minimum rows required per window

    Yields:
        Dict with window_idx, window_start, window_end, values, indices
    """
    if len(values) == 0:
        return

    # Get index range
    x0 = float(indices[0])      # First index value
    x_max = float(indices[-1])  # Last index value

    # If total index span is less than window size, skip
    if (x_max - x0) < window_size:
        return

    window_idx = 0
    window_start = x0

    while window_start + window_size <= x_max + stride:  # Allow last partial window
        window_end = window_start + window_size

        # Select rows in this window: window_start ≤ index < window_end
        mask = (indices >= window_start) & (indices < window_end)
        window_values = values[mask]
        window_indices = indices[mask]

        # Only yield if we have enough observations
        if len(window_values) >= min_observations:
            yield {
                'window_idx': window_idx,
                'window_start': window_start,
                'window_end': window_end,
                'values': window_values,
                'indices': window_indices,
            }
            window_idx += 1

        window_start += stride


# =============================================================================
# RESULT FLATTENING
# =============================================================================

def flatten_result(result: Any, prefix: str) -> Dict[str, float]:
    """Flatten engine result to dict of floats."""
    flat = {}

    if result is None:
        return flat

    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif isinstance(v, np.ndarray) and v.size == 1:
                val = float(v.item())
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif hasattr(result, '__dataclass_fields__'):
        # Dataclass
        for k in result.__dataclass_fields__:
            v = getattr(result, k)
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif isinstance(result, (int, float, np.integer, np.floating)):
        val = float(result)
        if np.isfinite(val):
            flat[prefix] = val

    return flat


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_vector(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute vector metrics for all signals.

    ORTHON Canonical Spec v1.0.0:
        Windows are defined in INDEX UNITS (not row counts).
        Window k contains all rows where:
            x₀ + k*stride ≤ index < x₀ + k*stride + window_size

    Args:
        observations: Raw observations DataFrame
        config: Domain configuration (window_size, stride in INDEX UNITS)
        engines: Dict of engine_name -> compute function
        force: Recompute all

    Returns:
        DataFrame with one row per (entity, signal, window)
    """
    # All values validated in load_config
    min_observations = config['min_observations']
    window_size = config['window_size']  # INDEX UNITS
    stride = config['stride']            # INDEX UNITS

    # Determine index column (accept both 'index' and 'timestamp')
    if 'index' in observations.columns:
        index_col = 'index'
    elif 'timestamp' in observations.columns:
        index_col = 'timestamp'
    else:
        raise ValueError(f"Observations must have 'index' or 'timestamp' column. Found: {observations.columns}")

    # Group observations by entity+signal
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col(index_col).alias('indices'),
    ])

    n_signals = len(signals)
    n_engines = len(engines)
    logger.info(f"Processing {n_signals} signals with {n_engines} engines")

    results = []
    total_windows = 0

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        indices = np.array(row['indices'], dtype=float)

        # Sort by index
        sort_idx = np.argsort(indices)
        values = values[sort_idx]
        indices = indices[sort_idx]

        # Remove NaN
        valid = ~np.isnan(values)
        values = values[valid]
        indices = indices[valid]

        if len(values) < min_observations:
            continue

        # Generate windows (ORTHON Canonical Spec - index-based)
        for window in generate_windows(values, indices, window_size, stride, min_observations):
            window_values = window['values']

            row_data = {
                'entity_id': entity_id,
                'signal_id': signal_id,
                'window_idx': window['window_idx'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'n_samples': len(window_values),
            }

            # Run all engines
            for engine_name, compute_fn in engines.items():
                try:
                    result = compute_fn(window_values)
                    flat = flatten_result(result, engine_name)
                    row_data.update(flat)
                except Exception as e:
                    # Engine failed for this window, skip silently
                    pass

            results.append(row_data)
            total_windows += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals ({total_windows} windows)")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Vector: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Vector - Signal-level metrics computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')
    # Window/stride in INDEX UNITS - override config (ORTHON Canonical Spec)
    parser.add_argument('--window', type=float,
                        help='Override window.size from config (in INDEX UNITS)')
    parser.add_argument('--stride', type=float,
                        help='Override window.stride from config (in INDEX UNITS)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Vector Engine")
    logger.info("=" * 60)

    ensure_directory()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found at {obs_path}")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'vector.parquet'

    if output_path.exists() and not args.force:
        logger.info(f"vector.parquet exists, use --force to recompute")
        return 0

    # Load config - WILL FAIL LOUDLY if window/stride not set
    try:
        config = load_config(data_path)
    except ConfigurationError as e:
        logger.error(str(e))
        sys.exit(1)

    # CLI overrides (only if explicitly provided)
    if args.window is not None:
        logger.info(f"CLI override: window_size={args.window}")
        config['window_size'] = args.window
    if args.stride is not None:
        logger.info(f"CLI override: stride={args.stride}")
        config['stride'] = args.stride

    # Load engines from config
    logger.info("Loading engines from config...")
    engines = import_engines(config)
    logger.info(f"Total engines: {len(engines)}")

    # Load data
    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    # Compute
    start = time.time()
    df = compute_vector(observations, config, engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
