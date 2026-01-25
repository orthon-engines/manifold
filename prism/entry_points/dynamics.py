#!/usr/bin/env python3
"""
PRISM Dynamics Entry Point
==========================

Answers: HOW is it moving through behavioral space?

REQUIRES: vector.parquet AND geometry.parquet

Orchestrates dynamics engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/dynamics/.

Engines (9 total):
    - hd_slope: Degradation rate (velocity of coherence loss)
    - lyapunov: Largest Lyapunov exponent (chaos measure)
    - embedding: Embedding dimension estimation
    - phase_space: Phase space reconstruction
    - attractor: Takens embedding, correlation dimension
    - phase_position: Track position on reconstructed attractor
    - basin: Basin membership and transition analysis
    - regime: Regime detection
    - transitions: Transition detection

Output:
    data/dynamics.parquet - ONE ROW PER ENTITY

Usage:
    python -m prism.entry_points.dynamics
    python -m prism.entry_points.dynamics --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from prism.core.dependencies import check_dependencies
from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS - All compute lives in engines
# =============================================================================

from prism.engines.dynamics import (
    compute_hd_slope,
    compute_lyapunov,
    compute_embedding,
    compute_phase_space,
    compute_attractor,
    compute_phase_position,
    compute_basin,
    compute_regime,
    compute_transitions,
)

# Engine registry
ENGINES = {
    'hd_slope': compute_hd_slope,
    'lyapunov': compute_lyapunov,
    'embedding': compute_embedding,
    'phase_space': compute_phase_space,
    'attractor': compute_attractor,
    'phase_position': compute_phase_position,
    'basin': compute_basin,
    'regime': compute_regime,
    'transitions': compute_transitions,
}


# =============================================================================
# CONFIG
# =============================================================================

from prism.config.validator import ConfigurationError


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'

    if not config_path.exists():
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: config.yaml not found\n"
            f"{'='*60}\n"
            f"Location: {config_path}\n\n"
            f"PRISM requires explicit configuration.\n"
            f"Create config.yaml with dynamics settings.\n"
            f"{'='*60}"
        )

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    return user_config.get('dynamics', {})


# =============================================================================
# BEHAVIORAL MATRIX CONSTRUCTION
# =============================================================================

def build_feature_matrix(
    vector_df: pl.DataFrame,
    entity_id: str,
) -> tuple:
    """
    Build feature matrix for one entity.

    Returns:
        (numpy array, timestamps array)
    """
    entity_data = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_data) == 0:
        return None, None

    # Identify metric columns
    id_cols = {'entity_id', 'signal_id', 'window_idx', 'window_start', 'window_end', 'n_samples'}
    metric_cols = [c for c in entity_data.columns
                   if c not in id_cols
                   and entity_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not metric_cols:
        return None, None

    # Build feature matrix
    feature_matrix = entity_data.select(metric_cols).to_numpy()
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Timestamps (use window_start if available)
    if 'window_start' in entity_data.columns:
        timestamps = entity_data['window_start'].to_numpy()
    else:
        timestamps = np.arange(len(entity_data), dtype=float)

    return feature_matrix, timestamps


def extract_geometry_info(geometry_df: pl.DataFrame, entity_id: str) -> Dict[str, Any]:
    """Extract relevant geometry info for dynamics calculations."""
    entity_geom = geometry_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_geom) == 0:
        return {}

    info = {}

    # Extract covariance inverse for Mahalanobis distance (if available)
    row = entity_geom.row(0, named=True)

    # Look for precision matrix or covariance info
    for key in row:
        if 'precision' in key or 'covariance' in key or 'pca' in key:
            val = row[key]
            if isinstance(val, (int, float)) and np.isfinite(val):
                info[key] = val
            elif isinstance(val, str) and val.startswith('['):
                try:
                    info[key] = np.array(json.loads(val))
                except:
                    pass

    return info


# =============================================================================
# ORCHESTRATOR - Routes to engines, no compute
# =============================================================================

def run_dynamics_engines(
    feature_matrix: np.ndarray,
    timestamps: np.ndarray,
    geometry_info: Dict[str, Any],
    entity_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all dynamics engines.

    Pure orchestration - all compute is in engines.
    """
    results = {'entity_id': entity_id}

    # Get engine config
    enabled_engines = config.get('enabled', list(ENGINES.keys()))
    engine_params = config.get('params', {})

    for engine_name, compute_fn in ENGINES.items():
        if engine_name not in enabled_engines:
            continue

        try:
            params = engine_params.get(engine_name, {})

            # Route to engine with appropriate arguments
            if engine_name == 'hd_slope':
                # hd_slope needs feature matrix and timestamps
                result = compute_fn(
                    feature_matrix=feature_matrix,
                    timestamps=timestamps,
                    **params
                )
            elif engine_name in ['lyapunov', 'embedding', 'phase_space']:
                # These work on 1D series - use first column (no inline compute)
                series = feature_matrix[:, 0] if feature_matrix.ndim > 1 else feature_matrix
                result = compute_fn(series, **params)
            elif engine_name == 'attractor':
                # Attractor reconstruction - use first column
                series = feature_matrix[:, 0] if feature_matrix.ndim > 1 else feature_matrix
                result = compute_fn(series, **params)
            elif engine_name == 'phase_position':
                # Phase position tracking - use first column
                series = feature_matrix[:, 0] if feature_matrix.ndim > 1 else feature_matrix
                result = compute_fn(series, **params)
            elif engine_name == 'basin':
                # Basin analysis
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'regime':
                # Regime detection
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'transitions':
                # Transition detection
                result = compute_fn(feature_matrix, **params)
            else:
                result = compute_fn(feature_matrix, **params)

            # Flatten result into row
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if np.isfinite(v):
                            results[f"{engine_name}_{k}"] = float(v)
                    elif isinstance(v, str):
                        results[f"{engine_name}_{k}"] = v
                    elif isinstance(v, np.ndarray) and v.size <= 50:
                        results[f"{engine_name}_{k}_json"] = json.dumps(v.tolist())
            elif isinstance(result, (int, float, np.integer, np.floating)):
                if np.isfinite(result):
                    results[engine_name] = float(result)

        except Exception as e:
            logger.debug(f"{engine_name} failed for {entity_id}: {e}")

    return results


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_dynamics(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute dynamics for all entities.

    Output: ONE ROW PER ENTITY
    """
    entities = vector_df.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)

    logger.info(f"Computing dynamics for {n_entities} entities")
    logger.info(f"Engines: {list(ENGINES.keys())}")

    results = []

    for i, entity_id in enumerate(entities):
        # Build feature matrix
        feature_matrix, timestamps = build_feature_matrix(vector_df, entity_id)

        if feature_matrix is None or len(feature_matrix) < 3:
            continue

        # Get geometry info for this entity
        geometry_info = extract_geometry_info(geometry_df, entity_id)

        # Run all engines
        row = run_dynamics_engines(
            feature_matrix, timestamps, geometry_info, entity_id, config
        )
        row['n_observations'] = len(feature_matrix)
        row['n_features'] = feature_matrix.shape[1] if feature_matrix.ndim > 1 else 1

        results.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for dynamics")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Dynamics: {len(df)} rows (one per entity), {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Dynamics - HOW is it moving? (requires geometry)"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Dynamics Engine")
    logger.info("HOW is it moving through behavioral space?")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(VECTOR).parent

    # Check dependencies (HARD FAIL if geometry missing)
    check_dependencies('dynamics', data_path)

    output_path = get_path(DYNAMICS)

    if output_path.exists() and not args.force:
        logger.info("dynamics.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Load BOTH required inputs
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)

    vector_df = read_parquet(vector_path)
    geometry_df = read_parquet(geometry_path)

    logger.info(f"Loaded vector.parquet: {len(vector_df):,} rows, {len(vector_df.columns)} columns")
    logger.info(f"Loaded geometry.parquet: {len(geometry_df):,} rows, {len(geometry_df.columns)} columns")

    start = time.time()
    df = compute_dynamics(vector_df, geometry_df, config)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
