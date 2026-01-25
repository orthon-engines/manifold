#!/usr/bin/env python3
"""
PRISM Geometry Entry Point
==========================

Answers: WHERE does the entity live in behavioral space?

REQUIRES: vector.parquet

Orchestrates geometry engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/geometry/.

Engines (17 total):
    Class-based (9):
        PCAEngine, MSTEngine, ClusteringEngine, LOFEngine, DistanceEngine,
        ConvexHullEngine, CopulaEngine, MutualInformationEngine, BarycenterEngine

    Function-based (8):
        compute_coupling_matrix, compute_divergence, discover_modes,
        compute_snapshot, compute_covariance, compute_effective_dim,
        compute_baseline_distance, compute_correlation_structure

Output:
    data/geometry.parquet - One row per entity

Usage:
    python -m prism.entry_points.geometry
    python -m prism.entry_points.geometry --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import polars as pl

from prism.core.dependencies import check_dependencies
from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY
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

from prism.engines.geometry import (
    # Class-based engines
    PCAEngine,
    MSTEngine,
    ClusteringEngine,
    LOFEngine,
    DistanceEngine,
    ConvexHullEngine,
    CopulaEngine,
    MutualInformationEngine,
    BarycenterEngine,
    # Function-based engines
    compute_coupling_matrix,
    compute_divergence,
    discover_modes,
    compute_snapshot,
    compute_covariance,
    compute_covariance_matrix,
    compute_effective_dim,
    compute_baseline_distance,
    compute_correlation_structure,
)

# Engine registry
CLASS_ENGINES = {
    'pca': PCAEngine,
    'mst': MSTEngine,
    'clustering': ClusteringEngine,
    'lof': LOFEngine,
    'distance': DistanceEngine,
    'convex_hull': ConvexHullEngine,
    'copula': CopulaEngine,
    'mutual_information': MutualInformationEngine,
    'barycenter': BarycenterEngine,
}

FUNCTION_ENGINES = {
    'coupling': compute_coupling_matrix,
    'divergence': compute_divergence,
    'modes': discover_modes,
    'snapshot': compute_snapshot,
    'covariance': compute_covariance,
    'effective_dim': compute_effective_dim,
    'baseline_distance': compute_baseline_distance,
    'correlation_structure': compute_correlation_structure,
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
            f"Create config.yaml with geometry settings.\n"
            f"{'='*60}"
        )

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    return user_config.get('geometry', {})


# =============================================================================
# BEHAVIORAL MATRIX CONSTRUCTION
# =============================================================================

def build_feature_matrix(
    vector_df: pl.DataFrame,
    entity_id: str,
) -> tuple:
    """
    Build feature matrix for one entity from vector metrics.

    Returns:
        (pandas DataFrame for engines, list of metric column names)
    """
    entity_data = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_data) == 0:
        return None, []

    # Identify metric columns (exclude identifiers)
    id_cols = {'entity_id', 'signal_id', 'window_idx', 'window_start', 'window_end', 'n_samples'}
    metric_cols = [c for c in entity_data.columns
                   if c not in id_cols
                   and entity_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not metric_cols:
        return None, []

    # Convert to pandas for engine compatibility
    # Engines expect: rows=observations, cols=features
    pdf = entity_data.select(metric_cols).to_pandas()
    pdf = pdf.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return pdf, metric_cols


# =============================================================================
# ORCHESTRATOR - Routes to engines, no compute
# =============================================================================

def run_geometry_engines(
    feature_df: pd.DataFrame,
    entity_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all geometry engines on feature matrix.

    Pure orchestration - all compute is in engines.
    """
    results = {'entity_id': entity_id}
    run_id = f"{entity_id}_geometry"

    # Get engine config
    enabled_engines = config.get('enabled', list(CLASS_ENGINES.keys()) + list(FUNCTION_ENGINES.keys()))
    engine_params = config.get('params', {})

    # === CLASS-BASED ENGINES ===
    for engine_name, EngineClass in CLASS_ENGINES.items():
        if engine_name not in enabled_engines:
            continue

        try:
            engine = EngineClass()
            params = engine_params.get(engine_name, {})
            result = engine.run(df=feature_df, run_id=run_id, **params)

            # Flatten result into row (no inline compute - just pass through)
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if np.isfinite(v):
                            results[f"{engine_name}_{k}"] = float(v)
                    elif isinstance(v, np.ndarray) and v.size <= 100:
                        # Store small arrays as JSON, skip large arrays
                        results[f"{engine_name}_{k}_json"] = json.dumps(v.tolist())

        except Exception as e:
            logger.debug(f"{engine_name} failed for {entity_id}: {e}")

    # === FUNCTION-BASED ENGINES ===
    feature_matrix = feature_df.values

    for engine_name, compute_fn in FUNCTION_ENGINES.items():
        if engine_name not in enabled_engines:
            continue

        try:
            params = engine_params.get(engine_name, {})

            # Each function has different signature - route appropriately
            if engine_name == 'coupling':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'divergence':
                # Needs two distributions - use first half vs second half
                mid = len(feature_matrix) // 2
                if mid > 0:
                    result = compute_fn(feature_matrix[:mid], feature_matrix[mid:], **params)
                else:
                    continue
            elif engine_name == 'modes':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'snapshot':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'covariance':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'effective_dim':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'baseline_distance':
                result = compute_fn(feature_matrix, **params)
            elif engine_name == 'correlation_structure':
                result = compute_fn(feature_matrix, **params)
            else:
                result = compute_fn(feature_matrix, **params)

            # Flatten result
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if np.isfinite(v):
                            results[f"{engine_name}_{k}"] = float(v)
                    elif isinstance(v, np.ndarray) and v.size <= 100:
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

def compute_geometry(
    vector_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute geometry for all entities.

    Output: ONE ROW PER ENTITY
    """
    entities = vector_df.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)

    logger.info(f"Computing geometry for {n_entities} entities")
    logger.info(f"Class engines: {list(CLASS_ENGINES.keys())}")
    logger.info(f"Function engines: {list(FUNCTION_ENGINES.keys())}")

    results = []

    for i, entity_id in enumerate(entities):
        # Build feature matrix
        feature_df, metric_cols = build_feature_matrix(vector_df, entity_id)

        if feature_df is None or len(feature_df) < 3:
            continue

        # Run all engines
        row = run_geometry_engines(feature_df, entity_id, config)
        row['n_observations'] = len(feature_df)
        row['n_features'] = len(metric_cols)

        results.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for geometry")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Geometry: {len(df)} rows (one per entity), {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Geometry - WHERE does it live? (requires vector)"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Geometry Engine")
    logger.info("WHERE does it live in behavioral space?")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(VECTOR).parent

    # Check dependencies (HARD FAIL if vector missing)
    check_dependencies('geometry', data_path)

    output_path = get_path(GEOMETRY)

    if output_path.exists() and not args.force:
        logger.info("geometry.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Load input
    vector_path = get_path(VECTOR)
    vector_df = read_parquet(vector_path)
    logger.info(f"Loaded vector.parquet: {len(vector_df):,} rows, {len(vector_df.columns)} columns")

    start = time.time()
    df = compute_geometry(vector_df, config)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
