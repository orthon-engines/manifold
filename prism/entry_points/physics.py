#!/usr/bin/env python3
"""
PRISM Physics Entry Point
=========================

Orchestrates physics and chemical engineering calculations.

REQUIRES: observations.parquet (for raw signal values)

Orchestrates physics engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/physics/.

Engines by Category:

    Classical Mechanics (6):
        kinetic_energy, potential_energy, hamiltonian, lagrangian,
        momentum, work_energy

    Fluid Mechanics (3):
        reynolds, pressure_drop, fluid_mechanics

    Heat Transfer (2):
        fourier, heat_transfer

    Mass Transfer (2):
        fick, mass_transfer

    Thermodynamics (2):
        thermodynamics, gibbs_free_energy

    Dimensionless Numbers (1):
        dimensionless (computes all: Re, Pr, Sc, Nu, Sh, etc.)

    Reaction Kinetics (2):
        reaction_kinetics, cstr_kinetics

    Process Control (1):
        process_control

Output:
    data/physics.parquet

Usage:
    python -m prism.entry_points.physics
    python -m prism.entry_points.physics --force
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS, PHYSICS
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

from prism.engines.physics import (
    # Classical Mechanics
    compute_kinetic,
    compute_potential,
    compute_hamilton,
    compute_lagrange,
    compute_momentum,
    compute_work_energy,
    # Fluid Mechanics
    compute_reynolds,
    compute_pressure_drop,
    # Heat Transfer
    compute_heat_flux,
    compute_conduction_slab,
    # Mass Transfer
    compute_molar_flux,
    # Thermodynamics
    compute_gibbs,
    # Dimensionless Numbers
    compute_all_dimensionless,
    # Reaction Kinetics
    analyze_cstr_kinetics,
    # Process Control
    first_order_response,
)

# Engine registry organized by category
ENGINES = {
    # Classical Mechanics
    'kinetic': compute_kinetic,
    'potential': compute_potential,
    'hamiltonian': compute_hamilton,
    'lagrangian': compute_lagrange,
    'momentum': compute_momentum,
    'work_energy': compute_work_energy,
    # Fluid Mechanics
    'reynolds': compute_reynolds,
    'pressure_drop': compute_pressure_drop,
    # Heat Transfer
    'heat_flux': compute_heat_flux,
    'conduction': compute_conduction_slab,
    # Mass Transfer
    'mass_flux': compute_molar_flux,
    # Thermodynamics
    'gibbs': compute_gibbs,
    # Dimensionless Numbers
    'dimensionless': compute_all_dimensionless,
    # Reaction Kinetics
    'cstr_kinetics': analyze_cstr_kinetics,
    # Process Control
    'process_dynamics': first_order_response,
}


# =============================================================================
# CONFIG
# =============================================================================

def get_signal_types(config: Dict[str, Any], signals: List[str]) -> Dict[str, List[str]]:
    """
    Get signal types from config.

    Config should have:
        physics:
          signal_types:
            velocity: [signal_1, signal_2]
            position: [signal_3]
            temperature: [signal_4]

    If not configured, returns empty dict (no signals matched to types).
    """
    physics_config = config.get('physics', {})
    return physics_config.get('signal_types', {})

def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config.json or config.yaml from data directory."""
    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    yaml_path = data_path / 'config.yaml'
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {}


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_signal_values(
    obs_df: pl.DataFrame,
    entity_id: str,
    signal_id: str,
) -> np.ndarray:
    """Extract sorted values for a specific entity/signal."""
    # Determine index column
    index_col = 'index' if 'index' in obs_df.columns else 'timestamp'

    filtered = obs_df.filter(
        (pl.col('entity_id') == entity_id) &
        (pl.col('signal_id') == signal_id)
    ).sort(index_col)

    return filtered['value'].to_numpy()


# =============================================================================
# ORCHESTRATOR - Routes to engines, no compute
# =============================================================================

def run_physics_engines(
    obs_df: pl.DataFrame,
    entity_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all physics engines for one entity.

    Pure orchestration - all compute is in engines.
    """
    results = {'entity_id': entity_id}

    # Get physics config
    physics_config = config.get('physics', {})
    enabled_engines = physics_config.get('enabled', list(ENGINES.keys()))
    engine_params = physics_config.get('params', {})

    # Get constants from config
    constants = config.get('global_constants', {})
    mass = constants.get('mass', constants.get('mass_kg', 1.0))
    spring_constant = constants.get('spring_constant', constants.get('k'))

    # Get entity signals
    entity_obs = obs_df.filter(pl.col('entity_id') == entity_id)
    signals = entity_obs['signal_id'].unique().to_list()
    signal_types = get_signal_types(config, signals)

    # Extract signals by type
    def get_signals_of_type(sig_type):
        return signal_types.get(sig_type, [])

    velocity_signals = get_signals_of_type('velocity')
    position_signals = get_signals_of_type('position')
    temperature_signals = get_signals_of_type('temperature')
    pressure_signals = get_signals_of_type('pressure')
    concentration_signals = get_signals_of_type('concentration')

    # === RUN ENGINES BY CATEGORY ===

    # --- Classical Mechanics ---
    if 'kinetic' in enabled_engines:
        params = engine_params.get('kinetic', {})
        # Prefer velocity signals, fall back to position (derive velocity)
        target_signals = velocity_signals[:3] or position_signals[:3]
        mode = 'velocity' if velocity_signals else 'position'

        for sig in target_signals:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_kinetic(values=values, mass=mass, mode=mode, **params)
                    _flatten_result(results, f"kinetic_{sig}", result)
                except Exception as e:
                    logger.debug(f"kinetic ({sig}): {e}")

    if 'potential' in enabled_engines:
        params = engine_params.get('potential', {})
        for sig in position_signals[:3]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_potential(values=values, spring_constant=spring_constant, mass=mass, **params)
                    _flatten_result(results, f"potential_{sig}", result)
                except Exception as e:
                    logger.debug(f"potential ({sig}): {e}")

    if 'hamiltonian' in enabled_engines and position_signals and velocity_signals:
        params = engine_params.get('hamiltonian', {})
        pos_sig = position_signals[0]
        vel_sig = velocity_signals[0]
        pos_values = extract_signal_values(obs_df, entity_id, pos_sig)
        vel_values = extract_signal_values(obs_df, entity_id, vel_sig)

        min_len = min(len(pos_values), len(vel_values))
        if min_len >= 2:
            try:
                result = compute_hamilton(
                    position=pos_values[:min_len],
                    velocity=vel_values[:min_len],
                    mass=mass,
                    spring_constant=spring_constant,
                    **params
                )
                _flatten_result(results, "hamiltonian", result)
            except Exception as e:
                logger.debug(f"hamiltonian: {e}")

    if 'lagrangian' in enabled_engines and position_signals and velocity_signals:
        params = engine_params.get('lagrangian', {})
        pos_sig = position_signals[0]
        vel_sig = velocity_signals[0]
        pos_values = extract_signal_values(obs_df, entity_id, pos_sig)
        vel_values = extract_signal_values(obs_df, entity_id, vel_sig)

        min_len = min(len(pos_values), len(vel_values))
        if min_len >= 2:
            try:
                result = compute_lagrange(
                    position=pos_values[:min_len],
                    velocity=vel_values[:min_len],
                    mass=mass,
                    spring_constant=spring_constant,
                    **params
                )
                _flatten_result(results, "lagrangian", result)
            except Exception as e:
                logger.debug(f"lagrangian: {e}")

    if 'momentum' in enabled_engines:
        params = engine_params.get('momentum', {})
        for sig in velocity_signals[:3]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_momentum(velocity=values, mass=mass, **params)
                    _flatten_result(results, f"momentum_{sig}", result)
                except Exception as e:
                    logger.debug(f"momentum ({sig}): {e}")

    if 'work_energy' in enabled_engines and position_signals and spring_constant:
        # Requires spring_constant to compute force (F = -k*x)
        params = engine_params.get('work_energy', {})
        for sig in position_signals[:2]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    force = -spring_constant * values
                    result = compute_work_energy(position=values, force=force, mass=mass, **params)
                    _flatten_result(results, f"work_energy_{sig}", result)
                except Exception as e:
                    logger.debug(f"work_energy ({sig}): {e}")

    # --- Fluid Mechanics ---
    if 'reynolds' in enabled_engines and velocity_signals:
        params = engine_params.get('reynolds', {})
        rho = constants.get('density', constants.get('rho', 1000.0))  # Default water
        mu = constants.get('viscosity', constants.get('mu', 0.001))
        L = constants.get('characteristic_length', constants.get('diameter', 0.1))

        for sig in velocity_signals[:2]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_reynolds(velocity=values, density=rho, viscosity=mu, length=L, **params)
                    _flatten_result(results, f"reynolds_{sig}", result)
                except Exception as e:
                    logger.debug(f"reynolds ({sig}): {e}")

    if 'pressure_drop' in enabled_engines and velocity_signals and pressure_signals:
        params = engine_params.get('pressure_drop', {})
        vel_sig = velocity_signals[0]
        pres_sig = pressure_signals[0]
        vel_values = extract_signal_values(obs_df, entity_id, vel_sig)
        pres_values = extract_signal_values(obs_df, entity_id, pres_sig)

        min_len = min(len(vel_values), len(pres_values))
        if min_len >= 2:
            try:
                result = compute_pressure_drop(
                    velocity=vel_values[:min_len],
                    pressure_in=pres_values[0],
                    pressure_out=pres_values[-1],
                    **params
                )
                _flatten_result(results, "pressure_drop", result)
            except Exception as e:
                logger.debug(f"pressure_drop: {e}")

    # --- Heat Transfer ---
    if 'heat_flux' in enabled_engines and temperature_signals:
        params = engine_params.get('heat_flux', {})
        k = constants.get('thermal_conductivity', constants.get('k_thermal', 0.6))  # Default water

        for sig in temperature_signals[:2]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    # Engine computes gradient internally
                    result = compute_heat_flux(k=k, temperature=values, **params)
                    _flatten_result(results, f"heat_flux_{sig}", result)
                except Exception as e:
                    logger.debug(f"heat_flux ({sig}): {e}")

    # --- Thermodynamics ---
    if 'gibbs' in enabled_engines and temperature_signals:
        params = engine_params.get('gibbs', {})
        temp_sig = temperature_signals[0]
        temp_values = extract_signal_values(obs_df, entity_id, temp_sig)

        pres_values = None
        if pressure_signals:
            pres_sig = pressure_signals[0]
            pres_values = extract_signal_values(obs_df, entity_id, pres_sig)
            min_len = min(len(temp_values), len(pres_values))
            temp_values = temp_values[:min_len]
            pres_values = pres_values[:min_len]

        if len(temp_values) >= 2:
            try:
                result = compute_gibbs(temperature=temp_values, pressure=pres_values, **params)
                _flatten_result(results, "gibbs", result)
            except Exception as e:
                logger.debug(f"gibbs: {e}")

    # --- Dimensionless Numbers ---
    if 'dimensionless' in enabled_engines:
        params = engine_params.get('dimensionless', {})
        # Need various properties
        rho = constants.get('density', 1000.0)
        mu = constants.get('viscosity', 0.001)
        cp = constants.get('specific_heat', 4180.0)
        k = constants.get('thermal_conductivity', 0.6)
        D = constants.get('diffusivity', 1e-9)
        L = constants.get('characteristic_length', 0.1)

        if velocity_signals:
            vel_sig = velocity_signals[0]
            vel_values = extract_signal_values(obs_df, entity_id, vel_sig)
            if len(vel_values) >= 2:
                try:
                    # Engine handles array input, computes mean internally
                    result = compute_all_dimensionless(
                        velocity=vel_values,
                        density=rho,
                        viscosity=mu,
                        specific_heat=cp,
                        thermal_conductivity=k,
                        diffusivity=D,
                        length=L,
                        **params
                    )
                    _flatten_result(results, "dimensionless", result)
                except Exception as e:
                    logger.debug(f"dimensionless: {e}")

    # --- Reaction Kinetics ---
    if 'cstr_kinetics' in enabled_engines and concentration_signals and temperature_signals:
        params = engine_params.get('cstr_kinetics', {})
        conc_sig = concentration_signals[0]
        temp_sig = temperature_signals[0]
        conc_values = extract_signal_values(obs_df, entity_id, conc_sig)
        temp_values = extract_signal_values(obs_df, entity_id, temp_sig)

        min_len = min(len(conc_values), len(temp_values))
        if min_len >= 2:
            try:
                result = analyze_cstr_kinetics(
                    concentrations=conc_values[:min_len],
                    temperatures=temp_values[:min_len],
                    **params
                )
                _flatten_result(results, "cstr_kinetics", result)
            except Exception as e:
                logger.debug(f"cstr_kinetics: {e}")

    # --- Fallback: Generic Physics for Unclassified Signals ---
    if len(results) == 1:  # Only entity_id
        logger.info(f"  {entity_id}: No classified signals, using generic physics")
        all_signals = signals[:5]

        for sig in all_signals:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                # Treat as position, derive velocity
                try:
                    result = compute_kinetic(values=values, mass=mass, mode='position')
                    _flatten_result(results, f"kinetic_{sig}", result)
                except Exception as e:
                    logger.debug(f"generic kinetic ({sig}): {e}")

                try:
                    result = compute_potential(values=values, spring_constant=spring_constant, mass=mass)
                    _flatten_result(results, f"potential_{sig}", result)
                except Exception as e:
                    logger.debug(f"generic potential ({sig}): {e}")

                try:
                    # Engine derives velocity from position internally
                    result = compute_momentum(position=values, mass=mass)
                    _flatten_result(results, f"momentum_{sig}", result)
                except Exception as e:
                    logger.debug(f"generic momentum ({sig}): {e}")

    return results


def _flatten_result(results: Dict, prefix: str, result: Any):
    """Flatten engine result into results dict."""
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if v is not None and np.isfinite(v):
                    results[f"{prefix}_{k}"] = float(v)
            elif isinstance(v, str):
                results[f"{prefix}_{k}"] = v
    elif isinstance(result, (int, float, np.integer, np.floating)):
        if result is not None and np.isfinite(result):
            results[prefix] = float(result)


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_physics(
    obs_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute physics metrics for all entities.

    Output: ONE ROW PER ENTITY
    """
    entities = obs_df['entity_id'].unique().to_list()
    n_entities = len(entities)

    # Show signal type configuration
    signals = obs_df['signal_id'].unique().to_list()
    signal_types = get_signal_types(config, signals)

    logger.info(f"Computing physics for {n_entities} entities")
    logger.info(f"Engines: {list(ENGINES.keys())}")
    if signal_types:
        logger.info("Signal types from config:")
        for sig_type, sigs in signal_types.items():
            logger.info(f"  {sig_type}: {sigs[:5]}{'...' if len(sigs) > 5 else ''}")
    else:
        logger.warning("No signal_types in config - physics engines may not run. Configure physics.signal_types in config.yaml")

    results = []

    for i, entity_id in enumerate(entities):
        row = run_physics_engines(obs_df, entity_id, config)

        n_metrics = len(row) - 1  # Exclude entity_id
        results.append(row)

        if (i + 1) % 50 == 0 or n_metrics > 0:
            logger.info(f"  {entity_id}: {n_metrics} physics metrics")

    df = pl.DataFrame(results)
    logger.info(f"Physics: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Physics")
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Physics")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(PHYSICS).parent

    # Check dependency - need observations for raw signal values
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("observations.parquet required - run: python -m prism.entry_points.fetch")
        return 1

    output_path = get_path(PHYSICS)
    if output_path.exists() and not args.force:
        logger.info("physics.parquet exists, use --force to recompute")
        return 0

    # Load observations and config
    obs_df = read_parquet(obs_path)
    config = load_config(data_path)

    n_entities = obs_df['entity_id'].n_unique()
    n_signals = obs_df['signal_id'].n_unique()
    logger.info(f"Observations: {len(obs_df)} rows, {n_entities} entities, {n_signals} signals")

    # Run physics engines
    start = time.time()
    physics_df = compute_physics(obs_df, config)

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(physics_df)} rows, {len(physics_df.columns)} columns")

    if len(physics_df.columns) <= 1:
        logger.warning("No physics metrics computed! Check signal names and config.")
        logger.warning("Physics engines need: velocity/position/temperature/pressure signals")

    # Save
    write_parquet_atomic(physics_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
