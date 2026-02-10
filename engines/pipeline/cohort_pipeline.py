"""
Cohort Pipeline — Scale 2.

state_geometry → cohort_vector → pairwise → system_geometry
               + dynamics on cohort trajectories
               + fingerprint on cohort_vectors

Only runs when n_cohorts > 1.

This pipeline processes cohorts across the system. It reuses the SAME
operations as the signal pipeline but wires them to cohort-level inputs.

Steps:
  1. vector/run           on state_geometry outputs  → cohort_vector.parquet
  2. decompose/run        on cohort feature matrix   → system_geometry.parquet
  3. pairwise/run         on cohort_vectors          → cohort_pairwise.parquet
  4. pairwise/info        on cohort pairs            → cohort_information_flow.parquet
  5. pairwise/topology    on cohort pairwise         → cohort_topology.parquet
  6. dynamics/ftle        on cohort trajectories      → cohort_ftle.parquet
  7. dynamics/velocity    on cohort state space       → cohort_velocity_field.parquet
  8. fingerprint/run      on cohort_vectors          → cohort_fingerprint.parquet
                                                       cohort_similarity.parquet
"""

import polars as pl
from pathlib import Path
from typing import Optional

from engines.pipeline.manifest import load_manifest, should_run_scale2


def run(
    manifest_path: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Run the Cohort Pipeline (Scale 2).

    Delegates to existing entry_point stage modules (25-32).
    Only runs when n_cohorts > 1 (checked via state_geometry).

    Args:
        manifest_path: Path to manifest.yaml
        output_dir: Override output directory
        verbose: Print progress
    """
    manifest = load_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent

    if output_dir:
        out = Path(output_dir)
    else:
        out = manifest_dir / manifest.get('paths', {}).get('output_dir', 'output')

    # Guard: check state_geometry exists and has multiple cohorts
    sg_path = out / 'state_geometry.parquet'
    if not sg_path.exists():
        if verbose:
            print("Scale 2 skipped: state_geometry.parquet not found")
        return

    sg = pl.read_parquet(str(sg_path))
    if len(sg) == 0:
        if verbose:
            print("Scale 2 skipped: state_geometry.parquet is empty")
        return

    if 'cohort' not in sg.columns or sg['cohort'].n_unique() < 2:
        n = sg['cohort'].n_unique() if 'cohort' in sg.columns else 0
        if verbose:
            print(f"Scale 2 skipped: n_cohorts={n} < 2")
        return

    if verbose:
        n_cohorts = sg['cohort'].n_unique()
        print("=" * 70)
        print(f"COHORT PIPELINE (Scale 2) — {n_cohorts} cohorts")
        print("=" * 70)

    # Step 1: cohort_vector
    _run_step('stage_25_cohort_vector', verbose, lambda mod: mod.run(
        str(sg_path),
        str(out / 'cohort_vector.parquet'),
        verbose=verbose,
    ))

    cv_path = out / 'cohort_vector.parquet'
    if not cv_path.exists():
        if verbose:
            print("  cohort_vector.parquet not produced, stopping Scale 2")
        return

    # Step 2: system_geometry
    _run_step('stage_26_system_geometry', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'system_geometry.parquet'),
        verbose=verbose,
    ))

    # Step 3: cohort_pairwise
    loadings_path = out / 'system_geometry_loadings.parquet'
    _run_step('stage_27_cohort_pairwise', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'cohort_pairwise.parquet'),
        system_geometry_loadings_path=str(loadings_path) if loadings_path.exists() else None,
        verbose=verbose,
    ))

    # Step 4: cohort_information_flow
    cp_path = out / 'cohort_pairwise.parquet'
    if cp_path.exists():
        cp = pl.read_parquet(str(cp_path))
        if len(cp) > 0:
            _run_step('stage_28_cohort_information_flow', verbose, lambda mod: mod.run(
                str(cv_path),
                str(cp_path),
                str(out / 'cohort_information_flow.parquet'),
                verbose=verbose,
            ))

    # Step 5: cohort_topology
    _run_step('stage_29_cohort_topology', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'cohort_topology.parquet'),
        verbose=verbose,
    ))

    # Step 6: cohort_ftle
    _run_step('stage_30_cohort_ftle', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'cohort_ftle.parquet'),
        verbose=verbose,
    ))

    # Step 7: cohort_velocity_field
    _run_step('stage_31_cohort_velocity_field', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'cohort_velocity_field.parquet'),
        verbose=verbose,
    ))

    # Step 8: cohort_fingerprint
    _run_step('stage_32_cohort_fingerprint', verbose, lambda mod: mod.run(
        str(cv_path),
        str(out / 'cohort_fingerprint.parquet'),
        str(out / 'cohort_similarity.parquet'),
        verbose=verbose,
    ))

    if verbose:
        print("=" * 70)
        print("COHORT PIPELINE COMPLETE")
        print("=" * 70)


def _run_step(stage_name: str, verbose: bool, fn) -> None:
    """Run a single pipeline step with error handling."""
    import importlib

    if verbose:
        print(f"\n--- {stage_name} ---")

    try:
        module = importlib.import_module(f'engines.entry_points.{stage_name}')
        fn(module)
    except Exception as e:
        if verbose:
            print(f"  Error in {stage_name}: {e}")
