"""
Signal Pipeline — Scale 1.

raw signals → signal_vector → pairwise → state_geometry
             + dynamics on signal trajectories
             + fingerprint on signal_vectors

This pipeline processes signals within cohorts. It produces all Scale 1
parquet outputs. The operations are scale-agnostic — this pipeline just
wires them to signal-level inputs.

Steps:
  1. dynamics/breaks      on raw signals           → breaks.parquet
  2. vector/run           on raw signal windows     → signal_vector.parquet
  3. decompose/run        on signal feature matrix  → state_geometry.parquet
  4. vector/state         on state_geometry         → state_vector.parquet
  5. pairwise/run         on signal_vectors         → signal_pairwise.parquet
  6. pairwise/info        on signal pairs           → information_flow.parquet
  7. pairwise/topology    on pairwise results       → topology.parquet
  8. decompose/signal_geom on state distances       → signal_geometry.parquet
  9. decompose/dynamics   on state_geometry series   → geometry_dynamics.parquet
 10. dynamics/ftle        on signal trajectories     → ftle.parquet
 11. dynamics/velocity    on signal state space      → velocity_field.parquet
 12. dynamics/ftle_rolling on signal trajectories    → ftle_rolling.parquet
 13. dynamics/ridge       on ftle + velocity         → ridge_proximity.parquet
 14. fingerprint/run      on signal_vectors          → gaussian_fingerprint.parquet
 15. io/statistics        on signal_vector           → statistics.parquet
 16. io/correlation       on signal_vector           → correlation.parquet
 17. io/cohorts           on state_geometry          → cohorts.parquet
 18. io/zscore            on all parquets            → zscore.parquet
"""

import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List

from engines.pipeline.manifest import load_manifest, validate_manifest, get_output_dir


# Operation → stage mapping. Each entry wires an operation to its
# existing implementation via the entry_points module.
STEPS = [
    # (step_name, stage_module, outputs)
    ('breaks',             'stage_00_breaks',             ['breaks.parquet']),
    ('signal_vector',      'stage_01_signal_vector',      ['signal_vector.parquet']),
    ('state_vector',       'stage_02_state_vector',       ['state_vector.parquet']),
    ('state_geometry',     'stage_03_state_geometry',      ['state_geometry.parquet']),
    ('cohorts',            'stage_04_cohorts',             ['cohorts.parquet']),
    ('signal_geometry',    'stage_05_signal_geometry',     ['signal_geometry.parquet']),
    ('signal_pairwise',    'stage_06_signal_pairwise',     ['signal_pairwise.parquet']),
    ('geometry_dynamics',  'stage_07_geometry_dynamics',    ['geometry_dynamics.parquet']),
    ('ftle',               'stage_08_ftle',                ['ftle.parquet']),
    ('information_flow',   'stage_10_information_flow',     ['information_flow.parquet']),
    ('topology',           'stage_11_topology',             ['topology.parquet']),
    ('zscore',             'stage_12_zscore',               ['zscore.parquet']),
    ('statistics',         'stage_13_statistics',           ['statistics.parquet']),
    ('correlation',        'stage_14_correlation',          ['correlation.parquet']),
]

ATLAS_STEPS = [
    ('ftle_field',           'stage_15_ftle_field',          ['ftle_field.parquet']),
    ('break_sequence',       'stage_16_break_sequence',      ['break_sequence.parquet']),
    ('ftle_backward',        'stage_17_ftle_backward',       []),  # merges into ftle.parquet
    ('segment_comparison',   'stage_18_segment_comparison',  ['segment_comparison.parquet']),
    ('info_flow_delta',      'stage_19_info_flow_delta',     ['info_flow_delta.parquet']),
    ('velocity_field',       'stage_21_velocity_field',      ['velocity_field.parquet']),
    ('ftle_rolling',         'stage_22_ftle_rolling',        ['ftle_rolling.parquet']),
    ('ridge_proximity',      'stage_23_ridge_proximity',     ['ridge_proximity.parquet']),
    ('gaussian_fingerprint', 'stage_24_gaussian_fingerprint', ['gaussian_fingerprint.parquet']),
]


def run(
    manifest_path: str,
    atlas: bool = False,
    skip: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """Run the Signal Pipeline (Scale 1).

    This delegates to existing entry_point stage modules. The reorganization
    is in the wiring (this file), not the engines (which stay where they are).

    Args:
        manifest_path: Path to manifest.yaml
        atlas: Include atlas stages (15-23)
        skip: Step names to skip
        verbose: Print progress
    """
    # Delegate to existing run_pipeline which already handles all wiring
    from engines.entry_points.run_pipeline import run as _run_pipeline

    stages_to_run = None
    if atlas:
        # Core + atlas
        stages_to_run = [f'{i:02d}' for i in range(24)]

    skip_stages = None
    if skip:
        # Map step names to stage numbers
        name_to_num = {}
        for step_name, stage_mod, _ in STEPS + ATLAS_STEPS:
            parts = stage_mod.split('_')
            if len(parts) >= 2:
                name_to_num[step_name] = parts[1]
        skip_stages = [name_to_num[s] for s in skip if s in name_to_num]

    _run_pipeline(manifest_path, stages=stages_to_run, skip=skip_stages, verbose=verbose)
