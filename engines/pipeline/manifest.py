"""
Manifest reading and validation.

Reads manifest.yaml. Validates cohort structure.
Decides which pipelines run. Wires inputs to outputs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load manifest.yaml.

    Args:
        manifest_path: path to manifest.yaml

    Returns:
        Manifest dict
    """
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required structure.

    Raises ValueError if structure is invalid.
    """
    if 'system' not in manifest:
        raise ValueError("Manifest missing 'system' section")

    system = manifest['system']
    if 'window' not in system:
        raise ValueError("Manifest missing system.window")
    if 'stride' not in system:
        raise ValueError("Manifest missing system.stride")

    if 'cohorts' not in manifest:
        raise ValueError("Manifest missing 'cohorts' section")

    # Validate each cohort has at least one signal
    for cohort_name, cohort_config in manifest['cohorts'].items():
        signals = [k for k, v in cohort_config.items() if isinstance(v, dict)]
        if not signals:
            raise ValueError(f"Cohort '{cohort_name}' has no signals")


def get_n_cohorts(manifest: Dict[str, Any]) -> int:
    """Count number of cohorts in manifest."""
    return len(manifest.get('cohorts', {}))


def get_all_signals(manifest: Dict[str, Any]) -> list:
    """Get flat list of all signal_ids across all cohorts."""
    signals = []
    for cohort_config in manifest.get('cohorts', {}).values():
        for signal_id, config in cohort_config.items():
            if isinstance(config, dict):
                signals.append(signal_id)
    return signals


def should_run_scale2(manifest: Dict[str, Any]) -> bool:
    """Determine if Scale 2 (cohort pipeline) should run.

    Scale 2 only runs when n_cohorts > 1.
    """
    return get_n_cohorts(manifest) > 1


def get_output_dir(manifest: Dict[str, Any], manifest_path: str) -> Path:
    """Get output directory from manifest."""
    manifest_dir = Path(manifest_path).parent
    output_rel = manifest.get('paths', {}).get('output_dir', 'output')
    output_dir = manifest_dir / output_rel
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
