"""
PRISM Analyze - Main Entry Point

Usage:
    from prism import analyze

    results = analyze("turbofan_data.csv")
    results = analyze("sensors.parquet", config={'entity_column': 'unit_id'})

Outputs 5 parquet files:
    - data.parquet     : Original data + characterization
    - vector.parquet   : Per-signal metrics (62 engines)
    - geometry.parquet : Pairwise relationships
    - dynamics.parquet : States, transitions, trends
    - physics.parquet  : Domain-specific calculations
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import polars as pl


@dataclass
class AnalysisResult:
    """Result container for PRISM analysis."""

    data_path: Path
    vector_path: Path
    geometry_path: Path
    dynamics_path: Path
    physics_path: Path

    # Lazy-loaded DataFrames
    _data: Optional[pl.DataFrame] = None
    _vector: Optional[pl.DataFrame] = None
    _geometry: Optional[pl.DataFrame] = None
    _dynamics: Optional[pl.DataFrame] = None
    _physics: Optional[pl.DataFrame] = None

    @property
    def data(self) -> pl.DataFrame:
        if self._data is None:
            self._data = pl.read_parquet(self.data_path)
        return self._data

    @property
    def vector(self) -> pl.DataFrame:
        if self._vector is None:
            self._vector = pl.read_parquet(self.vector_path)
        return self._vector

    @property
    def geometry(self) -> pl.DataFrame:
        if self._geometry is None:
            self._geometry = pl.read_parquet(self.geometry_path)
        return self._geometry

    @property
    def dynamics(self) -> pl.DataFrame:
        if self._dynamics is None:
            self._dynamics = pl.read_parquet(self.dynamics_path)
        return self._dynamics

    @property
    def physics(self) -> pl.DataFrame:
        if self._physics is None:
            self._physics = pl.read_parquet(self.physics_path)
        return self._physics

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics for all outputs."""
        return {
            'data': {'rows': len(self.data), 'columns': len(self.data.columns)},
            'vector': {'rows': len(self.vector), 'columns': len(self.vector.columns)},
            'geometry': {'rows': len(self.geometry), 'columns': len(self.geometry.columns)},
            'dynamics': {'rows': len(self.dynamics), 'columns': len(self.dynamics.columns)},
            'physics': {'rows': len(self.physics), 'columns': len(self.physics.columns)},
        }


def analyze(
    data: Union[str, Path, pl.DataFrame],
    *,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> AnalysisResult:
    """
    Analyze sensor data and output 5 parquet files.

    Args:
        data: Input data - file path (CSV/Parquet/TSV) or DataFrame
        config: Optional configuration dict:
            - entity_column: Column identifying entities (e.g., 'unit_id')
            - constants: Dict of physical constants (e.g., {'diameter_in': 4})
            - domain: Domain hint ('turbomachinery', 'fluid', etc.)
        output_dir: Directory for output files (default: ./data/)
        force: Recompute even if outputs exist

    Returns:
        AnalysisResult with paths to all 5 output files
    """
    from prism.intake import ingest
    from prism.capability import detect_level, route_engines
    from prism.output import write_outputs

    config = config or {}
    output_dir = Path(output_dir) if output_dir else Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Intake
    intake_result = ingest(data, config=config)

    # Stage 2: Detect data level (L0-L4)
    level = detect_level(intake_result)

    # Stage 3: Route to appropriate engines
    engines = route_engines(level, config=config)

    # Stage 4: Execute engines
    results = {
        'data': intake_result.data,
        'vector': _compute_vector(intake_result, engines['vector']),
        'geometry': _compute_geometry(intake_result, engines['geometry']),
        'dynamics': _compute_dynamics(intake_result, engines['dynamics']),
        'physics': _compute_physics(intake_result, engines['physics'], config),
    }

    # Stage 5: Write outputs
    paths = write_outputs(results, output_dir, force=force)

    return AnalysisResult(
        data_path=paths['data'],
        vector_path=paths['vector'],
        geometry_path=paths['geometry'],
        dynamics_path=paths['dynamics'],
        physics_path=paths['physics'],
    )


def _compute_vector(intake_result, engines: list) -> pl.DataFrame:
    """Compute per-signal vector metrics."""
    from prism.engines.vector import compute_all
    return compute_all(intake_result.signals, engines)


def _compute_geometry(intake_result, engines: list) -> pl.DataFrame:
    """Compute pairwise geometry metrics."""
    from prism.engines.geometry import compute_all
    return compute_all(intake_result.signals, engines)


def _compute_dynamics(intake_result, engines: list) -> pl.DataFrame:
    """Compute temporal dynamics metrics."""
    from prism.engines.dynamics import compute_all
    return compute_all(intake_result.signals, engines)


def _compute_physics(intake_result, engines: list, config: dict) -> pl.DataFrame:
    """Compute physics metrics (requires constants)."""
    from prism.engines.physics import compute_all
    constants = config.get('constants', {})
    return compute_all(intake_result.signals, engines, constants=constants)
