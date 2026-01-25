"""
PRISM Parquet Output - Standard 5-File Output Format

Writes analysis results to 5 standardized parquet files.
"""

from pathlib import Path
from typing import Any, Dict
import polars as pl


# Standard output files
OUTPUT_FILES = {
    'data': 'data.parquet',
    'vector': 'vector.parquet',
    'geometry': 'geometry.parquet',
    'dynamics': 'dynamics.parquet',
    'physics': 'physics.parquet',
}


def write_outputs(
    results: Dict[str, pl.DataFrame],
    output_dir: Path,
    *,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Write analysis results to parquet files.

    Args:
        results: Dict mapping output name -> DataFrame
        output_dir: Directory for output files
        force: Overwrite existing files

    Returns:
        Dict mapping output name -> file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for name, filename in OUTPUT_FILES.items():
        path = output_dir / filename

        # Skip if exists and not forcing
        if path.exists() and not force:
            paths[name] = path
            continue

        # Get data for this output
        df = results.get(name)

        if df is None:
            # Create empty DataFrame with standard columns
            df = _create_empty_output(name)

        # Write atomically (write to temp, then rename)
        temp_path = path.with_suffix('.tmp')
        df.write_parquet(temp_path)
        temp_path.rename(path)

        paths[name] = path

    return paths


def _create_empty_output(name: str) -> pl.DataFrame:
    """Create empty DataFrame with appropriate schema for output type."""
    schemas = {
        'data': {
            'entity_id': pl.Utf8,
            'signal_id': pl.Utf8,
            'timestamp': pl.Float64,
            'value': pl.Float64,
        },
        'vector': {
            'entity_id': pl.Utf8,
            'signal_id': pl.Utf8,
            'metric': pl.Utf8,
            'value': pl.Float64,
        },
        'geometry': {
            'entity_id': pl.Utf8,
            'signal_i': pl.Utf8,
            'signal_j': pl.Utf8,
            'metric': pl.Utf8,
            'value': pl.Float64,
        },
        'dynamics': {
            'entity_id': pl.Utf8,
            'signal_id': pl.Utf8,
            'timestamp': pl.Float64,
            'metric': pl.Utf8,
            'value': pl.Float64,
        },
        'physics': {
            'entity_id': pl.Utf8,
            'metric': pl.Utf8,
            'value': pl.Float64,
            'unit': pl.Utf8,
        },
    }

    schema = schemas.get(name, {'value': pl.Float64})
    return pl.DataFrame(schema={k: [] for k in schema.keys()}).cast(schema)


def read_output(output_dir: Path, name: str) -> pl.DataFrame:
    """Read a specific output file."""
    path = output_dir / OUTPUT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")
    return pl.read_parquet(path)


def list_outputs(output_dir: Path) -> Dict[str, bool]:
    """List which output files exist."""
    return {
        name: (output_dir / filename).exists()
        for name, filename in OUTPUT_FILES.items()
    }


__all__ = [
    'write_outputs',
    'read_output',
    'list_outputs',
    'OUTPUT_FILES',
]
