"""
PRISM Output - Standard 5-File Parquet Output

Always outputs exactly 5 files:
    - data.parquet     : Original data + characterization
    - vector.parquet   : Per-signal metrics
    - geometry.parquet : Pairwise relationships
    - dynamics.parquet : States, transitions, trends
    - physics.parquet  : Domain-specific calculations

Usage:
    from prism.output import write_outputs

    paths = write_outputs(results, output_dir)
"""

from prism.output.parquet import write_outputs, OUTPUT_FILES

__all__ = ['write_outputs', 'OUTPUT_FILES']
