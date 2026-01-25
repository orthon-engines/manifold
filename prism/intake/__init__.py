"""
PRISM Intake - Data Ingestion Layer

Handles:
- File reading (CSV, Parquet, TSV)
- Unit detection from column names
- Data validation with helpful errors

Usage:
    from prism.intake import ingest, detect_units, validate

    result = ingest("sensors.csv")
    units = detect_units(df)
    issues = validate(df)
"""

# Re-export from existing modules
from prism.intake.reader import ingest, IntakeResult
from prism.intake.units import detect_units, convert_units, UNIT_REGISTRY
from prism.intake.sanity import validate, SanityResult

__all__ = [
    'ingest',
    'IntakeResult',
    'detect_units',
    'convert_units',
    'UNIT_REGISTRY',
    'validate',
    'SanityResult',
]
