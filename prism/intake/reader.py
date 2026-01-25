"""
PRISM Intake Reader - Universal Data Ingestion

Reads CSV, Parquet, TSV files and auto-detects:
- Entity column (unit_id, engine_id, etc.)
- Signal columns (sensor values)
- Units from column name suffixes (_psi, _degC, etc.)
- Time/cycle columns
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import polars as pl

from prism.intake.units import detect_units
from prism.intake.sanity import validate


@dataclass
class Signal:
    """A detected signal with metadata."""
    name: str
    unit: Optional[str] = None
    category: Optional[str] = None  # temperature, pressure, etc.
    values: Optional[pl.Series] = None


@dataclass
class IntakeResult:
    """Result of data ingestion."""
    data: pl.DataFrame
    signals: List[Signal]
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    units: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    level: int = 0  # Data level (0-4)


def ingest(
    source: Union[str, Path, pl.DataFrame],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> IntakeResult:
    """
    Ingest data from file or DataFrame.

    Args:
        source: File path or DataFrame
        config: Optional config with:
            - entity_column: Override entity detection
            - time_column: Override time detection
            - units: Dict of column -> unit overrides

    Returns:
        IntakeResult with data, signals, and metadata
    """
    config = config or {}

    # Load data
    if isinstance(source, pl.DataFrame):
        df = source
    else:
        df = _read_file(Path(source))

    # Detect structure
    entity_col = config.get('entity_column') or _detect_entity_column(df)
    time_col = config.get('time_column') or _detect_time_column(df)

    # Detect units from column names
    units = detect_units(df)
    units.update(config.get('units', {}))

    # Build signal list
    signals = _build_signals(df, entity_col, time_col, units)

    # Validate data
    validation = validate(df)
    issues = [str(issue) for issue in validation.issues]

    # Determine data level
    level = _determine_level(df, units, config)

    return IntakeResult(
        data=df,
        signals=signals,
        entity_column=entity_col,
        time_column=time_col,
        units=units,
        issues=issues,
        level=level,
    )


def _read_file(path: Path) -> pl.DataFrame:
    """Read file based on extension."""
    suffix = path.suffix.lower()

    if suffix == '.parquet':
        return pl.read_parquet(path)
    elif suffix == '.csv':
        return pl.read_csv(path, infer_schema_length=10000)
    elif suffix in ('.tsv', '.txt'):
        return pl.read_csv(path, separator='\t', infer_schema_length=10000)
    else:
        # Try CSV as default
        return pl.read_csv(path, infer_schema_length=10000)


def _detect_entity_column(df: pl.DataFrame) -> Optional[str]:
    """Detect the entity/unit identifier column."""
    patterns = [
        'unit_id', 'unit', 'engine_id', 'engine', 'entity_id', 'entity',
        'machine_id', 'machine', 'asset_id', 'asset', 'id',
    ]

    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern in col_lower:
                return col

    return None


def _detect_time_column(df: pl.DataFrame) -> Optional[str]:
    """Detect the time/cycle column."""
    patterns = [
        'time', 'timestamp', 'cycle', 'cycles', 't', 'datetime',
        'date', 'step', 'index',
    ]

    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if col_lower == pattern or col_lower.startswith(pattern + '_'):
                return col

    return None


def _build_signals(
    df: pl.DataFrame,
    entity_col: Optional[str],
    time_col: Optional[str],
    units: Dict[str, str],
) -> List[Signal]:
    """Build list of Signal objects from DataFrame."""
    exclude = {entity_col, time_col} if entity_col or time_col else set()
    exclude.discard(None)

    signals = []
    for col in df.columns:
        if col in exclude:
            continue

        # Skip non-numeric columns
        if df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            continue

        unit = units.get(col)
        category = _unit_to_category(unit) if unit else None

        signals.append(Signal(
            name=col,
            unit=unit,
            category=category,
        ))

    return signals


def _unit_to_category(unit: str) -> Optional[str]:
    """Map unit to physical category."""
    categories = {
        # Temperature
        'degC': 'temperature', 'degF': 'temperature', 'K': 'temperature', 'degR': 'temperature',
        # Pressure
        'psi': 'pressure', 'psia': 'pressure', 'psig': 'pressure', 'bar': 'pressure',
        'Pa': 'pressure', 'kPa': 'pressure', 'MPa': 'pressure', 'atm': 'pressure',
        # Flow
        'kg/s': 'mass_flow', 'lb/s': 'mass_flow', 'lbm/s': 'mass_flow',
        'm3/s': 'volume_flow', 'L/min': 'volume_flow', 'gpm': 'volume_flow',
        # Speed
        'rpm': 'rotational_speed', 'rad/s': 'angular_velocity',
        'm/s': 'velocity', 'ft/s': 'velocity', 'km/h': 'velocity', 'mph': 'velocity',
        # Force/Torque
        'N': 'force', 'lbf': 'force', 'kN': 'force',
        'Nm': 'torque', 'ft-lbf': 'torque',
        # Power/Energy
        'W': 'power', 'kW': 'power', 'MW': 'power', 'hp': 'power',
        'J': 'energy', 'kJ': 'energy', 'BTU': 'energy',
        # Electrical
        'V': 'voltage', 'A': 'current', 'ohm': 'resistance',
        # Vibration
        'g': 'acceleration', 'm/s2': 'acceleration',
        'mm/s': 'velocity', 'um': 'displacement',
    }
    return categories.get(unit)


def _determine_level(df: pl.DataFrame, units: Dict[str, str], config: dict) -> int:
    """
    Determine data level (0-4).

    L0: Raw numbers only
    L1: Column names with units
    L2: + Physical constants
    L3: + Relationships defined
    L4: + Spatial grid (x, y coordinates)
    """
    # L0: No units detected
    if not units:
        return 0

    # L1: Has units
    level = 1

    # L2: Has constants
    if config.get('constants'):
        level = 2

    # L3: Has relationships
    if config.get('relationships'):
        level = 3

    # L4: Has spatial grid
    spatial_cols = {'x', 'y', 'z', 'x_coord', 'y_coord', 'z_coord'}
    if any(col.lower() in spatial_cols for col in df.columns):
        level = 4

    return level
