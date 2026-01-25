"""
PRISM Unit Detection - Extract Units from Column Names

Detects units from column name suffixes like:
- pressure_psi -> psi
- temp_degC -> degC
- flow_kg_s -> kg/s

Re-exports from prism.unitspec for full unit conversion.
"""

from typing import Dict, Optional
import polars as pl

# Import full unit system from existing module
try:
    from prism.unitspec import (
        UNIT_REGISTRY,
        convert,
        get_dimensions,
        is_compatible,
        to_si,
    )
except ImportError:
    # Fallback if unitspec not available
    UNIT_REGISTRY = {}
    def convert(value, from_unit, to_unit): return value
    def get_dimensions(unit): return None
    def is_compatible(unit1, unit2): return True
    def to_si(value, unit): return value


# Unit suffix patterns for auto-detection
UNIT_SUFFIXES = {
    # Temperature
    '_degc': 'degC',
    '_degf': 'degF',
    '_kelvin': 'K',
    '_k': 'K',
    '_degr': 'degR',
    '_r': 'degR',

    # Pressure
    '_psi': 'psi',
    '_psia': 'psia',
    '_psig': 'psig',
    '_bar': 'bar',
    '_pa': 'Pa',
    '_kpa': 'kPa',
    '_mpa': 'MPa',
    '_atm': 'atm',

    # Mass flow
    '_kg_s': 'kg/s',
    '_kgs': 'kg/s',
    '_lb_s': 'lb/s',
    '_lbs': 'lb/s',
    '_lbm_s': 'lbm/s',

    # Volume flow
    '_m3_s': 'm3/s',
    '_l_min': 'L/min',
    '_gpm': 'gpm',

    # Speed
    '_rpm': 'rpm',
    '_rad_s': 'rad/s',
    '_m_s': 'm/s',
    '_ft_s': 'ft/s',

    # Force
    '_n': 'N',
    '_kn': 'kN',
    '_lbf': 'lbf',

    # Power
    '_w': 'W',
    '_kw': 'kW',
    '_mw': 'MW',
    '_hp': 'hp',

    # Energy
    '_j': 'J',
    '_kj': 'kJ',
    '_btu': 'BTU',
    '_btu_lb': 'BTU/lb',

    # Electrical
    '_v': 'V',
    '_a': 'A',
    '_ohm': 'ohm',

    # Vibration
    '_g': 'g',
    '_m_s2': 'm/s2',
    '_mm_s': 'mm/s',
    '_um': 'um',

    # Length
    '_m': 'm',
    '_mm': 'mm',
    '_cm': 'cm',
    '_in': 'in',
    '_ft': 'ft',

    # Time
    '_s': 's',
    '_ms': 'ms',
    '_min': 'min',
    '_hr': 'hr',
}


def detect_units(df: pl.DataFrame) -> Dict[str, str]:
    """
    Detect units from DataFrame column names.

    Args:
        df: DataFrame with columns that may have unit suffixes

    Returns:
        Dict mapping column name -> detected unit
    """
    units = {}

    for col in df.columns:
        col_lower = col.lower()

        for suffix, unit in UNIT_SUFFIXES.items():
            if col_lower.endswith(suffix):
                units[col] = unit
                break

    return units


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> float:
    """Convert value between units."""
    return convert(value, from_unit, to_unit)


__all__ = [
    'detect_units',
    'convert_units',
    'UNIT_REGISTRY',
    'UNIT_SUFFIXES',
    'to_si',
    'get_dimensions',
    'is_compatible',
]
