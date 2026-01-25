"""
PRISM Data Levels - What You Have Determines What You Can Compute

Level 0: Raw Numbers
    Just values in columns. No names, no units, no context.
    Can compute: Statistics (mean, std, skew, kurtosis)

Level 1: Named Signals with Units
    Column names like 'pressure_psi', 'temp_degC'.
    Can compute: + Unit conversion, entropy, Hurst exponent

Level 2: Physical Constants
    Config includes: diameter, viscosity, density, etc.
    Can compute: + Reynolds number, energy, power

Level 3: Defined Relationships
    Config specifies: flow -> pressure, cause -> effect
    Can compute: + Transfer entropy, Granger causality

Level 4: Spatial Grid
    Data includes: x, y coordinates for field data
    Can compute: + Vorticity, divergence, Navier-Stokes
"""

from enum import IntEnum
from typing import Dict


class Level(IntEnum):
    """Data level enumeration."""
    L0_RAW = 0
    L1_UNITS = 1
    L2_CONSTANTS = 2
    L3_RELATIONSHIPS = 3
    L4_SPATIAL = 4


LEVEL_DESCRIPTIONS: Dict[Level, str] = {
    Level.L0_RAW: "Raw numbers only - statistics available",
    Level.L1_UNITS: "Units detected - dimensional analysis available",
    Level.L2_CONSTANTS: "Physical constants provided - physics calculations available",
    Level.L3_RELATIONSHIPS: "Relationships defined - causality analysis available",
    Level.L4_SPATIAL: "Spatial grid present - field analysis available",
}


# Capabilities unlocked at each level
LEVEL_CAPABILITIES: Dict[Level, list] = {
    Level.L0_RAW: [
        'statistics',     # mean, std, min, max
        'entropy',        # Shannon, permutation
        'memory',         # Hurst, ACF
        'correlation',    # Pearson, Spearman
    ],
    Level.L1_UNITS: [
        'unit_conversion',
        'dimensional_analysis',
        'physical_validation',
    ],
    Level.L2_CONSTANTS: [
        'reynolds',       # Re = ρvD/μ
        'energy',         # KE, PE, total
        'power',          # P = W/t
        'efficiency',     # η = output/input
    ],
    Level.L3_RELATIONSHIPS: [
        'transfer_entropy',
        'granger_causality',
        'mutual_information',
        'causal_discovery',
    ],
    Level.L4_SPATIAL: [
        'vorticity',      # ω = ∇×v
        'divergence',     # ∇·v
        'gradient',       # ∇f
        'laplacian',      # ∇²f
        'navier_stokes',  # Full NS analysis
    ],
}


def capabilities_at_level(level: Level) -> list:
    """Get all capabilities available at a given level (cumulative)."""
    caps = []
    for lvl in Level:
        if lvl <= level:
            caps.extend(LEVEL_CAPABILITIES.get(lvl, []))
    return caps


__all__ = [
    'Level',
    'LEVEL_DESCRIPTIONS',
    'LEVEL_CAPABILITIES',
    'capabilities_at_level',
]
