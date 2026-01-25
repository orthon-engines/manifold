"""
PRISM Capability - Data Level Detection & Engine Routing

Determines what analyses are possible based on data characteristics:
- L0: Raw numbers only -> Statistics
- L1: Column names with units -> + Unit conversion
- L2: + Physical constants -> + Reynolds, energy
- L3: + Relationships defined -> + Transfer entropy
- L4: + Spatial grid -> + Vorticity, Navier-Stokes

Usage:
    from prism.capability import detect_level, route_engines

    level = detect_level(intake_result)
    engines = route_engines(level)
"""

from prism.capability.levels import Level, LEVEL_DESCRIPTIONS
from prism.capability.router import detect_level, route_engines, get_available_engines

__all__ = [
    'Level',
    'LEVEL_DESCRIPTIONS',
    'detect_level',
    'route_engines',
    'get_available_engines',
]
