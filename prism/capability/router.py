"""
PRISM Capability Router - Route Data to Appropriate Engines

Examines data and config to determine:
1. What level of analysis is possible
2. Which engines to run
3. What outputs to generate
"""

from typing import Any, Dict, List, Optional
from prism.capability.levels import Level, capabilities_at_level


def detect_level(intake_result) -> Level:
    """
    Detect data level from intake result.

    Args:
        intake_result: Result from prism.intake.ingest()

    Returns:
        Level enum indicating data sophistication
    """
    # Check for spatial grid (L4)
    spatial_cols = {'x', 'y', 'z', 'x_coord', 'y_coord', 'z_coord'}
    col_names = {c.lower() for c in intake_result.data.columns}
    if spatial_cols & col_names:
        return Level.L4_SPATIAL

    # Use level from intake if set
    if hasattr(intake_result, 'level') and intake_result.level > 0:
        return Level(min(intake_result.level, 4))

    # Check for units (L1)
    if intake_result.units:
        return Level.L1_UNITS

    return Level.L0_RAW


def route_engines(
    level: Level,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """
    Determine which engines to run based on data level.

    Args:
        level: Detected data level
        config: Optional config with overrides

    Returns:
        Dict mapping output file -> list of engine names
    """
    config = config or {}

    # Base engines available at all levels
    vector_engines = [
        # Statistics
        'statistics',

        # Memory
        'hurst_rs',
        'hurst_dfa',
        'acf_decay',
        'spectral_slope',

        # Information
        'permutation_entropy',
        'sample_entropy',
        'entropy_rate',

        # Frequency
        'spectral_analysis',
        'wavelet_analysis',

        # Volatility
        'garch',
        'realized_volatility',
        'bipower_variation',

        # Recurrence
        'rqa',

        # Typology
        'stationarity',
        'trend_detection',
        'seasonality',
        'changepoint',
    ]

    geometry_engines = [
        'correlation',
        'partial_correlation',
        'distance',
        'clustering',
        'pca',
        'mutual_information',
    ]

    dynamics_engines = [
        'hd_slope',
        'regime_detection',
        'trend_velocity',
    ]

    physics_engines = []

    # Level 1+: Add dimensional analysis
    if level >= Level.L1_UNITS:
        vector_engines.append('unit_validation')

    # Level 2+: Add physics engines
    if level >= Level.L2_CONSTANTS:
        physics_engines.extend([
            'kinetic_energy',
            'potential_energy',
            'reynolds_number',
            'power',
        ])

    # Level 3+: Add causality engines
    if level >= Level.L3_RELATIONSHIPS:
        dynamics_engines.extend([
            'granger_causality',
            'transfer_entropy',
        ])
        geometry_engines.append('causal_graph')

    # Level 4: Add field analysis
    if level >= Level.L4_SPATIAL:
        physics_engines.extend([
            'vorticity',
            'divergence',
            'q_criterion',
        ])

    # Apply config overrides
    if 'engines' in config:
        if 'vector' in config['engines']:
            vector_engines = config['engines']['vector']
        if 'geometry' in config['engines']:
            geometry_engines = config['engines']['geometry']
        if 'dynamics' in config['engines']:
            dynamics_engines = config['engines']['dynamics']
        if 'physics' in config['engines']:
            physics_engines = config['engines']['physics']

    # Apply domain-specific engines
    domain = config.get('domain')
    if domain == 'turbomachinery':
        physics_engines.extend([
            'compressor_efficiency',
            'turbine_efficiency',
            'polytropic_efficiency',
            'corrected_parameters',
        ])
    elif domain == 'fluid':
        physics_engines.extend([
            'vorticity',
            'divergence',
            'stream_function',
        ])

    return {
        'vector': vector_engines,
        'geometry': geometry_engines,
        'dynamics': dynamics_engines,
        'physics': physics_engines,
    }


def get_available_engines() -> Dict[str, List[str]]:
    """Get all available engines by category."""
    return {
        'vector': [
            'statistics', 'hurst_rs', 'hurst_dfa', 'acf_decay', 'spectral_slope',
            'permutation_entropy', 'sample_entropy', 'entropy_rate',
            'spectral_analysis', 'wavelet_analysis',
            'garch', 'realized_volatility', 'bipower_variation',
            'rqa', 'stationarity', 'trend_detection', 'seasonality', 'changepoint',
        ],
        'geometry': [
            'correlation', 'partial_correlation', 'distance', 'clustering',
            'pca', 'mutual_information', 'mst', 'copula', 'lof',
        ],
        'dynamics': [
            'hd_slope', 'regime_detection', 'trend_velocity',
            'granger_causality', 'transfer_entropy', 'dtw', 'cointegration',
        ],
        'physics': [
            'kinetic_energy', 'potential_energy', 'reynolds_number', 'power',
            'vorticity', 'divergence', 'q_criterion',
            'compressor_efficiency', 'turbine_efficiency',
        ],
    }


__all__ = [
    'detect_level',
    'route_engines',
    'get_available_engines',
]
