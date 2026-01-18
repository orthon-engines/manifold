"""
PRISM Cohort Definitions

Cohort classification for indicators by domain.
Used for memory-efficient batched geometry and categorization.

Hierarchy:
    Domain (data/raw/) → Cohort → Indicator
"""

from prism.cohorts.climate import (
    CLIMATE_SUB_COHORTS,  # Legacy alias
    CLIMATE_COHORTS,
    get_climate_subcohort,  # Legacy alias
    get_climate_cohort,
)


def get_cohort(indicator_id: str, domain: str) -> str:
    """
    Get cohort for an indicator.

    Args:
        indicator_id: The indicator ID
        domain: Domain name ('climate', etc.)

    Returns:
        Cohort name or 'other' if not classified
    """
    if domain == 'climate':
        return get_climate_cohort(indicator_id)
    else:
        return 'other'


# Legacy alias
get_subcohort = get_cohort


__all__ = [
    # New names
    'CLIMATE_COHORTS',
    'get_cohort',
    'get_climate_cohort',
    # Legacy aliases
    'CLIMATE_SUB_COHORTS',
    'get_subcohort',
    'get_climate_subcohort',
]
