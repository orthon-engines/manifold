"""
ORTHON Utility Functions

Visual encoding, characterization, and narrative generation.
"""

from streamlit.utils.visual_encoding import (
    color_direct_rgb,
    color_from_dominant,
    color_from_hsl,
    compute_distinctiveness,
    compute_dominant_trait,
    compute_dominant_pole,
    add_visual_columns,
    add_projections,
    add_clusters,
    AXES,
)

__all__ = [
    'color_direct_rgb',
    'color_from_dominant',
    'color_from_hsl',
    'compute_distinctiveness',
    'compute_dominant_trait',
    'compute_dominant_pole',
    'add_visual_columns',
    'add_projections',
    'add_clusters',
    'AXES',
]
