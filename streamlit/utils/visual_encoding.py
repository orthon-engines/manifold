"""
Visual Encoding Utilities for ORTHON Dashboard

Computed at render time, not stored in parquet.
Keep data clean, compute visuals on demand.
"""

import numpy as np
import pandas as pd
from typing import Optional
import colorsys

# -----------------------------------------------------------------------------
# Color Encoding
# -----------------------------------------------------------------------------

def color_direct_rgb(row: pd.Series, 
                     r_axis: str = 'volatility',
                     g_axis: str = 'frequency', 
                     b_axis: str = 'memory') -> str:
    """
    Direct RGB mapping from 3 axes.
    
    Default: R=volatility, G=frequency, B=memory
    - Red/warm = volatile
    - Green = periodic
    - Blue = persistent
    """
    r = int(row.get(r_axis, 0.5) * 255)
    g = int(row.get(g_axis, 0.5) * 255)
    b = int(row.get(b_axis, 0.5) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


def color_from_dominant(row: pd.Series, axes: list = None) -> str:
    """
    Color based on dominant trait.
    """
    TRAIT_COLORS = {
        'memory': '#4C78A8',       # Blue
        'information': '#F58518',  # Orange
        'frequency': '#E45756',    # Red
        'volatility': '#72B7B2',   # Teal
        'dynamics': '#54A24B',     # Green
        'recurrence': '#EECA3B',   # Yellow
        'discontinuity': '#B279A2',# Purple
        'derivatives': '#FF9DA6',  # Pink
        'momentum': '#9D755D',     # Brown
    }
    
    axes = axes or list(TRAIT_COLORS.keys())
    scores = {a: row.get(a, 0) for a in axes if a in row.index}
    
    if not scores:
        return '#808080'  # Gray fallback
    
    dominant = max(scores, key=scores.get)
    return TRAIT_COLORS.get(dominant, '#808080')


def color_from_hsl(row: pd.Series,
                   h_axis: str = 'frequency',
                   s_axis: str = 'distinctiveness',
                   l_axis: str = 'volatility') -> str:
    """
    HSL color mapping for perceptually uniform variation.
    
    H = hue (which trait dominates)
    S = saturation (how distinct)
    L = lightness (secondary characteristic)
    """
    h = row.get(h_axis, 0.5)
    s = row.get(s_axis, 0.5) * 0.7 + 0.3  # 0.3-1.0 range
    l = row.get(l_axis, 0.5) * 0.4 + 0.3   # 0.3-0.7 range
    
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# -----------------------------------------------------------------------------
# Derived Metrics
# -----------------------------------------------------------------------------

AXES = ['memory', 'information', 'frequency', 'volatility', 'dynamics',
        'recurrence', 'discontinuity', 'derivatives', 'momentum']


def compute_distinctiveness(row: pd.Series, axes: list = None) -> float:
    """
    How distinct is this signal from neutral (0.5)?
    
    Range: 0 (all axes at 0.5) to 0.5 (all axes at 0 or 1)
    """
    axes = axes or AXES
    deviations = [abs(row.get(a, 0.5) - 0.5) for a in axes if a in row.index]
    if not deviations:
        return 0.0
    return sum(deviations) / len(deviations)


def compute_dominant_trait(row: pd.Series, axes: list = None) -> str:
    """
    Which axis has the strongest signal (furthest from 0.5)?
    """
    axes = axes or AXES
    scores = {a: abs(row.get(a, 0.5) - 0.5) for a in axes if a in row.index}
    
    if not scores:
        return 'unknown'
    
    return max(scores, key=scores.get)


def compute_dominant_pole(row: pd.Series, axes: list = None) -> tuple:
    """
    Which axis and which pole (low/high) is strongest?
    
    Returns: (axis_name, 'low' or 'high', score)
    """
    axes = axes or AXES
    
    best_axis = None
    best_pole = None
    best_distance = 0
    
    for a in axes:
        if a not in row.index:
            continue
        val = row.get(a, 0.5)
        dist = abs(val - 0.5)
        if dist > best_distance:
            best_distance = dist
            best_axis = a
            best_pole = 'high' if val > 0.5 else 'low'
    
    return (best_axis, best_pole, best_distance)


def compute_typology_vector(row: pd.Series, axes: list = None) -> np.ndarray:
    """
    Extract typology as numpy vector for ML/clustering.
    """
    axes = axes or AXES
    return np.array([row.get(a, 0.5) for a in axes])


# -----------------------------------------------------------------------------
# DataFrame Operations
# -----------------------------------------------------------------------------

def add_visual_columns(df: pd.DataFrame, 
                       color_method: str = 'dominant') -> pd.DataFrame:
    """
    Add visual encoding columns to typology dataframe.
    
    Args:
        df: Typology dataframe with axis scores
        color_method: 'dominant', 'rgb', or 'hsl'
    
    Returns:
        DataFrame with added columns:
        - color_hex
        - distinctiveness
        - dominant_trait
    """
    df = df.copy()
    
    # Distinctiveness
    df['distinctiveness'] = df.apply(compute_distinctiveness, axis=1)
    
    # Dominant trait
    df['dominant_trait'] = df.apply(compute_dominant_trait, axis=1)
    
    # Color
    if color_method == 'dominant':
        df['color_hex'] = df.apply(color_from_dominant, axis=1)
    elif color_method == 'rgb':
        df['color_hex'] = df.apply(color_direct_rgb, axis=1)
    elif color_method == 'hsl':
        df['color_hex'] = df.apply(color_from_hsl, axis=1)
    else:
        df['color_hex'] = '#808080'
    
    return df


def add_projections(df: pd.DataFrame, 
                    method: str = 'umap',
                    axes: list = None) -> pd.DataFrame:
    """
    Add 2D projection coordinates for scatter plots.
    
    Args:
        df: Typology dataframe
        method: 'umap', 'tsne', or 'pca'
        axes: Which axes to use for projection
    
    Returns:
        DataFrame with proj_x, proj_y columns
    """
    axes = axes or AXES
    available_axes = [a for a in axes if a in df.columns]
    
    if len(available_axes) < 2:
        df['proj_x'] = 0
        df['proj_y'] = 0
        return df
    
    X = df[available_axes].fillna(0.5).values
    
    if len(X) < 5:
        # Not enough points for projection
        df['proj_x'] = X[:, 0] if X.shape[1] > 0 else 0
        df['proj_y'] = X[:, 1] if X.shape[1] > 1 else 0
        return df
    
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1))
            embedding = reducer.fit_transform(X)
        except ImportError:
            method = 'pca'  # Fallback
    
    if method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, len(X) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedding = tsne.fit_transform(X)
        except ImportError:
            method = 'pca'  # Fallback
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(X)
    
    df = df.copy()
    df['proj_x'] = embedding[:, 0]
    df['proj_y'] = embedding[:, 1]
    
    return df


def add_clusters(df: pd.DataFrame,
                 n_clusters: int = None,
                 axes: list = None) -> pd.DataFrame:
    """
    Add cluster assignments.
    
    Args:
        df: Typology dataframe
        n_clusters: Number of clusters (auto-detect if None)
        axes: Which axes to cluster on
    
    Returns:
        DataFrame with cluster column
    """
    from sklearn.cluster import KMeans
    
    axes = axes or AXES
    available_axes = [a for a in axes if a in df.columns]
    
    if len(available_axes) < 2:
        df['cluster'] = 0
        return df
    
    X = df[available_axes].fillna(0.5).values
    
    if len(X) < 3:
        df['cluster'] = 0
        return df
    
    # Auto-detect clusters
    if n_clusters is None:
        from sklearn.metrics import silhouette_score
        best_k, best_score = 2, -1
        for k in range(2, min(8, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score
        n_clusters = best_k
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X)
    
    return df


# -----------------------------------------------------------------------------
# SQL Generation (for DuckDB in Streamlit)
# -----------------------------------------------------------------------------

def sql_visual_columns() -> str:
    """
    Generate SQL for visual encoding columns.
    Use in DuckDB queries within Streamlit.
    """
    return """
    -- Color (volatility=R, frequency=G, memory=B)
    '#' || printf('%02x%02x%02x', 
        CAST(COALESCE(volatility, 0.5) * 255 AS INT),
        CAST(COALESCE(frequency, 0.5) * 255 AS INT),
        CAST(COALESCE(memory, 0.5) * 255 AS INT)
    ) as color_hex,
    
    -- Distinctiveness (mean absolute deviation from 0.5)
    (
        ABS(COALESCE(memory, 0.5) - 0.5) +
        ABS(COALESCE(information, 0.5) - 0.5) +
        ABS(COALESCE(frequency, 0.5) - 0.5) +
        ABS(COALESCE(volatility, 0.5) - 0.5) +
        ABS(COALESCE(dynamics, 0.5) - 0.5) +
        ABS(COALESCE(recurrence, 0.5) - 0.5) +
        ABS(COALESCE(discontinuity, 0.5) - 0.5) +
        ABS(COALESCE(derivatives, 0.5) - 0.5) +
        ABS(COALESCE(momentum, 0.5) - 0.5)
    ) / 9.0 as distinctiveness
    """


def sql_with_visuals(base_table: str = 'signal_typology') -> str:
    """
    Full SQL query with visual columns.
    """
    return f"""
    SELECT 
        *,
        {sql_visual_columns()}
    FROM '{base_table}.parquet'
    """


# -----------------------------------------------------------------------------
# Streamlit Helpers
# -----------------------------------------------------------------------------

def get_color_scale(n: int, palette: str = 'viridis') -> list:
    """
    Get n colors from a palette for cluster coloring.
    """
    import plotly.express as px
    
    if palette == 'viridis':
        colors = px.colors.sequential.Viridis
    elif palette == 'set2':
        colors = px.colors.qualitative.Set2
    elif palette == 'paired':
        colors = px.colors.qualitative.Paired
    else:
        colors = px.colors.qualitative.Plotly
    
    # Cycle if needed
    return [colors[i % len(colors)] for i in range(n)]


def create_legend_data(df: pd.DataFrame) -> dict:
    """
    Create legend data for visual encoding explanation.
    """
    return {
        'color_method': 'RGB: R=volatility, G=frequency, B=memory',
        'distinctiveness_range': f'{df["distinctiveness"].min():.2f} - {df["distinctiveness"].max():.2f}',
        'dominant_traits': df['dominant_trait'].value_counts().to_dict(),
        'n_clusters': df['cluster'].nunique() if 'cluster' in df.columns else None,
    }
