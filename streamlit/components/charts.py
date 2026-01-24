"""
Reusable chart components for ORTHON dashboard.
All charts use Plotly for consistency.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


# -----------------------------------------------------------------------------
# Color Palettes
# -----------------------------------------------------------------------------

TRAIT_COLORS = {
    'memory': '#4C78A8',
    'information': '#F58518',
    'frequency': '#E45756',
    'volatility': '#72B7B2',
    'dynamics': '#54A24B',
    'recurrence': '#EECA3B',
    'discontinuity': '#B279A2',
    'derivatives': '#FF9DA6',
    'momentum': '#9D755D',
}

DEFAULT_COLORS = list(TRAIT_COLORS.values())


# -----------------------------------------------------------------------------
# Line Charts
# -----------------------------------------------------------------------------

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = None,
    height: int = 300,
    show_legend: bool = True,
) -> go.Figure:
    """
    Basic line chart for time series.

    Args:
        df: DataFrame with data
        x: Column for x-axis (usually time/timestamp)
        y: Column for y-axis (values)
        color: Optional column for grouping/coloring lines
        title: Optional chart title
        height: Chart height in pixels
    """
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=30 if title else 20, b=40),
        hovermode='x unified',
        showlegend=show_legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return fig


def multi_line_chart(
    df: pd.DataFrame,
    x: str,
    y_columns: List[str],
    colors: List[str] = None,
    title: str = None,
    height: int = 300,
) -> go.Figure:
    """
    Multiple lines on same chart.

    Args:
        df: DataFrame with data
        x: Column for x-axis
        y_columns: List of columns to plot as separate lines
        colors: Optional list of colors for each line
        title: Optional chart title
    """
    colors = colors or DEFAULT_COLORS

    fig = go.Figure()

    for i, col in enumerate(y_columns):
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=30 if title else 20, b=40),
        title=title,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return fig


# -----------------------------------------------------------------------------
# Sparklines (Small Multiples)
# -----------------------------------------------------------------------------

def sparklines(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str,
    value_col: str,
    n_cols: int = 5,
    height_per_row: int = 60,
) -> go.Figure:
    """
    Small multiples showing all signals as mini line charts.

    Args:
        df: DataFrame with time series data
        signal_col: Column with signal identifiers
        time_col: Column with time values
        value_col: Column with signal values
        n_cols: Number of columns in grid
        height_per_row: Height per row in pixels
    """
    signals = df[signal_col].unique()
    n_signals = len(signals)
    n_rows = (n_signals + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[str(s) for s in signals],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for i, sig in enumerate(signals):
        row = i // n_cols + 1
        col = i % n_cols + 1

        sig_data = df[df[signal_col] == sig].sort_values(time_col)

        fig.add_trace(
            go.Scatter(
                x=sig_data[time_col],
                y=sig_data[value_col],
                mode='lines',
                line=dict(width=1, color='#4C78A8'),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=height_per_row * n_rows,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Hide axes for cleaner look
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    return fig


# -----------------------------------------------------------------------------
# Radar Charts
# -----------------------------------------------------------------------------

def radar_chart(
    values: List[float],
    axes: List[str],
    name: str = None,
    color: str = '#4C78A8',
    fill_opacity: float = 0.3,
    height: int = 400,
) -> go.Figure:
    """
    Single radar chart for typology profile.

    Args:
        values: List of values (0-1 normalized)
        axes: List of axis names
        name: Optional trace name
        color: Line/fill color
        fill_opacity: Opacity of fill
    """
    # Close the polygon
    values = list(values) + [values[0]]
    axes = list(axes) + [axes[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=axes,
        fill='toself',
        name=name,
        line_color=color,
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {fill_opacity})',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
            )
        ),
        showlegend=False,
        height=height,
        margin=dict(l=40, r=40, t=20, b=20),
    )

    return fig


def radar_compare(
    profiles: List[Dict[str, Any]],
    axes: List[str],
    height: int = 400,
) -> go.Figure:
    """
    Overlaid radar chart comparing multiple signals.

    Args:
        profiles: List of dicts with 'name', 'values', and optional 'color'
        axes: List of axis names
    """
    fig = go.Figure()

    colors = DEFAULT_COLORS

    for i, profile in enumerate(profiles):
        values = list(profile['values']) + [profile['values'][0]]
        theta = list(axes) + [axes[0]]
        color = profile.get('color', colors[i % len(colors)])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=profile['name'],
            line_color=color,
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        height=height,
        margin=dict(l=40, r=40, t=20, b=20),
    )

    return fig


# -----------------------------------------------------------------------------
# Heatmaps
# -----------------------------------------------------------------------------

def heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str = None,
    colorscale: str = 'RdBu_r',
    symmetric: bool = True,
    height: int = 400,
    show_values: bool = True,
) -> go.Figure:
    """
    Heatmap for correlation or transfer entropy matrices.

    Args:
        matrix: 2D numpy array
        x_labels: Column labels
        y_labels: Row labels
        title: Optional title
        colorscale: Plotly colorscale name
        symmetric: If True, use symmetric color range around 0
        height: Chart height
        show_values: Show values in cells
    """
    if symmetric:
        max_val = np.abs(matrix).max()
        zmin, zmax = -max_val, max_val
    else:
        zmin, zmax = matrix.min(), matrix.max()

    text = [[f'{v:.2f}' for v in row] for row in matrix] if show_values else None

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{y} → %{x}: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 20, b=60),
        xaxis=dict(side='bottom'),
    )

    return fig


def typology_heatmap(
    df: pd.DataFrame,
    signal_col: str,
    axes: List[str],
    height: int = 400,
) -> go.Figure:
    """
    Heatmap showing typology scores for all signals across axes.

    Args:
        df: DataFrame with signal typology data
        signal_col: Column with signal identifiers
        axes: List of typology axis columns
    """
    signals = df[signal_col].tolist()
    matrix = df[axes].values

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=axes,
        y=signals,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        text=[[f'{v:.2f}' for v in row] for row in matrix],
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{y}<br>%{x}: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=100, r=20, t=20, b=60),
        xaxis=dict(side='bottom', tickangle=45),
    )

    return fig


# -----------------------------------------------------------------------------
# Scatter Plots
# -----------------------------------------------------------------------------

def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    size: str = None,
    hover_name: str = None,
    color_map: Dict[str, str] = None,
    title: str = None,
    height: int = 400,
) -> go.Figure:
    """
    2D scatter plot for projections (UMAP, PCA, etc.).

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        color: Column for color encoding
        size: Column for size encoding
        hover_name: Column for hover labels
        color_map: Dict mapping color values to hex colors
        title: Optional title
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        color_discrete_map=color_map or TRAIT_COLORS,
        title=title,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=40),
        xaxis_title='',
        yaxis_title='',
    )

    return fig


def scatter_3d(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color: str = None,
    hover_name: str = None,
    title: str = None,
    height: int = 500,
) -> go.Figure:
    """
    3D scatter plot for phase space visualization.

    Args:
        df: DataFrame with data
        x, y, z: Axis columns
        color: Column for color encoding (e.g., time)
        hover_name: Column for hover labels
        title: Optional title
    """
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color,
        hover_name=hover_name,
        title=title,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
    )

    return fig


# -----------------------------------------------------------------------------
# Bar Charts
# -----------------------------------------------------------------------------

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    orientation: str = 'v',
    title: str = None,
    height: int = 300,
) -> go.Figure:
    """
    Bar chart for distributions.

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        color: Optional color column
        orientation: 'v' for vertical, 'h' for horizontal
        title: Optional title
    """
    fig = px.bar(
        df,
        x=x if orientation == 'v' else y,
        y=y if orientation == 'v' else x,
        color=color,
        orientation=orientation,
        title=title,
        color_discrete_map=TRAIT_COLORS if color else None,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=40),
        showlegend=False,
    )

    return fig


def distribution_bars(
    counts: Dict[str, int],
    colors: Dict[str, str] = None,
    orientation: str = 'h',
    height: int = 200,
) -> go.Figure:
    """
    Simple bar chart from a dict of counts.

    Args:
        counts: Dict mapping labels to counts
        colors: Optional dict mapping labels to colors
        orientation: 'v' or 'h'
    """
    labels = list(counts.keys())
    values = list(counts.values())
    bar_colors = [colors.get(l, '#4C78A8') for l in labels] if colors else None

    if orientation == 'h':
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=bar_colors,
        ))
    else:
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=bar_colors,
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=80, r=20, t=20, b=40),
        showlegend=False,
    )

    return fig


# -----------------------------------------------------------------------------
# Network Graphs
# -----------------------------------------------------------------------------

def network_graph(
    edges: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]] = None,
    height: int = 400,
    title: str = None,
) -> go.Figure:
    """
    Network graph for causal relationships.

    Args:
        edges: List of dicts with 'source', 'target', 'weight'
        nodes: Optional list of dicts with 'id', 'x', 'y', 'size', 'color'
        height: Chart height
        title: Optional title

    Note: For complex networks, consider using pyvis or networkx with plotly.
    """
    # Simple force-directed layout simulation
    import networkx as nx

    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))

    # Get positions
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Edge traces
    edge_x, edge_y = [], []
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none',
    )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=20, color='#4C78A8'),
        text=node_text,
        textposition='top center',
        hoverinfo='text',
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig
