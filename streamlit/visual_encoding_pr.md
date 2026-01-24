# PR: Visual Encoding Layer

## Summary

Defines how framework outputs (vectors) are transformed into visual representations. Each framework produces a vector; the visual encoding layer maps these to colors, shapes, positions, and graphs.

---

## Core Principle

**Framework → Vector → Visual Encoding → Render**

Every framework output can be:
1. Displayed as raw numbers (table)
2. Encoded as a visual fingerprint (glyph, color)
3. Projected into 2D/3D space (scatter, map)
4. Rendered as relationships (graph, heatmap)

---

## Framework Vector Definitions

### signal_typology

```python
S = [memory, information, frequency, volatility, wavelet, 
     derivatives, recurrence, discontinuity, momentum]

# 9-dimensional vector, all values 0-1
# One vector per signal
```

### structural_geometry

```python
G = [attractor_dimension, density_modes, boundary_range, 
     trajectory_complexity, embedding_dimension, ...]

# Variable dimensions based on engines run
# One vector per signal
```

### dynamical_systems

```python
D = {
    'coherence_matrix': NxN,      # Signal pair coherences
    'sync_matrix': NxN,           # Phase synchronization
    'stability_vector': [N],      # Per-signal stability
    'system_stability': float,    # Aggregate
}

# Matrices for signal relationships
# One result set per system (all signals together)
```

### causal_mechanics

```python
C = {
    'causal_matrix': NxN,         # Directed causal strength
    'information_flow': NxN,      # Transfer entropy
    'lead_lag_matrix': NxN,       # Temporal relationships
    'drivers': [signal_ids],      # Identified drivers
}

# Directed matrices
# One result set per system
```

---

## Visual Encoding Schema

```python
# prism/visualization/encoding.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SignalVisualEncoding:
    """Visual encoding for a single signal."""
    
    signal_id: str
    
    # Color signature (from typology)
    color_rgb: tuple[int, int, int]
    color_hex: str
    
    # 2D projection coordinates (for scatter/map views)
    proj_x: float
    proj_y: float
    
    # Glyph parameters
    glyph_shape: str      # 'circle', 'square', 'triangle', 'star', 'hexagon'
    glyph_size: float     # Normalized 0-1
    glyph_rotation: float # Degrees
    
    # Radar chart data
    radar_values: list[float]
    radar_labels: list[str]
    
    # Classification summary
    dominant_trait: str   # Highest-scoring axis
    trait_strength: float # Score of dominant trait


@dataclass 
class SystemVisualEncoding:
    """Visual encoding for system-level (multi-signal) results."""
    
    # Node positions (for graph layouts)
    node_positions: dict[str, tuple[float, float]]
    
    # Edge data (for relationship graphs)
    edges: list[dict]  # [{source, target, weight, direction}, ...]
    
    # Heatmap data
    matrix: np.ndarray
    row_labels: list[str]
    col_labels: list[str]
    
    # Cluster assignments
    clusters: dict[str, int]  # signal_id -> cluster_id
    cluster_colors: dict[int, str]  # cluster_id -> color
```

---

## Color Encoding Strategies

### Strategy 1: PCA → RGB

Map first 3 principal components of typology vector to RGB:

```python
def typology_to_color_pca(profiles: pd.DataFrame) -> dict[str, str]:
    """
    Map typology profiles to colors via PCA.
    Similar signals → similar colors.
    """
    from sklearn.decomposition import PCA
    
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    X = profiles[axes].values
    
    # Reduce 9D → 3D
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    
    # Normalize to 0-255
    mins = components.min(axis=0)
    maxs = components.max(axis=0)
    normalized = (components - mins) / (maxs - mins + 1e-8)
    rgb = (normalized * 255).astype(int)
    
    colors = {}
    for i, signal_id in enumerate(profiles['signal_id']):
        r, g, b = rgb[i]
        colors[signal_id] = f'#{r:02x}{g:02x}{b:02x}'
    
    return colors
```

### Strategy 2: Dominant Trait → Categorical Color

```python
TRAIT_COLORS = {
    'memory': '#4C78A8',      # Blue
    'information': '#F58518', # Orange  
    'frequency': '#E45756',   # Red
    'volatility': '#72B7B2',  # Teal
    'wavelet': '#54A24B',     # Green
    'derivatives': '#EECA3B', # Yellow
    'recurrence': '#B279A2',  # Purple
    'discontinuity': '#FF9DA6', # Pink
    'momentum': '#9D755D',    # Brown
}

def typology_to_color_dominant(profile: dict) -> str:
    """Color based on dominant trait."""
    axes = list(TRAIT_COLORS.keys())
    scores = [profile.get(a, 0) for a in axes]
    dominant = axes[np.argmax(scores)]
    return TRAIT_COLORS[dominant]
```

### Strategy 3: Direct Mapping (3 key axes)

```python
def typology_to_color_direct(profile: dict) -> str:
    """
    Direct mapping of 3 interpretable axes to RGB.
    
    R = volatility (red = unstable)
    G = frequency (green = periodic)
    B = memory (blue = persistent)
    """
    r = int(255 * profile.get('volatility', 0))
    g = int(255 * profile.get('frequency', 0))
    b = int(255 * profile.get('memory', 0))
    return f'#{r:02x}{g:02x}{b:02x}'
```

---

## Glyph Encoding

Map typology to shape properties:

```python
def typology_to_glyph(profile: dict) -> dict:
    """
    Encode typology as glyph parameters.
    
    Shape: dominant trait category
    Size: overall signal "strength" (mean absolute deviation from 0.5)
    Rotation: momentum direction
    """
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    scores = [profile.get(a, 0.5) for a in axes]
    
    # Shape from dominant trait
    dominant_idx = np.argmax(scores)
    shapes = ['circle', 'square', 'triangle', 'diamond', 'star',
              'hexagon', 'pentagon', 'cross', 'arrow']
    shape = shapes[dominant_idx]
    
    # Size from distinctiveness (how far from neutral)
    distinctiveness = np.mean([abs(s - 0.5) for s in scores])
    size = 0.3 + (distinctiveness * 1.4)  # Range 0.3-1.0
    
    # Rotation from momentum
    momentum = profile.get('momentum', 0.5)
    rotation = (momentum - 0.5) * 180  # -90 to +90 degrees
    
    return {
        'shape': shape,
        'size': size,
        'rotation': rotation,
    }
```

---

## 2D Projection Methods

### UMAP (recommended for cohort visualization)

```python
def project_typology_umap(profiles: pd.DataFrame) -> pd.DataFrame:
    """Project 9D typology vectors to 2D via UMAP."""
    import umap
    
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    X = profiles[axes].values
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)
    
    result = profiles[['signal_id']].copy()
    result['proj_x'] = embedding[:, 0]
    result['proj_y'] = embedding[:, 1]
    
    return result
```

### t-SNE (alternative)

```python
def project_typology_tsne(profiles: pd.DataFrame) -> pd.DataFrame:
    """Project 9D typology vectors to 2D via t-SNE."""
    from sklearn.manifold import TSNE
    
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    X = profiles[axes].values
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    embedding = tsne.fit_transform(X)
    
    result = profiles[['signal_id']].copy()
    result['proj_x'] = embedding[:, 0]
    result['proj_y'] = embedding[:, 1]
    
    return result
```

---

## Relationship Visuals

### Coherence Heatmap

```python
def dynamics_to_heatmap(dynamics_result: dict) -> dict:
    """Prepare coherence matrix for heatmap rendering."""
    matrix = dynamics_result['coherence_matrix']
    labels = dynamics_result['signal_ids']
    
    return {
        'z': matrix.tolist(),
        'x': labels,
        'y': labels,
        'colorscale': 'RdBu',
        'zmid': 0.5,
    }
```

### Causal Network Graph

```python
def causality_to_graph(causal_result: dict, threshold: float = 0.1) -> dict:
    """
    Convert causal matrix to node/edge graph data.
    
    Returns format suitable for networkx, plotly, or cytoscape.
    """
    matrix = causal_result['causal_matrix']
    labels = causal_result['signal_ids']
    n = len(labels)
    
    nodes = [{'id': label, 'label': label} for label in labels]
    
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > threshold:
                edges.append({
                    'source': labels[i],
                    'target': labels[j],
                    'weight': float(matrix[i, j]),
                    'direction': 'directed',
                })
    
    return {'nodes': nodes, 'edges': edges}
```

---

## Aggregation Functions

### Cohort Summary

```python
def aggregate_cohort_typology(profiles: pd.DataFrame) -> dict:
    """
    Aggregate multiple signal profiles into cohort summary.
    """
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    return {
        'mean': {a: profiles[a].mean() for a in axes},
        'std': {a: profiles[a].std() for a in axes},
        'min': {a: profiles[a].min() for a in axes},
        'max': {a: profiles[a].max() for a in axes},
        'n_signals': len(profiles),
    }
```

### Cluster Detection

```python
def cluster_signals(profiles: pd.DataFrame, n_clusters: int = None) -> dict:
    """
    Cluster signals by typology similarity.
    
    Auto-detects n_clusters if not specified.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    axes = ['memory', 'information', 'frequency', 'volatility',
            'wavelet', 'derivatives', 'recurrence', 'discontinuity', 'momentum']
    
    X = profiles[axes].values
    
    # Auto-detect clusters
    if n_clusters is None:
        best_k, best_score = 2, -1
        for k in range(2, min(10, len(X))):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k
    
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    
    # Assign colors to clusters
    cluster_colors = {
        i: f'hsl({int(360 * i / n_clusters)}, 70%, 50%)'
        for i in range(n_clusters)
    }
    
    return {
        'assignments': dict(zip(profiles['signal_id'], labels)),
        'colors': cluster_colors,
        'n_clusters': n_clusters,
        'centroids': km.cluster_centers_.tolist(),
    }
```

---

## Streamlit Integration

```python
# prism/visualization/streamlit_components.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_signal_glyph(encoding: SignalVisualEncoding):
    """Render a signal as its visual glyph."""
    # Using plotly for custom shapes
    fig = go.Figure()
    
    # Add glyph (simplified - circle with color)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            size=encoding.glyph_size * 100,
            color=encoding.color_hex,
            symbol=encoding.glyph_shape,
        ),
        name=encoding.signal_id,
    ))
    
    fig.update_layout(
        showlegend=False,
        width=150, height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return fig


def render_cohort_map(profiles: pd.DataFrame, encodings: dict):
    """Render 2D projection of signal cohort."""
    
    fig = px.scatter(
        profiles,
        x='proj_x',
        y='proj_y',
        color='dominant_trait',
        hover_name='signal_id',
        size='distinctiveness',
    )
    
    fig.update_layout(
        title='Signal Cohort Map',
        xaxis_title='',
        yaxis_title='',
    )
    
    return fig


def render_coherence_heatmap(dynamics_result: dict):
    """Render system coherence as heatmap."""
    
    heatmap_data = dynamics_to_heatmap(dynamics_result)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data['z'],
        x=heatmap_data['x'],
        y=heatmap_data['y'],
        colorscale='RdBu',
        zmid=0.5,
    ))
    
    fig.update_layout(
        title='Signal Coherence Matrix',
        xaxis_title='Signal',
        yaxis_title='Signal',
    )
    
    return fig


def render_causal_graph(causal_result: dict):
    """Render causal relationships as directed graph."""
    import networkx as nx
    
    graph_data = causality_to_graph(causal_result)
    
    G = nx.DiGraph()
    for node in graph_data['nodes']:
        G.add_node(node['id'])
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    pos = nx.spring_layout(G)
    
    # Create plotly figure
    edge_traces = []
    for edge in graph_data['edges']:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge['weight'] * 5, color='#888'),
            hoverinfo='none',
        ))
    
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        marker=dict(size=20, color='#4C78A8'),
        text=list(G.nodes()),
        textposition='top center',
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='Causal Network',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return fig
```

---

## File Structure

```
prism/
  visualization/
    __init__.py
    encoding.py           # Visual encoding dataclasses & functions
    colors.py             # Color mapping strategies
    projection.py         # UMAP, t-SNE projections
    aggregation.py        # Cohort summaries, clustering
    streamlit_components.py  # Streamlit rendering functions
```

---

## Checklist for CC

- [ ] Create `prism/visualization/` directory
- [ ] Implement `encoding.py` with dataclasses
- [ ] Implement `colors.py` with 3 color strategies
- [ ] Implement `projection.py` with UMAP/t-SNE
- [ ] Implement `aggregation.py` with clustering
- [ ] Add visualization dependencies to requirements: `umap-learn`, `scikit-learn`
- [ ] Wire into Streamlit dashboard
- [ ] Test color encoding produces distinguishable colors
- [ ] Test projection produces meaningful clusters
