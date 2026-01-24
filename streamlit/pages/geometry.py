"""
Geometry Screen
===============

Structural relationships. How do signals relate in space?

Tabs:
- Correlation: Correlation matrix and pairs
- Distance: Pairwise distances
- Network: Graph visualization
- 3D Phase: Phase space attractor
- Stability: Regime detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from components.charts import heatmap, scatter_3d, network_graph, line_chart
from components.tables import correlation_table, stability_events_table, render_table


def render(
    signals_df: pd.DataFrame,
    geometry_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    data_dir: Path = None,
):
    """
    Render the Geometry screen.

    Args:
        signals_df: Raw signal data
        geometry_df: Optional pre-computed geometry data
        profile_df: Optional typology profiles
        data_dir: Data directory path
    """
    # Horizontal sub-menu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Correlation", "Distance", "Network", "3D Phase", "Stability"
    ])

    signal_ids = sorted(signals_df['signal_id'].unique().tolist())
    n_signals = len(signal_ids)

    # ---------------------------------------------------------------------
    # Tab 1: Correlation Matrix
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**STRUCTURAL GEOMETRY:** Correlation Matrix")

        # Pivot signals to wide format for correlation
        if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
            try:
                wide_df = signals_df.pivot(
                    index='timestamp',
                    columns='signal_id',
                    values='value'
                )
                corr_matrix = wide_df.corr().values
                labels = wide_df.columns.tolist()

                # Filter controls
                col1, col2 = st.columns([1, 3])
                with col1:
                    threshold = st.slider(
                        "Min |correlation|",
                        0.0, 1.0, 0.5,
                        key="corr_threshold"
                    )

                # Heatmap
                fig = heatmap(
                    corr_matrix,
                    labels,
                    labels,
                    colorscale='RdBu_r',
                    symmetric=True,
                    height=max(400, n_signals * 30),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Correlation pairs table
                st.markdown("**Significant Pairs**")
                pairs_df = correlation_table(corr_matrix, labels, threshold=threshold)
                render_table(pairs_df)

            except Exception as e:
                st.error(f"Could not compute correlations: {e}")
        else:
            st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 2: Distance Matrix
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Pairwise Distance Matrix**")

        if profile_df is not None:
            # Use typology profiles for distance
            axes = [c for c in profile_df.columns if c not in ['signal_id', 'entity_id']]

            if axes:
                X = profile_df[axes].fillna(0.5).values
                labels = profile_df['signal_id'].tolist()

                # Compute Euclidean distance
                from scipy.spatial.distance import pdist, squareform
                dist_matrix = squareform(pdist(X, metric='euclidean'))

                # Distance method selector
                method = st.selectbox(
                    "Distance Metric",
                    ["Euclidean", "Cosine", "Manhattan"],
                    key="dist_method"
                )

                if method == "Cosine":
                    dist_matrix = squareform(pdist(X, metric='cosine'))
                elif method == "Manhattan":
                    dist_matrix = squareform(pdist(X, metric='cityblock'))

                fig = heatmap(
                    dist_matrix,
                    labels,
                    labels,
                    colorscale='Viridis',
                    symmetric=False,
                    height=max(400, n_signals * 30),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No typology axes found for distance computation.")
        else:
            st.info("Load typology profiles for distance analysis.")

    # ---------------------------------------------------------------------
    # Tab 3: Network Graph
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**Correlation Network**")

        if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
            try:
                wide_df = signals_df.pivot(
                    index='timestamp',
                    columns='signal_id',
                    values='value'
                )
                corr_matrix = wide_df.corr().values
                labels = wide_df.columns.tolist()

                # Threshold for edges
                edge_threshold = st.slider(
                    "Edge threshold (|correlation|)",
                    0.3, 0.9, 0.6,
                    key="network_threshold"
                )

                # Build edges
                edges = []
                n = len(labels)
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(corr_matrix[i, j]) >= edge_threshold:
                            edges.append({
                                'source': labels[i],
                                'target': labels[j],
                                'weight': corr_matrix[i, j],
                            })

                if edges:
                    fig = network_graph(edges, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(f"{len(edges)} edges above threshold")
                else:
                    st.info("No edges above threshold. Try lowering the threshold.")

            except Exception as e:
                st.error(f"Could not build network: {e}")
        else:
            st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 4: 3D Phase Space
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Phase Space Attractor**")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_signal = st.selectbox(
                "Select Signal",
                signal_ids,
                key="phase_signal"
            )

        with col2:
            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [3, 4, 5],
                key="embedding_dim"
            )

        with col3:
            delay = st.number_input(
                "Time Delay",
                1, 50, 10,
                key="time_delay"
            )

        # Get signal data
        sig_data = signals_df[signals_df['signal_id'] == selected_signal]['value'].values

        if len(sig_data) > embedding_dim * delay:
            # Create time-delay embedding
            n = len(sig_data) - (embedding_dim - 1) * delay

            embedded = np.zeros((n, embedding_dim))
            for d in range(embedding_dim):
                embedded[:, d] = sig_data[d * delay:d * delay + n]

            # Create DataFrame for plotting
            embed_df = pd.DataFrame({
                'x': embedded[:, 0],
                'y': embedded[:, 1],
                'z': embedded[:, 2],
                't': np.arange(n),
            })

            fig = scatter_3d(
                embed_df,
                x='x',
                y='y',
                z='z',
                color='t',
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics placeholder
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Embedding Dim", embedding_dim)
            col2.metric("Delay", delay)
            col3.metric("Points", n)
        else:
            st.warning("Signal too short for selected embedding parameters.")

    # ---------------------------------------------------------------------
    # Tab 5: Stability Analysis
    # ---------------------------------------------------------------------
    with tab5:
        st.markdown("**Stability & Regime Detection**")

        st.info("Stability analysis requires pre-computed dynamical systems data.")

        # Placeholder for stability visualization
        st.markdown("**Stability Timeline**")

        # Create synthetic example
        n_points = 100
        t = np.arange(n_points)
        stability = np.ones(n_points) * 0.8
        stability[40:60] = np.linspace(0.8, 0.2, 20)
        stability[60:80] = 0.2 + np.random.randn(20) * 0.1

        example_df = pd.DataFrame({'t': t, 'stability': stability})

        fig = line_chart(
            example_df,
            x='t',
            y='stability',
            height=250,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Example stability timeline (synthetic data)")

        st.markdown("---")
        st.markdown("**Detected Transitions**")

        # Example events
        events = [
            {'time': 40, 'from_state': 'Stable', 'to_state': 'Unstable', 'duration': 40, 'confidence': 0.89},
            {'time': 80, 'from_state': 'Unstable', 'to_state': 'Stable', 'duration': '-', 'confidence': 0.92},
        ]

        events_df = stability_events_table(events)
        render_table(events_df)
