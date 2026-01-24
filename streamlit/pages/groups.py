"""
Groups Screen
=============

Cluster similar signals. Which signals behave alike?

Tabs:
- Auto-Detect: Automatic clustering
- Custom Groups: Manual grouping
- Compare: Group comparison
- Export: Download group assignments
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict

from components.charts import (
    scatter_2d, radar_compare, distribution_bars
)
from components.tables import render_table
from components.metrics import group_summary_card


DEFAULT_AXES = [
    'memory', 'information', 'frequency', 'volatility', 'dynamics',
    'recurrence', 'discontinuity', 'derivatives', 'momentum'
]

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


def render(
    profile_df: pd.DataFrame,
    axes: List[str] = None,
    classify_fn: Callable = None,
):
    """
    Render the Groups screen.

    Args:
        profile_df: DataFrame with typology scores
        axes: List of axis names
        classify_fn: Function(score, axis) -> label
    """
    axes = axes or DEFAULT_AXES
    available_axes = [a for a in axes if a in profile_df.columns]

    if len(available_axes) < 2:
        st.warning("Need at least 2 typology axes for clustering.")
        return

    # Horizontal sub-menu
    tab1, tab2, tab3, tab4 = st.tabs([
        "Auto-Detect", "Custom Groups", "Compare", "Export"
    ])

    # ---------------------------------------------------------------------
    # Tab 1: Auto-Detect Clusters
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**Automatic Group Detection**")

        col1, col2 = st.columns([1, 3])

        with col1:
            n_clusters = st.slider(
                "Number of groups",
                min_value=2,
                max_value=min(8, len(profile_df) - 1),
                value=3,
                key="n_clusters"
            )

            projection_method = st.selectbox(
                "Projection",
                ["PCA", "UMAP", "t-SNE"],
                key="proj_method"
            )

        # Perform clustering
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = profile_df[available_axes].fillna(0.5).values

        if len(X) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            if len(set(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = 0

            profile_df = profile_df.copy()
            profile_df['cluster'] = labels

            # Compute projections
            if projection_method == "PCA":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X)
            elif projection_method == "UMAP":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1))
                    coords = reducer.fit_transform(X)
                except ImportError:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(X)
                    st.warning("UMAP not available, using PCA")
            else:  # t-SNE
                from sklearn.manifold import TSNE
                perplexity = min(30, len(X) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                coords = tsne.fit_transform(X)

            profile_df['proj_x'] = coords[:, 0]
            profile_df['proj_y'] = coords[:, 1]

            # Compute dominant trait
            profile_df['dominant_trait'] = profile_df.apply(
                lambda row: max(available_axes, key=lambda a: row.get(a, 0)),
                axis=1
            )

            # Compute distinctiveness
            profile_df['distinctiveness'] = profile_df[available_axes].apply(
                lambda row: np.mean(np.abs(row - 0.5)),
                axis=1
            )

        with col2:
            st.markdown(f"**GROUPS:** {n_clusters} detected (silhouette: {silhouette:.2f})")

        # Two-column layout for scatter and summary
        col1, col2 = st.columns([3, 2])

        with col1:
            # Scatter plot
            cluster_colors = {i: f'hsl({i * 360 // n_clusters}, 70%, 50%)'
                              for i in range(n_clusters)}

            fig = scatter_2d(
                profile_df,
                x='proj_x',
                y='proj_y',
                color='cluster',
                size='distinctiveness',
                hover_name='signal_id',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Group Summary**")

            for cluster_id in range(n_clusters):
                cluster_members = profile_df[profile_df['cluster'] == cluster_id]
                n_members = len(cluster_members)

                # Get dominant traits in this cluster
                traits = cluster_members['dominant_trait'].value_counts()
                top_traits = traits.head(3).index.tolist()

                with st.expander(f"Group {cluster_id + 1} ({n_members} signals)", expanded=cluster_id == 0):
                    st.caption(f"Dominant: {', '.join(top_traits)}")

                    # Show member signals
                    members = cluster_members['signal_id'].tolist()
                    st.text(f"Members: {', '.join(members[:5])}")
                    if len(members) > 5:
                        st.text(f"  + {len(members) - 5} more")

        st.markdown("---")

        # Membership table
        st.markdown("**Group Membership**")
        membership_df = profile_df[['signal_id', 'cluster', 'dominant_trait', 'distinctiveness']].copy()
        membership_df.columns = ['Signal', 'Group', 'Dominant', 'Distinctiveness']
        membership_df['Group'] = membership_df['Group'] + 1
        membership_df['Distinctiveness'] = membership_df['Distinctiveness'].apply(lambda x: f'{x:.3f}')
        render_table(membership_df)

    # ---------------------------------------------------------------------
    # Tab 2: Custom Groups
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Define Custom Groups**")

        st.info("Manual group definition is not yet implemented. Use Auto-Detect for now.")

        # TODO: Implement custom group interface
        # - Allow user to select signals and assign to groups
        # - Save group definitions
        # - Load previously saved groups

    # ---------------------------------------------------------------------
    # Tab 3: Compare Groups
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**Compare Groups**")

        if 'cluster' not in profile_df.columns:
            st.info("Run Auto-Detect first to create groups.")
        else:
            clusters = sorted(profile_df['cluster'].unique())

            col1, col2 = st.columns(2)

            with col1:
                group_a = st.selectbox(
                    "Group A",
                    clusters,
                    format_func=lambda x: f"Group {x + 1}",
                    key="compare_group_a"
                )

            with col2:
                group_b = st.selectbox(
                    "Group B",
                    [c for c in clusters if c != group_a],
                    format_func=lambda x: f"Group {x + 1}",
                    key="compare_group_b"
                )

            # Get group averages
            group_a_data = profile_df[profile_df['cluster'] == group_a][available_axes].mean()
            group_b_data = profile_df[profile_df['cluster'] == group_b][available_axes].mean()

            # Radar comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Group {group_a + 1} Average**")
                fig = radar_compare(
                    [{'name': f'Group {group_a + 1}', 'values': group_a_data.tolist()}],
                    available_axes,
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"**Group {group_b + 1} Average**")
                fig = radar_compare(
                    [{'name': f'Group {group_b + 1}', 'values': group_b_data.tolist()}],
                    available_axes,
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Difference table
            st.markdown("**Axis Differences**")

            diff_data = []
            for axis in available_axes:
                val_a = group_a_data[axis]
                val_b = group_b_data[axis]
                delta = val_a - val_b

                # Significance (simplified - just based on magnitude)
                significant = abs(delta) > 0.15

                diff_data.append({
                    'Axis': axis.title(),
                    f'Group {group_a + 1}': f'{val_a:.3f}',
                    f'Group {group_b + 1}': f'{val_b:.3f}',
                    'Δ': f'{delta:+.3f}',
                    'Significant?': 'YES' if significant else 'no',
                })

            diff_df = pd.DataFrame(diff_data)
            render_table(diff_df)

    # ---------------------------------------------------------------------
    # Tab 4: Export
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Export Group Assignments**")

        if 'cluster' not in profile_df.columns:
            st.info("Run Auto-Detect first to create groups.")
        else:
            export_df = profile_df[['signal_id', 'cluster'] + available_axes].copy()
            export_df['cluster'] = export_df['cluster'] + 1
            export_df.columns = ['signal_id', 'group'] + available_axes

            st.dataframe(export_df, hide_index=True)

            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="signal_groups.csv",
                mime="text/csv",
            )
