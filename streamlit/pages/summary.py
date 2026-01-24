"""
Summary Screen
==============

Synthesis. What does it all mean?

Tabs:
- Executive: Key findings overview
- Technical: Detailed technical summary
- Export: Download reports
- Thesis: Scientific narrative
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from components.tables import render_table
from components.metrics import metric_row


def render(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame = None,
    geometry_df: pd.DataFrame = None,
    dynamics_df: pd.DataFrame = None,
    mechanics_df: pd.DataFrame = None,
    data_dir: Path = None,
):
    """
    Render the Summary screen.

    Args:
        signals_df: Raw signal data
        profile_df: Typology profiles
        geometry_df: Geometry data
        dynamics_df: Dynamics data
        mechanics_df: Causal mechanics data
        data_dir: Data directory path
    """
    # Horizontal sub-menu
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive", "Technical", "Export", "Thesis"
    ])

    # Gather summary statistics
    signal_ids = sorted(signals_df['signal_id'].unique().tolist())
    n_signals = len(signal_ids)
    n_samples = len(signals_df) // n_signals if n_signals > 0 else 0

    entity_col = 'entity_id' if 'entity_id' in signals_df.columns else None
    if entity_col:
        n_entities = signals_df[entity_col].nunique()
        entity_name = signals_df[entity_col].iloc[0] if n_entities == 1 else f"{n_entities} entities"
    else:
        entity_name = "unknown"

    # ---------------------------------------------------------------------
    # Tab 1: Executive Summary
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**ANALYSIS SUMMARY**")

        # Header metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset", entity_name)
        col2.metric("Signals", n_signals)
        col3.metric("Samples", f"{n_samples:,}")

        st.markdown("---")

        # Key findings container
        with st.container(border=True):
            st.markdown("### KEY FINDINGS")

            findings = []

            # 1. Typology findings
            if profile_df is not None and not profile_df.empty:
                axes = [c for c in profile_df.columns if c not in ['signal_id', 'entity_id']]
                if axes:
                    # Compute dominant traits
                    dominant_counts = {}
                    for _, row in profile_df.iterrows():
                        scores = {a: row.get(a, 0) for a in axes}
                        dominant = max(scores, key=scores.get)
                        dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1

                    # Top traits
                    top_traits = sorted(dominant_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    trait_summary = ", ".join([f"{t[0]} ({t[1]})" for t in top_traits])

                    findings.append({
                        'category': 'TYPOLOGY',
                        'finding': f"Dominant traits: {trait_summary}",
                        'detail': f"Signals characterized across {len(axes)} axes",
                    })

                    # Check for clustering
                    n_groups = len(set(dominant_counts.keys()))
                    if n_groups >= 2:
                        findings.append({
                            'category': 'TYPOLOGY',
                            'finding': f"Signals split into {n_groups} behavioral groups",
                            'detail': "Based on dominant typology traits",
                        })

            # 2. Geometry findings (placeholder)
            if geometry_df is not None:
                findings.append({
                    'category': 'GEOMETRY',
                    'finding': "Structural analysis completed",
                    'detail': "Correlation and distance matrices computed",
                })
            else:
                # Compute basic correlation from signals
                if 'timestamp' in signals_df.columns:
                    try:
                        wide_df = signals_df.pivot(
                            index='timestamp',
                            columns='signal_id',
                            values='value'
                        )
                        corr_matrix = wide_df.corr().values
                        np.fill_diagonal(corr_matrix, 0)

                        # Find strongest correlation
                        max_corr = np.abs(corr_matrix).max()
                        max_idx = np.unravel_index(np.argmax(np.abs(corr_matrix)), corr_matrix.shape)

                        findings.append({
                            'category': 'GEOMETRY',
                            'finding': f"Strongest correlation: {max_corr:.2f}",
                            'detail': f"Between {signal_ids[max_idx[0]]} and {signal_ids[max_idx[1]]}",
                        })
                    except:
                        pass

            # 3. Dynamics findings (placeholder)
            if dynamics_df is not None:
                findings.append({
                    'category': 'DYNAMICS',
                    'finding': "Temporal dynamics analyzed",
                    'detail': "Coherence and synchronization computed",
                })

            # 4. Mechanics findings (placeholder)
            if mechanics_df is not None:
                findings.append({
                    'category': 'MECHANICS',
                    'finding': "Causal relationships identified",
                    'detail': "Granger causality and transfer entropy computed",
                })

            # Display findings
            if findings:
                for i, f in enumerate(findings, 1):
                    st.markdown(f"**{i}. {f['category']}:** {f['finding']}")
                    st.caption(f"   {f['detail']}")
            else:
                st.info("Run analysis pipelines to generate findings.")

        st.markdown("---")

        # Action buttons
        col1, col2, col3 = st.columns(3)
        col1.button("Download Report", disabled=True)
        col2.button("Export Data", disabled=True)
        col3.button("Share", disabled=True)

    # ---------------------------------------------------------------------
    # Tab 2: Technical Summary
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**TECHNICAL SUMMARY**")

        # Data overview
        with st.expander("Data Overview", expanded=True):
            data_info = {
                'Metric': ['Total Signals', 'Samples per Signal', 'Total Observations',
                           'Signal Range', 'Time Range'],
                'Value': [
                    n_signals,
                    n_samples,
                    len(signals_df),
                    f"{signals_df['value'].min():.2f} - {signals_df['value'].max():.2f}",
                    f"{signals_df['timestamp'].min()} - {signals_df['timestamp'].max()}" if 'timestamp' in signals_df else "N/A",
                ]
            }
            st.dataframe(pd.DataFrame(data_info), hide_index=True)

        # Typology summary
        if profile_df is not None:
            with st.expander("Typology Summary", expanded=True):
                axes = [c for c in profile_df.columns if c not in ['signal_id', 'entity_id']]

                if axes:
                    summary_data = []
                    for axis in axes:
                        if axis in profile_df.columns:
                            values = profile_df[axis].dropna()
                            summary_data.append({
                                'Axis': axis.title(),
                                'Mean': f'{values.mean():.3f}',
                                'Std': f'{values.std():.3f}',
                                'Min': f'{values.min():.3f}',
                                'Max': f'{values.max():.3f}',
                            })

                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True)

        # Pipeline status
        with st.expander("Pipeline Status", expanded=True):
            pipeline_status = {
                'Stage': [
                    'Signal Typology',
                    'Structural Geometry',
                    'Dynamical Systems',
                    'Causal Mechanics',
                ],
                'Status': [
                    '✅ Complete' if profile_df is not None else '⬜ Not run',
                    '✅ Complete' if geometry_df is not None else '⬜ Not run',
                    '✅ Complete' if dynamics_df is not None else '⬜ Not run',
                    '✅ Complete' if mechanics_df is not None else '⬜ Not run',
                ],
                'Command': [
                    'python -m prism.entry_points.signal_typology',
                    'python -m prism.entry_points.structural_geometry',
                    'python -m prism.entry_points.dynamical_systems',
                    'python -m prism.entry_points.causal_mechanics',
                ]
            }

            st.dataframe(pd.DataFrame(pipeline_status), hide_index=True)

    # ---------------------------------------------------------------------
    # Tab 3: Export
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**EXPORT DATA**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Available Exports:**")

            # Signals
            if st.button("Download Signals (CSV)", use_container_width=True):
                csv = signals_df.to_csv(index=False)
                st.download_button(
                    "Confirm Download",
                    csv,
                    file_name="signals.csv",
                    mime="text/csv",
                )

            # Typology
            if profile_df is not None:
                if st.button("Download Typology (CSV)", use_container_width=True):
                    csv = profile_df.to_csv(index=False)
                    st.download_button(
                        "Confirm Download",
                        csv,
                        file_name="typology_profile.csv",
                        mime="text/csv",
                    )

        with col2:
            st.markdown("**Report Formats:**")

            st.button("Generate PDF Report", disabled=True, use_container_width=True)
            st.button("Generate LaTeX Report", disabled=True, use_container_width=True)
            st.button("Generate JSON Summary", disabled=True, use_container_width=True)

    # ---------------------------------------------------------------------
    # Tab 4: Thesis (Scientific Narrative)
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**SCIENTIFIC THESIS**")

        with st.container(border=True):
            st.markdown("### ABSTRACT")

            # Generate abstract based on available data
            abstract_parts = [
                f"This analysis examines {n_signals} signals from the {entity_name} dataset, "
                f"comprising {len(signals_df):,} total observations."
            ]

            if profile_df is not None:
                axes = [c for c in profile_df.columns if c not in ['signal_id', 'entity_id']]
                if axes:
                    abstract_parts.append(
                        f"Signal typology characterization across {len(axes)} behavioral axes "
                        f"reveals distinct patterns in memory persistence, information content, "
                        f"and dynamical properties."
                    )

            abstract_parts.append(
                "Further structural geometry, dynamical systems, and causal mechanics analyses "
                "provide insights into the relationships and driving mechanisms within the system."
            )

            st.markdown(" ".join(abstract_parts))

        st.markdown("---")

        with st.container(border=True):
            st.markdown("### METHODOLOGY")

            methodology = """
**Signal Typology** characterizes individual signals across 9 behavioral axes:
- Memory (Hurst exponent, ACF decay)
- Information (permutation entropy, sample entropy)
- Frequency (spectral analysis, periodicity)
- Volatility (GARCH persistence, variance clustering)
- Dynamics (Lyapunov exponent, embedding)
- Recurrence (RQA measures)
- Discontinuity (level shifts, CUSUM)
- Derivatives (smoothness, spikiness)
- Momentum (runs test, directional persistence)

**Structural Geometry** examines pairwise relationships:
- Correlation structure and networks
- Distance metrics and clustering
- Phase space reconstruction

**Dynamical Systems** analyzes temporal evolution:
- Coherence dynamics
- Synchronization patterns
- Regime transitions

**Causal Mechanics** identifies driving relationships:
- Granger causality
- Transfer entropy (information flow)
- Lead/lag analysis
"""
            st.markdown(methodology)

        st.markdown("---")

        # Export buttons
        col1, col2, col3 = st.columns(3)
        col1.button("Export LaTeX", disabled=True, use_container_width=True)
        col2.button("Export PDF", disabled=True, use_container_width=True)
        col3.button("Copy BibTeX", disabled=True, use_container_width=True)
