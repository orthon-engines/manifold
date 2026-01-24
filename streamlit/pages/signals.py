"""
Signals Screen
==============

Overview of raw signal data. What do we have?

Tabs:
- Overview: Stats and sparklines
- Time Series: Interactive signal plots
- Statistics: Detailed distributions
- Upload: Add new data (tier-gated)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from components.charts import line_chart, sparklines, multi_line_chart
from components.tables import signal_overview_table, render_table
from components.metrics import stat_cards


def render(signals_df: pd.DataFrame, data_dir: Path):
    """Render the Signals screen."""

    # Horizontal sub-menu
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Time Series", "Statistics", "Upload"
    ])

    signal_ids = sorted(signals_df['signal_id'].unique().tolist())
    n_signals = len(signal_ids)
    n_samples = len(signals_df) // n_signals if n_signals > 0 else 0

    # Get entity count if available
    n_entities = signals_df['entity_id'].nunique() if 'entity_id' in signals_df.columns else 1

    # ---------------------------------------------------------------------
    # Tab 1: Overview
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**SIGNALS** loaded from data directory")

        # Stat cards
        stat_cards(
            total=n_signals,
            signals=n_signals,
            samples=n_samples,
            entities=n_entities if n_entities > 1 else None,
        )

        st.markdown("---")

        # Sparklines
        if n_signals <= 20:
            st.markdown("**Signal Previews**")
            fig = sparklines(
                signals_df,
                signal_col='signal_id',
                time_col='timestamp',
                value_col='value',
                n_cols=5,
                height_per_row=80,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Summary table
        st.markdown("**Signal Statistics**")
        overview_df = signal_overview_table(signals_df)
        render_table(overview_df)

    # ---------------------------------------------------------------------
    # Tab 2: Time Series
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Interactive Time Series**")

        col1, col2 = st.columns([2, 3])

        with col1:
            selected_signal = st.selectbox(
                "Select Signal",
                signal_ids,
                key="ts_signal_select"
            )

        with col2:
            compare_signals = st.multiselect(
                "Compare with",
                [s for s in signal_ids if s != selected_signal],
                key="ts_compare_select",
                max_selections=4,
            )

        # Plot
        if compare_signals:
            # Multi-signal comparison
            all_signals = [selected_signal] + compare_signals
            plot_data = signals_df[signals_df['signal_id'].isin(all_signals)]

            fig = line_chart(
                plot_data,
                x='timestamp',
                y='value',
                color='signal_id',
                height=400,
            )
        else:
            # Single signal
            plot_data = signals_df[signals_df['signal_id'] == selected_signal]

            fig = line_chart(
                plot_data,
                x='timestamp',
                y='value',
                height=400,
                show_legend=False,
            )

        st.plotly_chart(fig, use_container_width=True)

        # Window selector
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            window_preset = st.selectbox(
                "Window Size",
                ["All", "100", "500", "1000", "Custom"],
                key="ts_window"
            )

        if window_preset == "Custom":
            with col2:
                start_idx = st.number_input("Start Index", 0, n_samples - 1, 0)
            with col3:
                end_idx = st.number_input("End Index", 1, n_samples, n_samples)

    # ---------------------------------------------------------------------
    # Tab 3: Statistics
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**Signal Statistics**")

        selected_stat_signal = st.selectbox(
            "Select Signal",
            signal_ids,
            key="stat_signal_select"
        )

        sig_data = signals_df[signals_df['signal_id'] == selected_stat_signal]['value']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Descriptive Statistics**")
            stats = {
                'Count': len(sig_data),
                'Mean': f'{sig_data.mean():.6f}',
                'Std': f'{sig_data.std():.6f}',
                'Min': f'{sig_data.min():.6f}',
                '25%': f'{sig_data.quantile(0.25):.6f}',
                'Median': f'{sig_data.median():.6f}',
                '75%': f'{sig_data.quantile(0.75):.6f}',
                'Max': f'{sig_data.max():.6f}',
                'Skewness': f'{sig_data.skew():.6f}',
                'Kurtosis': f'{sig_data.kurtosis():.6f}',
            }

            for k, v in stats.items():
                st.text(f"{k}: {v}")

        with col2:
            st.markdown("**Distribution**")

            import plotly.express as px
            fig = px.histogram(
                sig_data,
                nbins=50,
                title=None,
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------
    # Tab 4: Upload
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Upload Signal Data**")

        # Import auth functions
        try:
            from auth import check_upload_permission, record_upload
            can_upload = check_upload_permission()
        except ImportError:
            can_upload = True  # Fallback if auth not available

        if can_upload:
            uploaded_file = st.file_uploader(
                "Upload CSV or Parquet file",
                type=['csv', 'parquet'],
                key="signal_upload"
            )

            if uploaded_file is not None:
                st.info("File upload processing would go here.")
                # TODO: Implement file processing
                # - Validate format
                # - Preview data
                # - Save to data directory
                # - record_upload()

            st.markdown("---")
            st.markdown("**Expected Format:**")
            st.code("""
signal_id,entity_id,timestamp,value
HYD_PS1,hyd_rig,0,160.48
HYD_PS1,hyd_rig,1,160.52
...
            """)
