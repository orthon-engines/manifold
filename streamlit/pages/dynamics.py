"""
Dynamics Screen
===============

System evolution. How does the system change over time?

Tabs:
- Coherence: System-wide coherence over time
- Synchronization: Pairwise sync measures
- Transitions: State changes
- Windowed: Typology evolution
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Callable, Optional

from components.charts import line_chart, multi_line_chart, heatmap
from components.tables import render_table
from components.metrics import metric_row


def render(
    signals_df: pd.DataFrame,
    dynamics_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    axes: List[str] = None,
    classify_fn: Callable = None,
    data_dir: Path = None,
):
    """
    Render the Dynamics screen.

    Args:
        signals_df: Raw signal data
        dynamics_df: Optional pre-computed dynamics data
        profile_df: Optional typology profiles
        axes: List of typology axis names
        classify_fn: Classification function
        data_dir: Data directory path
    """
    DEFAULT_AXES = [
        'memory', 'information', 'frequency', 'volatility', 'dynamics',
        'recurrence', 'discontinuity', 'derivatives', 'momentum'
    ]
    axes = axes or DEFAULT_AXES

    # Horizontal sub-menu
    tab1, tab2, tab3, tab4 = st.tabs([
        "Coherence", "Synchronization", "Transitions", "Windowed"
    ])

    signal_ids = sorted(signals_df['signal_id'].unique().tolist())

    # ---------------------------------------------------------------------
    # Tab 1: Coherence Evolution
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**DYNAMICAL SYSTEMS:** Coherence Evolution")

        col1, col2 = st.columns([1, 3])

        with col1:
            window_size = st.slider(
                "Window Size",
                50, 500, 200,
                step=50,
                key="coherence_window"
            )
            step_size = st.slider(
                "Step Size",
                10, 100, 20,
                step=10,
                key="coherence_step"
            )

        # Compute rolling coherence (simplified - variance of correlations)
        if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
            try:
                wide_df = signals_df.pivot(
                    index='timestamp',
                    columns='signal_id',
                    values='value'
                ).sort_index()

                n_samples = len(wide_df)
                t_values = []
                coherence_values = []

                for start in range(0, n_samples - window_size + 1, step_size):
                    end = start + window_size
                    window = wide_df.iloc[start:end]

                    # Coherence = mean absolute correlation
                    corr = window.corr().values
                    np.fill_diagonal(corr, 0)
                    coherence = np.mean(np.abs(corr))

                    t_values.append(start + window_size // 2)
                    coherence_values.append(coherence)

                coherence_df = pd.DataFrame({
                    't': t_values,
                    'coherence': coherence_values,
                })

                with col2:
                    fig = line_chart(
                        coherence_df,
                        x='t',
                        y='coherence',
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                if coherence_values:
                    mean_coh = np.mean(coherence_values)
                    min_coh = np.min(coherence_values)
                    max_coh = np.max(coherence_values)
                    min_t = t_values[np.argmin(coherence_values)]
                    max_t = t_values[np.argmax(coherence_values)]

                    metric_row([
                        ("Mean Coherence", f"{mean_coh:.3f}", None),
                        ("Min", f"{min_coh:.3f} (t={min_t})", None),
                        ("Max", f"{max_coh:.3f} (t={max_t})", None),
                    ])

            except Exception as e:
                st.error(f"Could not compute coherence: {e}")
        else:
            st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 2: Synchronization
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Pairwise Synchronization**")

        if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
            try:
                wide_df = signals_df.pivot(
                    index='timestamp',
                    columns='signal_id',
                    values='value'
                ).sort_index()

                # Select signal pair
                col1, col2 = st.columns(2)
                with col1:
                    signal_a = st.selectbox(
                        "Signal A",
                        signal_ids,
                        key="sync_signal_a"
                    )
                with col2:
                    signal_b = st.selectbox(
                        "Signal B",
                        [s for s in signal_ids if s != signal_a],
                        key="sync_signal_b"
                    )

                # Compute rolling correlation
                window = st.slider(
                    "Rolling Window",
                    20, 200, 50,
                    key="sync_window"
                )

                if signal_a in wide_df.columns and signal_b in wide_df.columns:
                    rolling_corr = wide_df[signal_a].rolling(window).corr(wide_df[signal_b])

                    sync_df = pd.DataFrame({
                        't': wide_df.index,
                        'correlation': rolling_corr,
                    })

                    fig = line_chart(
                        sync_df.dropna(),
                        x='t',
                        y='correlation',
                        height=300,
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Corr", f"{rolling_corr.mean():.3f}")
                    col2.metric("Std", f"{rolling_corr.std():.3f}")
                    col3.metric("Current", f"{rolling_corr.iloc[-1]:.3f}" if not pd.isna(rolling_corr.iloc[-1]) else "N/A")

            except Exception as e:
                st.error(f"Could not compute synchronization: {e}")
        else:
            st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 3: Transitions
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**State Transitions**")

        st.info("State transition detection requires pre-computed dynamics data.")

        # Placeholder visualization
        st.markdown("**Regime Timeline**")

        # Synthetic example
        n_points = 100
        t = np.arange(n_points)
        regime = np.zeros(n_points)
        regime[30:50] = 1
        regime[70:85] = 2

        example_df = pd.DataFrame({'t': t, 'regime': regime})

        fig = line_chart(
            example_df,
            x='t',
            y='regime',
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Example regime timeline (synthetic data)")

        st.markdown("---")
        st.markdown("**Transition Events**")

        transitions = [
            {'time': 30, 'from': 'Regime 0', 'to': 'Regime 1'},
            {'time': 50, 'from': 'Regime 1', 'to': 'Regime 0'},
            {'time': 70, 'from': 'Regime 0', 'to': 'Regime 2'},
            {'time': 85, 'from': 'Regime 2', 'to': 'Regime 0'},
        ]

        transitions_df = pd.DataFrame(transitions)
        render_table(transitions_df)

    # ---------------------------------------------------------------------
    # Tab 4: Windowed Typology
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Windowed Typology:** How signal character changes")

        col1, col2 = st.columns(2)

        with col1:
            selected_signal = st.selectbox(
                "Select Signal",
                signal_ids,
                key="windowed_signal"
            )

        with col2:
            selected_axis = st.selectbox(
                "Select Axis",
                axes,
                key="windowed_axis"
            )

        # Windowed computation would go here
        st.info("Windowed typology requires running engines in windowed mode.")

        # Placeholder visualization
        st.markdown("**Typology Score Over Time**")

        # Synthetic example
        n_points = 50
        t = np.arange(n_points) * 20
        score = 0.7 + 0.2 * np.sin(t / 200) + 0.05 * np.random.randn(n_points)
        score = np.clip(score, 0, 1)

        example_df = pd.DataFrame({'t': t, 'score': score})

        fig = line_chart(
            example_df,
            x='t',
            y='score',
            height=300,
        )

        # Add threshold lines
        fig.add_hline(y=0.75, line_dash="dash", line_color="green", annotation_text="strong high")
        fig.add_hline(y=0.60, line_dash="dot", line_color="green")
        fig.add_hline(y=0.40, line_dash="dot", line_color="red")
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="strong low")

        st.plotly_chart(fig, use_container_width=True)

        st.caption("Example windowed typology (synthetic data)")

        # Classification changes
        if classify_fn:
            st.markdown("---")
            st.markdown("**Classification Changes:**")

            # Example changes
            changes = [
                {'time': 200, 'from': 'persistent', 'to': 'weak persistent'},
                {'time': 600, 'from': 'weak persistent', 'to': 'indeterminate'},
                {'time': 800, 'from': 'indeterminate', 'to': 'weak persistent'},
            ]

            for change in changes:
                st.text(f"t={change['time']}: {change['from']} → {change['to']}")
