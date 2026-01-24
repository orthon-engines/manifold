"""
Typology Screen
===============

Signal classification. What KIND of signals are these?

Tabs:
- Profiles: Cohort heatmap and distribution
- Radar: Individual signal radar charts
- Classification: Detailed breakdown
- Metrics: Raw engine outputs
- Equations: Mathematical definitions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Callable

from components.charts import (
    radar_chart, radar_compare, typology_heatmap,
    distribution_bars, bar_chart
)
from components.tables import (
    typology_table, classification_table, render_table
)
from components.metrics import typology_summary_card


# Default axes (can be overridden)
DEFAULT_AXES = [
    'memory', 'information', 'frequency', 'volatility', 'dynamics',
    'recurrence', 'discontinuity', 'derivatives', 'momentum'
]


def render(
    profile_df: pd.DataFrame,
    metrics_df: pd.DataFrame = None,
    axes: List[str] = None,
    classify_fn: Callable = None,
    signals_df: pd.DataFrame = None,
):
    """
    Render the Typology screen.

    Args:
        profile_df: DataFrame with typology scores (0-1 normalized)
        metrics_df: Optional DataFrame with raw engine metrics
        axes: List of axis names
        classify_fn: Function(score, axis) -> label
        signals_df: Optional signals DataFrame for signal selection
    """
    axes = axes or DEFAULT_AXES
    available_axes = [a for a in axes if a in profile_df.columns]

    if not available_axes:
        st.warning("No typology axes found in profile data.")
        return

    # Horizontal sub-menu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Profiles", "Radar", "Classification", "Metrics", "Equations"
    ])

    signal_ids = profile_df['signal_id'].unique().tolist()

    # ---------------------------------------------------------------------
    # Tab 1: Profiles (Cohort Overview)
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown(f"**SIGNAL TYPOLOGY:** {len(available_axes)} axes measured")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("**Cohort Heatmap**")
            fig = typology_heatmap(
                profile_df,
                signal_col='signal_id',
                axes=available_axes,
                height=max(300, len(signal_ids) * 25),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Classification Distribution**")

            if classify_fn:
                # Count classifications per axis
                for axis in available_axes[:4]:  # Show top 4
                    if axis in profile_df.columns:
                        labels = profile_df[axis].apply(
                            lambda x: classify_fn(x, axis)
                        )
                        counts = labels.value_counts().to_dict()

                        with st.expander(axis.title(), expanded=axis == available_axes[0]):
                            fig = distribution_bars(counts, height=120)
                            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Summary table
        st.markdown("**Typology Summary**")
        table_df = typology_table(
            profile_df,
            axes=available_axes,
            signal_col='signal_id',
            classify_fn=classify_fn,
        )
        render_table(table_df)

    # ---------------------------------------------------------------------
    # Tab 2: Radar Charts
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Signal Typology Radar**")

        col1, col2 = st.columns([1, 1])

        with col1:
            selected_signal = st.selectbox(
                "Select Signal",
                signal_ids,
                key="radar_signal_select"
            )

        with col2:
            compare_signals = st.multiselect(
                "Compare with",
                [s for s in signal_ids if s != selected_signal],
                key="radar_compare_select",
                max_selections=3,
            )

        # Get profile data
        signal_profile = profile_df[profile_df['signal_id'] == selected_signal]

        if signal_profile.empty:
            st.warning(f"No profile data for {selected_signal}")
        else:
            profile_row = signal_profile.iloc[0]
            values = [profile_row.get(a, 0) for a in available_axes]

            # Summary metrics
            scores = {a: profile_row.get(a, 0) for a in available_axes}
            dominant = max(scores, key=scores.get)
            dominant_score = scores[dominant]

            # Color from RGB mapping
            r = int(255 * profile_row.get('volatility', 0.5))
            g = int(255 * profile_row.get('frequency', 0.5))
            b = int(255 * profile_row.get('memory', 0.5))
            color_hex = f'#{r:02x}{g:02x}{b:02x}'

            typology_summary_card(
                selected_signal,
                dominant,
                dominant_score,
                color_hex,
            )

            st.markdown("---")

            if compare_signals:
                # Multi-signal radar
                profiles = [{'name': selected_signal, 'values': values}]

                for sig in compare_signals:
                    sig_profile = profile_df[profile_df['signal_id'] == sig]
                    if not sig_profile.empty:
                        sig_values = [sig_profile.iloc[0].get(a, 0) for a in available_axes]
                        profiles.append({'name': sig, 'values': sig_values})

                fig = radar_compare(profiles, available_axes, height=450)
            else:
                # Single radar
                fig = radar_chart(
                    values,
                    available_axes,
                    name=selected_signal,
                    color=color_hex,
                    height=450,
                )

            st.plotly_chart(fig, use_container_width=True)

            # Narrative
            if classify_fn:
                st.markdown("---")
                st.markdown("**Narrative:**")

                labels = {a: classify_fn(profile_row.get(a, 0.5), a)
                          for a in available_axes}

                notable = {k: v for k, v in labels.items()
                           if v not in ['indeterminate', 'N/A', 'insufficient data']}

                if notable:
                    strong = [f"{v} ({k})" for k, v in notable.items()
                              if not v.startswith('weak')]
                    weak = [f"{v.replace('weak ', '')} ({k})"
                            for k, v in notable.items()
                            if v.startswith('weak')]

                    parts = [f"Signal **{selected_signal}**"]
                    if strong:
                        parts.append(f"is {', '.join(strong)}")
                    if weak:
                        if strong:
                            parts.append(f"with weak tendencies toward {', '.join(weak)}")
                        else:
                            parts.append(f"shows weak tendencies toward {', '.join(weak)}")

                    st.markdown(' '.join(parts) + '.')
                else:
                    st.markdown(f"Signal **{selected_signal}** shows no strong characteristics.")

    # ---------------------------------------------------------------------
    # Tab 3: Classification Detail
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**Classification Breakdown**")

        selected_class_signal = st.selectbox(
            "Select Signal",
            signal_ids,
            key="class_signal_select"
        )

        signal_profile = profile_df[profile_df['signal_id'] == selected_class_signal]

        if signal_profile.empty:
            st.warning(f"No profile data for {selected_class_signal}")
        else:
            profile_row = signal_profile.iloc[0]

            if classify_fn:
                class_df = classification_table(profile_row, available_axes, classify_fn)
                render_table(class_df)
            else:
                # Just show scores without classification
                st.markdown("**Axis Scores:**")
                for axis in available_axes:
                    score = profile_row.get(axis, np.nan)
                    st.text(f"{axis.title()}: {score:.3f}" if not pd.isna(score) else f"{axis.title()}: N/A")

    # ---------------------------------------------------------------------
    # Tab 4: Raw Metrics
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Raw Engine Metrics**")

        if metrics_df is None or metrics_df.empty:
            st.info("Run signal_typology to generate metrics data.")
            st.code("python -m prism.entry_points.signal_typology", language="bash")
        else:
            selected_metrics_signal = st.selectbox(
                "Select Signal",
                signal_ids,
                key="metrics_signal_select"
            )

            signal_metrics = metrics_df[metrics_df['signal_id'] == selected_metrics_signal]

            if signal_metrics.empty:
                st.warning(f"No metrics found for {selected_metrics_signal}")
            else:
                metrics_row = signal_metrics.iloc[0]

                # Group metrics by category
                metric_groups = {
                    'Memory': ['hurst_exponent', 'acf_decay_rate', 'acf_lag1'],
                    'Information': ['permutation_entropy', 'sample_entropy'],
                    'Frequency': ['spectral_entropy', 'dominant_frequency', 'spectral_centroid', 'spectral_flatness'],
                    'Volatility': ['garch_alpha', 'garch_beta', 'garch_persistence', 'rolling_std_ratio'],
                    'Derivatives': ['derivative_mean', 'derivative_std', 'derivative_kurtosis', 'zero_crossing_rate'],
                    'Recurrence': ['recurrence_rate', 'determinism', 'laminarity', 'trapping_time'],
                    'Discontinuity': ['cusum_max', 'cusum_crossings', 'level_shift_count', 'level_shift_magnitude_mean'],
                    'Momentum': ['runs_test_z', 'runs_ratio', 'momentum_score'],
                }

                cols = st.columns(2)

                for i, (group_name, group_metrics) in enumerate(metric_groups.items()):
                    with cols[i % 2]:
                        with st.expander(group_name, expanded=True):
                            for m in group_metrics:
                                if m in metrics_row.index:
                                    val = metrics_row[m]
                                    if isinstance(val, float):
                                        st.text(f"{m}: {val:.6f}")
                                    else:
                                        st.text(f"{m}: {val}")

    # ---------------------------------------------------------------------
    # Tab 5: Equations
    # ---------------------------------------------------------------------
    with tab5:
        st.markdown("**Mathematical Definitions**")

        equations = {
            'Memory (Hurst Exponent)': r'''
$$H = \lim_{n \to \infty} \frac{\log(R/S)}{\log(n)}$$

Where $R/S$ is the rescaled range statistic.

- $H < 0.5$: Anti-persistent (mean-reverting)
- $H = 0.5$: Random walk
- $H > 0.5$: Persistent (trending)
''',
            'Information (Permutation Entropy)': r'''
$$H_p = -\sum_{\pi} p(\pi) \log p(\pi)$$

Where $\pi$ are ordinal patterns of length $m$.

Normalized: $H_{norm} = H_p / \log(m!)$
''',
            'Frequency (Spectral Entropy)': r'''
$$H_{spectral} = -\sum_f P(f) \log P(f)$$

Where $P(f)$ is the normalized power spectral density.

Low entropy = narrowband (periodic)
High entropy = broadband (aperiodic)
''',
            'Volatility (GARCH Persistence)': r'''
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Persistence $= \alpha + \beta$

- Near 1: volatility clusters persist
- Near 0: volatility mean-reverts quickly
''',
            'Dynamics (Lyapunov Exponent)': r'''
$$\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta Z(t)||}{||\delta Z(0)||}$$

- $\lambda > 0$: Chaotic (sensitive to initial conditions)
- $\lambda = 0$: Edge of chaos
- $\lambda < 0$: Deterministic/stable
''',
            'Recurrence (RQA)': r'''
$$RR = \frac{1}{N^2} \sum_{i,j=1}^{N} R_{i,j}$$

$$DET = \frac{\sum_{l=l_{min}}^{N} l \cdot P(l)}{\sum_{l=1}^{N} l \cdot P(l)}$$

Where $R_{i,j} = \Theta(\epsilon - ||x_i - x_j||)$
''',
            'Momentum (Runs Test)': r'''
$$Z = \frac{R - E(R)}{\sqrt{Var(R)}}$$

Where:
- $E(R) = \frac{2n_+n_-}{n_+ + n_-} + 1$
- Fewer runs than expected → trending
- More runs than expected → reverting
''',
        }

        for title, equation in equations.items():
            with st.expander(title, expanded=False):
                st.markdown(equation)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.button("Expand All", disabled=True)
        col2.button("Collapse All", disabled=True)
        col3.button("Export PDF", disabled=True)
