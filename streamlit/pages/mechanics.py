"""
Mechanics Screen
================

Causal relationships. Why do signals move? What drives what?

Tabs:
- Causality: Granger causality analysis
- Transfer Entropy: Information flow
- Lead/Lag: Temporal relationships
- Network: Causal network visualization
- Equations: Mathematical definitions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from components.charts import heatmap, network_graph, bar_chart
from components.tables import causal_links_table, render_table
from components.metrics import metric_row


def render(
    signals_df: pd.DataFrame,
    mechanics_df: pd.DataFrame = None,
    data_dir: Path = None,
):
    """
    Render the Mechanics screen.

    Args:
        signals_df: Raw signal data
        mechanics_df: Optional pre-computed causal mechanics data
        data_dir: Data directory path
    """
    # Horizontal sub-menu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Causality", "Transfer Entropy", "Lead/Lag", "Network", "Equations"
    ])

    signal_ids = sorted(signals_df['signal_id'].unique().tolist())
    n_signals = len(signal_ids)

    # ---------------------------------------------------------------------
    # Tab 1: Granger Causality
    # ---------------------------------------------------------------------
    with tab1:
        st.markdown("**CAUSAL MECHANICS:** Granger Causality")

        if mechanics_df is not None and 'granger_f' in mechanics_df.columns:
            # Use pre-computed data
            st.info("Using pre-computed Granger causality data.")
        else:
            st.info("Computing Granger causality on demand...")

            # Compute Granger causality
            if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
                try:
                    from statsmodels.tsa.stattools import grangercausalitytests

                    wide_df = signals_df.pivot(
                        index='timestamp',
                        columns='signal_id',
                        values='value'
                    ).sort_index().dropna()

                    max_lag = st.slider(
                        "Max Lag",
                        1, 20, 5,
                        key="granger_lag"
                    )

                    # Compute pairwise Granger causality
                    granger_matrix = np.zeros((n_signals, n_signals))
                    p_matrix = np.ones((n_signals, n_signals))

                    with st.spinner("Computing Granger causality..."):
                        for i, sig_i in enumerate(signal_ids):
                            for j, sig_j in enumerate(signal_ids):
                                if i != j and sig_i in wide_df.columns and sig_j in wide_df.columns:
                                    try:
                                        data = pd.DataFrame({
                                            'y': wide_df[sig_j],
                                            'x': wide_df[sig_i],
                                        }).dropna()

                                        if len(data) > max_lag * 2:
                                            result = grangercausalitytests(
                                                data[['y', 'x']],
                                                maxlag=max_lag,
                                                verbose=False
                                            )
                                            # Get F-statistic from best lag
                                            best_lag = max(result.keys())
                                            f_stat = result[best_lag][0]['ssr_ftest'][0]
                                            p_val = result[best_lag][0]['ssr_ftest'][1]

                                            granger_matrix[i, j] = f_stat
                                            p_matrix[i, j] = p_val
                                    except:
                                        pass

                    # Display heatmap
                    fig = heatmap(
                        granger_matrix,
                        signal_ids,
                        signal_ids,
                        title="Granger F-statistic (row → col)",
                        colorscale='Reds',
                        symmetric=False,
                        height=max(400, n_signals * 40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Significant links
                    st.markdown("---")
                    st.markdown("**Significant Causal Links** (p < 0.05)")

                    links = []
                    for i, sig_i in enumerate(signal_ids):
                        for j, sig_j in enumerate(signal_ids):
                            if i != j and p_matrix[i, j] < 0.05:
                                links.append({
                                    'source': sig_i,
                                    'target': sig_j,
                                    'granger_f': granger_matrix[i, j],
                                    'p_value': p_matrix[i, j],
                                    'te': 0,  # Placeholder
                                })

                    if links:
                        links_df = causal_links_table(links)
                        render_table(links_df)
                    else:
                        st.info("No significant causal links found at p < 0.05")

                except ImportError:
                    st.error("statsmodels not installed. Run: pip install statsmodels")
                except Exception as e:
                    st.error(f"Could not compute Granger causality: {e}")
            else:
                st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 2: Transfer Entropy
    # ---------------------------------------------------------------------
    with tab2:
        st.markdown("**Information Flow:** Transfer Entropy")

        st.info("Transfer entropy computation is resource-intensive. Using placeholder data.")

        # Placeholder TE matrix
        te_matrix = np.random.rand(n_signals, n_signals) * 0.3
        np.fill_diagonal(te_matrix, 0)

        # Make it somewhat asymmetric
        te_matrix = te_matrix + np.triu(te_matrix.T * 0.5, 1)

        col1, col2 = st.columns([3, 2])

        with col1:
            fig = heatmap(
                te_matrix,
                signal_ids,
                signal_ids,
                title="Transfer Entropy (bits): row → col",
                colorscale='Viridis',
                symmetric=False,
                height=max(400, n_signals * 40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Driver Ranking**")

            # Rank by outgoing TE
            outgoing_te = np.sum(te_matrix, axis=1)
            incoming_te = np.sum(te_matrix, axis=0)
            net_te = outgoing_te - incoming_te

            ranking = sorted(zip(signal_ids, net_te), key=lambda x: x[1], reverse=True)

            for i, (sig, net) in enumerate(ranking):
                role = "driver" if net > 0.1 else ("follower" if net < -0.1 else "neutral")
                st.text(f"{i+1}. {sig} ({role})")

        # Summary
        st.markdown("---")
        total_te = np.sum(te_matrix)
        metric_row([
            ("Total Information Flow", f"{total_te:.2f} bits", None),
            ("Signals", n_signals, None),
            ("Links > 0.1 bits", int(np.sum(te_matrix > 0.1)), None),
        ])

    # ---------------------------------------------------------------------
    # Tab 3: Lead/Lag Analysis
    # ---------------------------------------------------------------------
    with tab3:
        st.markdown("**Lead/Lag Relationships**")

        if 'timestamp' in signals_df.columns and 'value' in signals_df.columns:
            col1, col2 = st.columns(2)

            with col1:
                signal_a = st.selectbox(
                    "Signal A",
                    signal_ids,
                    key="leadlag_a"
                )

            with col2:
                signal_b = st.selectbox(
                    "Signal B",
                    [s for s in signal_ids if s != signal_a],
                    key="leadlag_b"
                )

            max_lag = st.slider(
                "Max Lag",
                1, 100, 20,
                key="leadlag_max"
            )

            # Compute cross-correlation
            wide_df = signals_df.pivot(
                index='timestamp',
                columns='signal_id',
                values='value'
            ).sort_index()

            if signal_a in wide_df.columns and signal_b in wide_df.columns:
                a = wide_df[signal_a].dropna().values
                b = wide_df[signal_b].dropna().values

                min_len = min(len(a), len(b))
                a, b = a[:min_len], b[:min_len]

                # Normalize
                a = (a - np.mean(a)) / np.std(a)
                b = (b - np.mean(b)) / np.std(b)

                # Cross-correlation at different lags
                lags = np.arange(-max_lag, max_lag + 1)
                xcorr = []

                for lag in lags:
                    if lag < 0:
                        corr = np.corrcoef(a[:lag], b[-lag:])[0, 1]
                    elif lag > 0:
                        corr = np.corrcoef(a[lag:], b[:-lag])[0, 1]
                    else:
                        corr = np.corrcoef(a, b)[0, 1]

                    xcorr.append(corr if not np.isnan(corr) else 0)

                # Plot
                xcorr_df = pd.DataFrame({'lag': lags, 'correlation': xcorr})

                import plotly.express as px
                fig = px.bar(
                    xcorr_df,
                    x='lag',
                    y='correlation',
                    title=f"Cross-correlation: {signal_a} vs {signal_b}",
                )
                fig.update_layout(height=300)
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                best_lag = lags[np.argmax(np.abs(xcorr))]
                best_corr = xcorr[np.argmax(np.abs(xcorr))]

                if best_lag < 0:
                    st.info(f"**{signal_b}** leads **{signal_a}** by {-best_lag} time steps (r = {best_corr:.3f})")
                elif best_lag > 0:
                    st.info(f"**{signal_a}** leads **{signal_b}** by {best_lag} time steps (r = {best_corr:.3f})")
                else:
                    st.info(f"Signals are synchronous (r = {best_corr:.3f})")
        else:
            st.warning("Signal data must have 'timestamp' and 'value' columns.")

    # ---------------------------------------------------------------------
    # Tab 4: Causal Network
    # ---------------------------------------------------------------------
    with tab4:
        st.markdown("**Causal Network**")

        # Use Granger or TE data to build network
        st.info("Displaying example causal network based on correlation thresholds.")

        # Build edges from correlation
        if 'timestamp' in signals_df.columns:
            try:
                wide_df = signals_df.pivot(
                    index='timestamp',
                    columns='signal_id',
                    values='value'
                ).sort_index()

                corr_matrix = wide_df.corr().values

                threshold = st.slider(
                    "Edge Threshold",
                    0.3, 0.9, 0.6,
                    key="causal_network_threshold"
                )

                edges = []
                for i in range(n_signals):
                    for j in range(n_signals):
                        if i != j and abs(corr_matrix[i, j]) >= threshold:
                            edges.append({
                                'source': signal_ids[i],
                                'target': signal_ids[j],
                                'weight': corr_matrix[i, j],
                            })

                if edges:
                    fig = network_graph(edges, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(edges)} edges above threshold")
                else:
                    st.info("No edges above threshold.")

            except Exception as e:
                st.error(f"Could not build network: {e}")

    # ---------------------------------------------------------------------
    # Tab 5: Equations
    # ---------------------------------------------------------------------
    with tab5:
        st.markdown("**Mathematical Definitions**")

        equations = {
            'Granger Causality': r'''
$$X \text{ Granger-causes } Y \text{ if } $$
$$\text{Var}(Y_t | Y_{t-1},...,Y_{t-p}) > \text{Var}(Y_t | Y_{t-1},...,Y_{t-p}, X_{t-1},...,X_{t-p})$$

F-test statistic:
$$F = \frac{(RSS_{restricted} - RSS_{unrestricted}) / p}{RSS_{unrestricted} / (n - 2p - 1)}$$
''',
            'Transfer Entropy': r'''
$$T_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log \frac{p(y_{t+1} | y_t^{(k)}, x_t^{(l)})}{p(y_{t+1} | y_t^{(k)})}$$

Where:
- $y_t^{(k)}$ = history of Y with embedding dimension k
- $x_t^{(l)}$ = history of X with embedding dimension l

Interpretation: Bits of information X provides about Y's future beyond Y's own past.
''',
            'Cross-Correlation': r'''
$$r_{XY}(\tau) = \frac{E[(X_t - \mu_X)(Y_{t+\tau} - \mu_Y)]}{\sigma_X \sigma_Y}$$

Lead/Lag detection:
- $\tau^* = \arg\max_\tau |r_{XY}(\tau)|$
- If $\tau^* > 0$: X leads Y
- If $\tau^* < 0$: Y leads X
''',
            'Convergent Cross Mapping': r'''
$$\hat{X}_t = \sum_{i=1}^{k+1} w_i X_{t_i}$$

Where:
- $t_i$ are nearest neighbors in the Y manifold
- $w_i$ are distance-based weights

If prediction improves with library size: X causally influences Y.
''',
        }

        for title, equation in equations.items():
            with st.expander(title, expanded=False):
                st.markdown(equation)
