"""
Metric card components for ORTHON dashboard.
"""

import streamlit as st
from typing import List, Tuple, Optional, Any


def metric_row(metrics: List[Tuple[str, Any, Optional[str]]]):
    """
    Display a row of metric cards.

    Args:
        metrics: List of (label, value, delta) tuples
                 delta is optional and can be None
    """
    cols = st.columns(len(metrics))

    for col, (label, value, *rest) in zip(cols, metrics):
        delta = rest[0] if rest else None
        col.metric(label, value, delta)


def metric_grid(metrics: List[Tuple[str, Any]], n_cols: int = 4):
    """
    Display metrics in a grid layout.

    Args:
        metrics: List of (label, value) tuples
        n_cols: Number of columns
    """
    for i in range(0, len(metrics), n_cols):
        row_metrics = metrics[i:i + n_cols]
        cols = st.columns(n_cols)

        for j, (label, value) in enumerate(row_metrics):
            cols[j].metric(label, value)


def summary_card(title: str, items: List[Tuple[str, str]]):
    """
    Display a summary card with title and key-value pairs.

    Args:
        title: Card title
        items: List of (key, value) tuples
    """
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for key, value in items:
            st.text(f"{key}: {value}")


def stat_cards(
    total: int,
    signals: int,
    samples: int,
    entities: int = None,
):
    """
    Standard stat cards for signal overview.
    """
    cols = st.columns(4 if entities else 3)

    cols[0].metric("Total Signals", total)
    cols[1].metric("Samples/Signal", samples)

    if entities:
        cols[2].metric("Entities", entities)
        cols[3].metric("Signals/Entity", f"{total / entities:.1f}")
    else:
        cols[2].metric("Total Samples", total * samples)


def typology_summary_card(
    signal_id: str,
    dominant_trait: str,
    trait_score: float,
    color_hex: str,
):
    """
    Summary card for typology profile.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Dominant Trait", dominant_trait.title())
    col2.metric("Trait Score", f"{trait_score:.3f}")
    col3.metric("Signal Color", color_hex.upper())


def group_summary_card(
    group_id: int,
    n_members: int,
    silhouette: float,
    dominant_traits: List[str],
):
    """
    Summary card for a signal group/cluster.
    """
    st.markdown(f"**Group {group_id}** ({n_members} signals)")
    st.caption(f"Silhouette: {silhouette:.2f}")
    st.caption(f"Dominant: {', '.join(dominant_traits[:3])}")
