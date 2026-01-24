"""
Reusable table components for ORTHON dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def signal_overview_table(
    signals_df: pd.DataFrame,
    signal_col: str = 'signal_id',
    entity_col: str = 'entity_id',
    value_col: str = 'value',
) -> pd.DataFrame:
    """
    Generate signal overview table with statistics.

    Returns DataFrame ready for st.dataframe().
    """
    rows = []

    for sig_id in signals_df[signal_col].unique():
        sig_data = signals_df[signals_df[signal_col] == sig_id]
        values = sig_data[value_col]

        row = {
            'Signal ID': sig_id,
            'Entity': sig_data[entity_col].iloc[0] if entity_col in sig_data.columns else 'N/A',
            'Count': len(values),
            'Mean': f'{values.mean():.4f}',
            'Std': f'{values.std():.4f}',
            'Min': f'{values.min():.4f}',
            'Max': f'{values.max():.4f}',
        }
        rows.append(row)

    return pd.DataFrame(rows)


def typology_table(
    profile_df: pd.DataFrame,
    axes: List[str],
    signal_col: str = 'signal_id',
    classify_fn=None,
) -> pd.DataFrame:
    """
    Generate typology classification table.

    Args:
        profile_df: DataFrame with typology scores
        axes: List of axis columns
        signal_col: Signal identifier column
        classify_fn: Optional function(score, axis) -> label
    """
    rows = []

    for _, row in profile_df.iterrows():
        entry = {'Signal': row[signal_col]}

        for axis in axes:
            if axis in row.index:
                score = row[axis]
                entry[axis.title()] = f'{score:.3f}'

        # Add dominant trait
        scores = {a: row.get(a, 0) for a in axes if a in row.index}
        if scores:
            dominant = max(scores, key=scores.get)
            entry['Dominant'] = dominant.title()

            if classify_fn:
                entry['Class'] = classify_fn(scores[dominant], dominant)

        rows.append(entry)

    return pd.DataFrame(rows)


def classification_table(
    profile_row: pd.Series,
    axes: List[str],
    classify_fn,
) -> pd.DataFrame:
    """
    Generate single-signal classification breakdown.

    Args:
        profile_row: Series with typology scores
        axes: List of axis names
        classify_fn: Function(score, axis) -> label
    """
    rows = []

    for axis in axes:
        score = profile_row.get(axis, np.nan)
        label = classify_fn(score, axis) if not pd.isna(score) else 'N/A'

        rows.append({
            'Axis': axis.title(),
            'Score': f'{score:.3f}' if not pd.isna(score) else 'N/A',
            'Class': label,
        })

    return pd.DataFrame(rows)


def correlation_table(
    corr_matrix: np.ndarray,
    labels: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate table of significant correlations.

    Args:
        corr_matrix: Correlation matrix
        labels: Row/column labels
        threshold: Minimum absolute correlation to include
    """
    rows = []
    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                if corr > 0.7:
                    interp = 'Strong positive'
                elif corr > 0:
                    interp = 'Moderate positive'
                elif corr > -0.7:
                    interp = 'Moderate negative'
                else:
                    interp = 'Strong negative'

                rows.append({
                    'Pair': f'{labels[i]} ↔ {labels[j]}',
                    'Correlation': f'{corr:.3f}',
                    'Interpretation': interp,
                })

    # Sort by absolute correlation
    rows.sort(key=lambda x: abs(float(x['Correlation'])), reverse=True)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['Pair', 'Correlation', 'Interpretation']
    )


def group_membership_table(
    df: pd.DataFrame,
    signal_col: str = 'signal_id',
    group_col: str = 'cluster',
    distance_col: str = 'distance_to_center',
    dominant_col: str = 'dominant_trait',
) -> pd.DataFrame:
    """
    Generate group/cluster membership table.
    """
    cols = [signal_col]
    if group_col in df.columns:
        cols.append(group_col)
    if distance_col in df.columns:
        cols.append(distance_col)
    if dominant_col in df.columns:
        cols.append(dominant_col)

    result = df[cols].copy()
    result.columns = ['Signal', 'Group', 'Distance', 'Dominant'][:len(cols)]

    if 'Distance' in result.columns:
        result['Distance'] = result['Distance'].apply(lambda x: f'{x:.3f}')

    return result


def causal_links_table(
    edges: List[Dict],
) -> pd.DataFrame:
    """
    Generate causal relationships table.

    Args:
        edges: List of dicts with source, target, granger_f, p_value, te
    """
    rows = []

    for edge in edges:
        rows.append({
            'Source': edge.get('source', ''),
            'Target': edge.get('target', ''),
            'Granger F': f"{edge.get('granger_f', 0):.2f}",
            'p-value': f"{edge.get('p_value', 1):.4f}",
            'TE (bits)': f"{edge.get('te', 0):.3f}",
        })

    return pd.DataFrame(rows)


def stability_events_table(
    events: List[Dict],
) -> pd.DataFrame:
    """
    Generate stability transition events table.

    Args:
        events: List of dicts with time, from_state, to_state, duration, confidence
    """
    rows = []

    for event in events:
        rows.append({
            'Time': event.get('time', ''),
            'From': event.get('from_state', ''),
            'To': event.get('to_state', ''),
            'Duration': event.get('duration', '-'),
            'Confidence': f"{event.get('confidence', 0):.2f}",
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Display Helpers
# -----------------------------------------------------------------------------

def render_table(df: pd.DataFrame, hide_index: bool = True, **kwargs):
    """Render DataFrame with consistent styling."""
    st.dataframe(df, hide_index=hide_index, use_container_width=True, **kwargs)


def render_styled_table(
    df: pd.DataFrame,
    highlight_cols: List[str] = None,
    color_map: Dict[str, str] = None,
):
    """
    Render table with conditional styling.

    Args:
        df: DataFrame to display
        highlight_cols: Columns to apply coloring
        color_map: Dict mapping values to colors
    """
    # For now, just render normally - can add styling later
    render_table(df)
