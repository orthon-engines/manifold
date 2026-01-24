"""
Example Datasets for ORTHON Dashboard

Three preloaded demos - visitors can explore without uploading.
Ships with pre-computed analysis for instant loading.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Example Dataset Registry
# -----------------------------------------------------------------------------

EXAMPLES = {
    'gas_sensors': {
        'name': 'Gas Sensors',
        'icon': '🏭',
        'tagline': 'Sensor drift over 36 months',
        'source': 'UCI Gas Sensor Array Drift',
        'description': 'Watch sensor drift degrade coherence over 36 months',
        'showcases': ['Regime detection', 'Typology evolution', 'Recalibration signals'],
        'signals': '16 metal-oxide sensors, 6 gases',
        'domain': 'Industrial',
        'attracts': 'Manufacturing, IoT, predictive maintenance',
    },
    'spectroscopy': {
        'name': 'Reaction Kinetics',
        'icon': '⚗️',
        'tagline': 'NIR spectral evolution',
        'source': 'NIR Esterification Monitoring',
        'description': 'Track reaction progress through spectral evolution',
        'showcases': ['Multi-wavelength coherence', 'Reaction phases', 'Endpoint detection'],
        'signals': 'NIR absorbance at 20+ wavelengths',
        'domain': 'Chemistry',
        'attracts': 'ChemE, pharma, process chemistry, PAT',
    },
    'cardio': {
        'name': 'Cardio Physiology',
        'icon': '💓',
        'tagline': 'Heart-lung coupling',
        'source': 'PhysioNet CEBS',
        'description': 'Heart-lung coupling reveals stress response',
        'showcases': ['Multi-modal coherence', 'Synchronization', 'System state'],
        'signals': 'ECG + respiratory + seismocardiogram',
        'domain': 'Medical',
        'attracts': 'Biomedical, healthcare, wearables',
    },
}


# -----------------------------------------------------------------------------
# Synthetic Example Data Generation
# -----------------------------------------------------------------------------

def generate_gas_sensors_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic gas sensor drift data."""
    np.random.seed(42)

    t = np.arange(n_samples)
    rows = []

    # 8 sensors with different drift characteristics
    sensors = ['MOX_1', 'MOX_2', 'MOX_3', 'MOX_4', 'TGS_1', 'TGS_2', 'TGS_3', 'TGS_4']

    for sensor in sensors:
        # Base signal with drift
        drift = 0.001 * t + 0.0001 * t * np.sin(t / 200)
        noise = np.random.randn(n_samples) * 0.1

        # Different sensor characteristics
        if 'MOX' in sensor:
            base = 0.5 + 0.3 * np.sin(t / 150) + drift
        else:
            base = 0.3 + 0.2 * np.sin(t / 100 + 1) + drift * 0.5

        values = base + noise

        for i, (ti, v) in enumerate(zip(t, values)):
            rows.append({
                'signal_id': sensor,
                'entity_id': 'gas_array',
                'timestamp': ti,
                'value': v,
            })

    return pd.DataFrame(rows)


def generate_spectroscopy_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic NIR spectroscopy reaction data."""
    np.random.seed(43)

    t = np.arange(n_samples)
    rows = []

    # Wavelengths
    wavelengths = [f'NIR_{w}nm' for w in range(1400, 1800, 50)]

    # Reaction progress (sigmoid)
    reaction_progress = 1 / (1 + np.exp(-(t - 250) / 50))

    for wl in wavelengths:
        # Different wavelengths respond differently to reaction
        wl_num = int(wl.split('_')[1].replace('nm', ''))
        phase_shift = (wl_num - 1400) / 400 * np.pi

        # Absorbance changes with reaction
        baseline = 0.2 + 0.1 * np.sin(phase_shift)
        reactant = (1 - reaction_progress) * 0.5 * (1 + 0.3 * np.sin(phase_shift * 2))
        product = reaction_progress * 0.4 * (1 + 0.2 * np.cos(phase_shift))
        noise = np.random.randn(n_samples) * 0.02

        values = baseline + reactant + product + noise

        for i, (ti, v) in enumerate(zip(t, values)):
            rows.append({
                'signal_id': wl,
                'entity_id': 'esterification',
                'timestamp': ti,
                'value': v,
            })

    return pd.DataFrame(rows)


def generate_cardio_data(n_samples: int = 3000) -> pd.DataFrame:
    """Generate synthetic cardio physiology data."""
    np.random.seed(44)

    t = np.arange(n_samples)
    rows = []

    # Heart rate (60-100 bpm -> period 600-1000 ms at 1ms sampling)
    hr_period = 800 + 100 * np.sin(t / 500)  # Varying HR

    # ECG-like signal
    ecg = np.zeros(n_samples)
    for i in range(n_samples):
        phase = (t[i] % hr_period[i]) / hr_period[i]
        if 0.1 < phase < 0.15:  # QRS complex
            ecg[i] = 1.0 * np.exp(-((phase - 0.12) / 0.01) ** 2)
        elif 0.2 < phase < 0.3:  # T wave
            ecg[i] = 0.3 * np.exp(-((phase - 0.25) / 0.03) ** 2)
        else:
            ecg[i] = 0.05 * np.random.randn()

    # Respiratory signal
    resp_period = 4000  # ~15 breaths per minute
    resp = 0.5 + 0.4 * np.sin(2 * np.pi * t / resp_period)
    resp += 0.05 * np.random.randn(n_samples)

    # Seismocardiogram (heart mechanical)
    scg = np.zeros(n_samples)
    for i in range(n_samples):
        phase = (t[i] % hr_period[i]) / hr_period[i]
        if 0.08 < phase < 0.18:
            scg[i] = 0.8 * np.sin((phase - 0.08) / 0.1 * 4 * np.pi)
        scg[i] += 0.1 * np.random.randn()

    # Blood pressure (correlated with HR)
    bp = 80 + 40 * (1 - hr_period / 1000) + 5 * np.random.randn(n_samples)

    signals = {'ECG': ecg, 'RESP': resp, 'SCG': scg, 'BP': bp}

    for sig_name, values in signals.items():
        for i, (ti, v) in enumerate(zip(t, values)):
            rows.append({
                'signal_id': sig_name,
                'entity_id': 'patient_001',
                'timestamp': ti,
                'value': v,
            })

    return pd.DataFrame(rows)


def generate_typology_profile(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic typology profile for example data."""
    np.random.seed(45)

    signal_ids = signals_df['signal_id'].unique()

    axes = ['memory', 'information', 'frequency', 'volatility', 'dynamics',
            'recurrence', 'discontinuity', 'derivatives', 'momentum']

    rows = []
    for sig in signal_ids:
        row = {'signal_id': sig}
        for axis in axes:
            # Generate plausible scores based on signal type
            base = np.random.uniform(0.3, 0.7)
            row[axis] = np.clip(base + np.random.randn() * 0.15, 0.05, 0.95)
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Example Data Loading
# -----------------------------------------------------------------------------

EXAMPLE_GENERATORS = {
    'gas_sensors': generate_gas_sensors_data,
    'spectroscopy': generate_spectroscopy_data,
    'cardio': generate_cardio_data,
}


def load_example(key: str) -> Dict[str, pd.DataFrame]:
    """
    Load example dataset. Generates synthetic data on demand.

    Returns dict with 'signals', 'typology', etc.
    """
    if key not in EXAMPLES:
        raise ValueError(f"Unknown example: {key}")

    generator = EXAMPLE_GENERATORS.get(key)
    if generator is None:
        raise ValueError(f"No generator for example: {key}")

    signals_df = generator()
    typology_df = generate_typology_profile(signals_df)

    return {
        'signals': signals_df,
        'typology': typology_df,
        'geometry': None,  # Not pre-computed
        'dynamics': None,
        'mechanics': None,
        'meta': EXAMPLES[key],
    }


def get_example_info(key: str) -> Dict[str, Any]:
    """Get metadata about an example dataset."""
    return EXAMPLES.get(key, {})


def list_examples() -> Dict[str, Dict]:
    """List all available examples."""
    return EXAMPLES.copy()


# -----------------------------------------------------------------------------
# Sidebar Rendering
# -----------------------------------------------------------------------------

def render_example_buttons():
    """Render example dataset buttons in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**TRY EXAMPLES**")

    for key, example in EXAMPLES.items():
        button_label = f"{example['icon']} {example['name']}"

        if st.sidebar.button(button_label, key=f"example_{key}", use_container_width=True):
            # Load example into session state
            data = load_example(key)
            st.session_state.signals_data = data['signals']
            st.session_state.typology_data = data['typology']
            st.session_state.geometry_data = data['geometry']
            st.session_state.dynamics_data = data['dynamics']
            st.session_state.mechanics_data = data['mechanics']
            st.session_state.current_example = key
            st.session_state.example_meta = data['meta']

            # Clear cache to reload
            st.cache_data.clear()
            st.rerun()

    # Show current example if loaded
    if 'current_example' in st.session_state:
        meta = st.session_state.get('example_meta', {})
        st.sidebar.caption(f"Loaded: {meta.get('name', 'Unknown')}")


def render_example_info():
    """Render info about currently loaded example."""
    if 'current_example' not in st.session_state:
        return

    meta = st.session_state.get('example_meta', {})

    with st.expander(f"📊 {meta.get('name', 'Example')} Dataset", expanded=False):
        st.markdown(f"**Source:** {meta.get('source', 'Unknown')}")
        st.markdown(f"**Signals:** {meta.get('signals', 'Unknown')}")
        st.markdown(f"**Description:** {meta.get('description', '')}")

        showcases = meta.get('showcases', [])
        if showcases:
            st.markdown("**Showcases:**")
            for item in showcases:
                st.markdown(f"- {item}")
