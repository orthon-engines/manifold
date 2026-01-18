#!/usr/bin/env python3
"""
MIMIC Temporal Decoupling Analysis

Tracks vital-to-vital correlation trajectories leading up to sepsis onset.

Question: Does coupling decrease BEFORE sepsis diagnosis?

Method:
    1. Identify infection onset time for each patient
    2. Compute correlations in sliding windows (e.g., 4-hour)
    3. Align to t=0 (infection onset)
    4. Track trajectory: t=-12h, t=-8h, t=-4h, t=0

This tests whether decoupling is PREDICTIVE, not just ASSOCIATIVE.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd
    import pyarrow.parquet as pq


def load_mimic_data(mimic_dir: Path):
    """Load MIMIC-IV demo tables needed for temporal analysis."""
    base = mimic_dir / "mimic-iv-clinical-database-demo-2.2"

    # Load chartevents (vitals with timestamps)
    chartevents_path = base / "icu" / "chartevents.csv.gz"
    if not chartevents_path.exists():
        chartevents_path = base / "icu" / "chartevents.csv"

    print(f"Loading chartevents from {chartevents_path}...")
    if HAS_POLARS:
        chartevents = pl.read_csv(
            chartevents_path,
            columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
            schema_overrides={"valuenum": pl.Float64}
        )
    else:
        chartevents = pd.read_csv(
            chartevents_path,
            usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
        )

    # Load ICU stays for admission times
    icustays_path = base / "icu" / "icustays.csv.gz"
    if not icustays_path.exists():
        icustays_path = base / "icu" / "icustays.csv"

    print(f"Loading icustays from {icustays_path}...")
    if HAS_POLARS:
        icustays = pl.read_csv(icustays_path)
    else:
        icustays = pd.read_csv(icustays_path)

    # Load microbiology for infection timing
    micro_path = base / "hosp" / "microbiologyevents.csv.gz"
    if not micro_path.exists():
        micro_path = base / "hosp" / "microbiologyevents.csv"

    if micro_path.exists():
        print(f"Loading microbiologyevents from {micro_path}...")
        if HAS_POLARS:
            micro = pl.read_csv(micro_path)
        else:
            micro = pd.read_csv(micro_path)
    else:
        micro = None

    # Load prescriptions for antibiotic timing
    rx_path = base / "hosp" / "prescriptions.csv.gz"
    if not rx_path.exists():
        rx_path = base / "hosp" / "prescriptions.csv"

    if rx_path.exists():
        print(f"Loading prescriptions from {rx_path}...")
        if HAS_POLARS:
            prescriptions = pl.read_csv(rx_path, infer_schema_length=10000, ignore_errors=True)
        else:
            prescriptions = pd.read_csv(rx_path, low_memory=False)
    else:
        prescriptions = None

    # Load sepsis diagnoses
    diag_path = base / "hosp" / "diagnoses_icd.csv.gz"
    if not diag_path.exists():
        diag_path = base / "hosp" / "diagnoses_icd.csv"

    if HAS_POLARS:
        diagnoses = pl.read_csv(diag_path)
    else:
        diagnoses = pd.read_csv(diag_path)

    return chartevents, icustays, micro, prescriptions, diagnoses


def get_infection_onset_times(icustays, micro, prescriptions, diagnoses):
    """
    Determine infection onset time for each stay.

    Uses earliest of:
    - First positive culture
    - First antibiotic administration

    Returns dict: stay_id -> infection_onset_datetime
    """
    infection_times = {}

    # Get sepsis patients
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]

    if HAS_POLARS:
        sepsis_hadms = diagnoses.filter(
            pl.col("icd_code").is_in(sepsis_codes)
        )["hadm_id"].unique().to_list()

        # Get ICU stays for sepsis patients
        sepsis_stays = icustays.filter(pl.col("hadm_id").is_in(sepsis_hadms))

        for row in sepsis_stays.iter_rows(named=True):
            stay_id = row["stay_id"]
            hadm_id = row["hadm_id"]
            icu_intime = row["intime"]

            # Parse ICU admission time
            if isinstance(icu_intime, str):
                icu_intime = datetime.fromisoformat(icu_intime.replace(" ", "T"))

            onset_time = None

            # Check microbiology for culture times
            if micro is not None:
                patient_micro = micro.filter(pl.col("hadm_id") == hadm_id)
                if len(patient_micro) > 0 and "charttime" in patient_micro.columns:
                    culture_times = patient_micro["charttime"].drop_nulls().to_list()
                    if culture_times:
                        first_culture = min(culture_times)
                        if isinstance(first_culture, str):
                            first_culture = datetime.fromisoformat(first_culture.replace(" ", "T"))
                        onset_time = first_culture

            # Check prescriptions for antibiotic times
            if prescriptions is not None:
                patient_rx = prescriptions.filter(pl.col("hadm_id") == hadm_id)
                if len(patient_rx) > 0 and "starttime" in patient_rx.columns:
                    # Filter to antibiotics (simplified - look for common antibiotic terms)
                    abx_terms = ["cillin", "mycin", "floxacin", "cef", "vanco", "metro", "zosyn", "meropenem"]
                    if "drug" in patient_rx.columns:
                        for term in abx_terms:
                            abx_rx = patient_rx.filter(pl.col("drug").str.to_lowercase().str.contains(term))
                            if len(abx_rx) > 0:
                                abx_times = abx_rx["starttime"].drop_nulls().to_list()
                                if abx_times:
                                    first_abx = min(abx_times)
                                    if isinstance(first_abx, str):
                                        first_abx = datetime.fromisoformat(first_abx.replace(" ", "T"))
                                    if onset_time is None or first_abx < onset_time:
                                        onset_time = first_abx

            if onset_time is not None:
                infection_times[stay_id] = {
                    "onset_time": onset_time,
                    "icu_intime": icu_intime,
                    "hours_after_icu": (onset_time - icu_intime).total_seconds() / 3600
                }
    else:
        # Pandas version
        sepsis_hadms = diagnoses[diagnoses["icd_code"].isin(sepsis_codes)]["hadm_id"].unique()
        sepsis_stays = icustays[icustays["hadm_id"].isin(sepsis_hadms)]

        for _, row in sepsis_stays.iterrows():
            stay_id = row["stay_id"]
            hadm_id = row["hadm_id"]
            icu_intime = pd.to_datetime(row["intime"])

            onset_time = None

            if micro is not None and "charttime" in micro.columns:
                patient_micro = micro[micro["hadm_id"] == hadm_id]
                culture_times = patient_micro["charttime"].dropna()
                if len(culture_times) > 0:
                    first_culture = pd.to_datetime(culture_times.min())
                    onset_time = first_culture

            if prescriptions is not None and "starttime" in prescriptions.columns:
                patient_rx = prescriptions[prescriptions["hadm_id"] == hadm_id]
                if "drug" in patient_rx.columns:
                    abx_terms = ["cillin", "mycin", "floxacin", "cef", "vanco", "metro", "zosyn", "meropenem"]
                    for term in abx_terms:
                        abx_rx = patient_rx[patient_rx["drug"].str.lower().str.contains(term, na=False)]
                        if len(abx_rx) > 0:
                            first_abx = pd.to_datetime(abx_rx["starttime"].min())
                            if onset_time is None or first_abx < onset_time:
                                onset_time = first_abx

            if onset_time is not None:
                infection_times[stay_id] = {
                    "onset_time": onset_time,
                    "icu_intime": icu_intime,
                    "hours_after_icu": (onset_time - icu_intime).total_seconds() / 3600
                }

    return infection_times


def compute_windowed_correlations(chartevents, stay_id, vital_pairs, infection_onset,
                                   window_hours=4, step_hours=2,
                                   hours_before=24, hours_after=6):
    """
    Compute correlations in sliding windows aligned to infection onset.

    Returns list of dicts with:
        - hours_to_onset: time relative to infection (negative = before)
        - pair: vital pair name
        - correlation: Pearson r
    """
    # Vital itemids
    VITAL_ITEMIDS = {
        220045: "heart_rate",
        220050: "arterial_bp_systolic",
        220051: "arterial_bp_diastolic",
        220052: "arterial_bp_mean",
        220179: "nibp_systolic",
        220180: "nibp_diastolic",
        220181: "nibp_mean",
        220210: "respiratory_rate",
        220277: "spo2",
    }

    results = []

    # Filter to this stay and vital signs
    if HAS_POLARS:
        stay_data = chartevents.filter(
            (pl.col("stay_id") == stay_id) &
            (pl.col("itemid").is_in(list(VITAL_ITEMIDS.keys())))
        )

        if len(stay_data) < 50:
            return results

        # Add vital name and parse time
        vital_map = pl.DataFrame({
            "itemid": list(VITAL_ITEMIDS.keys()),
            "vital_name": list(VITAL_ITEMIDS.values())
        })
        stay_data = stay_data.join(vital_map, on="itemid", how="left")
        stay_data = stay_data.with_columns(
            pl.col("charttime").str.to_datetime().alias("charttime")
        )

        # Get time range
        start_time = infection_onset - timedelta(hours=hours_before)
        end_time = infection_onset + timedelta(hours=hours_after)

        # Filter to time range
        stay_data = stay_data.filter(
            (pl.col("charttime") >= start_time) &
            (pl.col("charttime") <= end_time)
        )

        if len(stay_data) < 20:
            return results

        # Compute correlations for each window
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + timedelta(hours=window_hours)
            hours_to_onset = (current_time + timedelta(hours=window_hours/2) - infection_onset).total_seconds() / 3600

            window_data = stay_data.filter(
                (pl.col("charttime") >= current_time) &
                (pl.col("charttime") < window_end)
            )

            if len(window_data) < 10:
                current_time += timedelta(hours=step_hours)
                continue

            # Get values for each vital in this window
            vital_values = {}
            for vital in VITAL_ITEMIDS.values():
                vital_data = window_data.filter(pl.col("vital_name") == vital)
                if len(vital_data) >= 3:
                    vital_values[vital] = vital_data["valuenum"].drop_nulls().to_numpy()

            # Compute correlations for requested pairs
            for v1, v2 in vital_pairs:
                if v1 in vital_values and v2 in vital_values:
                    vals1 = vital_values[v1]
                    vals2 = vital_values[v2]

                    # Align lengths
                    min_len = min(len(vals1), len(vals2))
                    if min_len >= 5:
                        try:
                            r, _ = pearsonr(vals1[:min_len], vals2[:min_len])
                            if not np.isnan(r):
                                results.append({
                                    "stay_id": stay_id,
                                    "hours_to_onset": round(hours_to_onset, 1),
                                    "pair": f"{v1}___{v2}",
                                    "correlation": r,
                                    "abs_correlation": abs(r),
                                    "n_points": min_len,
                                })
                        except:
                            pass

            current_time += timedelta(hours=step_hours)

    return results


def compute_stay_phase_correlations(chartevents, icustays, diagnoses, vital_pairs):
    """
    Compare correlations in early vs late phase of ICU stay.

    For each stay:
    - Early phase: first 12 hours
    - Late phase: 12-24 hours

    Returns DataFrame with phase-based correlations.
    """
    VITAL_ITEMIDS = {
        220045: "heart_rate",
        220050: "arterial_bp_systolic",
        220051: "arterial_bp_diastolic",
        220052: "arterial_bp_mean",
        220179: "nibp_systolic",
        220180: "nibp_diastolic",
        220181: "nibp_mean",
        220210: "respiratory_rate",
        220277: "spo2",
    }

    # Get sepsis status for each stay
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]

    if HAS_POLARS:
        sepsis_hadms = diagnoses.filter(
            pl.col("icd_code").is_in(sepsis_codes)
        )["hadm_id"].unique().to_list()

        # Filter chartevents to vitals
        vital_data = chartevents.filter(pl.col("itemid").is_in(list(VITAL_ITEMIDS.keys())))

        # Add vital name
        vital_map = pl.DataFrame({
            "itemid": list(VITAL_ITEMIDS.keys()),
            "vital_name": list(VITAL_ITEMIDS.values())
        })
        vital_data = vital_data.join(vital_map, on="itemid", how="left")
        vital_data = vital_data.with_columns(
            pl.col("charttime").str.to_datetime().alias("charttime")
        )

        results = []
        stays = icustays.to_dicts()

        print(f"Processing {len(stays)} ICU stays...")

        for i, stay in enumerate(stays):
            if i % 20 == 0:
                print(f"  {i}/{len(stays)}...")

            stay_id = stay["stay_id"]
            hadm_id = stay["hadm_id"]
            intime = stay["intime"]

            if isinstance(intime, str):
                intime = datetime.fromisoformat(intime.replace(" ", "T"))

            regime = "septic" if hadm_id in sepsis_hadms else "stable"

            # Get stay's vital data
            stay_vitals = vital_data.filter(pl.col("stay_id") == stay_id)

            if len(stay_vitals) < 30:
                continue

            # Define phases
            early_end = intime + timedelta(hours=12)
            late_start = intime + timedelta(hours=12)
            late_end = intime + timedelta(hours=24)

            for phase, start, end in [("early", intime, early_end), ("late", late_start, late_end)]:
                phase_data = stay_vitals.filter(
                    (pl.col("charttime") >= start) &
                    (pl.col("charttime") < end)
                )

                if len(phase_data) < 10:
                    continue

                # Get values for each vital
                vital_values = {}
                for vital in VITAL_ITEMIDS.values():
                    v_data = phase_data.filter(pl.col("vital_name") == vital)
                    if len(v_data) >= 5:
                        vital_values[vital] = v_data["valuenum"].drop_nulls().to_numpy()

                # Compute correlations
                for v1, v2 in vital_pairs:
                    if v1 in vital_values and v2 in vital_values:
                        vals1 = vital_values[v1]
                        vals2 = vital_values[v2]
                        min_len = min(len(vals1), len(vals2))

                        if min_len >= 5:
                            try:
                                r, _ = pearsonr(vals1[:min_len], vals2[:min_len])
                                if not np.isnan(r):
                                    results.append({
                                        "stay_id": stay_id,
                                        "regime": regime,
                                        "phase": phase,
                                        "pair": f"{v1}___{v2}",
                                        "correlation": r,
                                        "abs_correlation": abs(r),
                                        "n_points": min_len,
                                    })
                            except:
                                pass

        return pl.DataFrame(results) if results else None

    return None


def main():
    data_dir = Path("data/mimic_demo")

    print("=" * 70)
    print("MIMIC Temporal Decoupling: Early vs Late Phase Correlations")
    print("=" * 70)
    print()
    print("Question: Does vital coupling DECREASE over time in septic patients?")
    print()
    print("Method:")
    print("  - Early phase: First 12 hours of ICU stay")
    print("  - Late phase: Hours 12-24 of ICU stay")
    print("  - Compare correlation change (Δ) between septic and stable")
    print()

    # Load data
    chartevents, icustays, micro, prescriptions, diagnoses = load_mimic_data(data_dir)

    # Key vital pairs
    vital_pairs = [
        ("heart_rate", "arterial_bp_mean"),
        ("heart_rate", "respiratory_rate"),
        ("heart_rate", "spo2"),
        ("respiratory_rate", "spo2"),
        ("heart_rate", "nibp_mean"),
        ("nibp_systolic", "heart_rate"),
    ]

    print()
    results = compute_stay_phase_correlations(chartevents, icustays, diagnoses, vital_pairs)

    if results is None or len(results) == 0:
        print("No results computed.")
        return

    print(f"\nComputed {len(results)} phase correlations")
    print()

    # Save
    output_dir = data_dir / "geometry"
    output_dir.mkdir(exist_ok=True)
    results.write_parquet(output_dir / "temporal_phases.parquet")
    print(f"Saved to {output_dir / 'temporal_phases.parquet'}")
    print()

    # Analysis: Early vs Late phase
    df = results

    print("=" * 70)
    print("RESULTS: Early vs Late Phase Correlations")
    print("=" * 70)
    print()

    # Summary by regime and phase
    summary = df.group_by(["regime", "phase"]).agg([
        pl.col("abs_correlation").mean().round(3).alias("mean_abs_corr"),
        pl.col("correlation").mean().round(3).alias("mean_corr"),
        pl.len().alias("n"),
    ]).sort(["regime", "phase"])

    print("Mean |correlation| by regime and phase:")
    print(summary)
    print()

    # Compute delta (late - early) for each stay
    print("=" * 70)
    print("TRAJECTORY ANALYSIS: Correlation Change Over Time")
    print("=" * 70)
    print()

    # Pivot to get early and late for each stay+pair
    early = df.filter(pl.col("phase") == "early").select([
        "stay_id", "regime", "pair", pl.col("abs_correlation").alias("early_corr")
    ])
    late = df.filter(pl.col("phase") == "late").select([
        "stay_id", "pair", pl.col("abs_correlation").alias("late_corr")
    ])

    paired = early.join(late, on=["stay_id", "pair"], how="inner")
    paired = paired.with_columns(
        (pl.col("late_corr") - pl.col("early_corr")).alias("delta")
    )

    if len(paired) > 0:
        # Summary by regime
        delta_summary = paired.group_by("regime").agg([
            pl.col("delta").mean().round(3).alias("mean_delta"),
            pl.col("early_corr").mean().round(3).alias("mean_early"),
            pl.col("late_corr").mean().round(3).alias("mean_late"),
            pl.len().alias("n_pairs"),
        ])

        print("Correlation change (Δ = late - early) by regime:")
        print(delta_summary)
        print()

        # ANOVA on delta
        septic_delta = paired.filter(pl.col("regime") == "septic")["delta"].drop_nulls().to_numpy()
        stable_delta = paired.filter(pl.col("regime") == "stable")["delta"].drop_nulls().to_numpy()

        if len(septic_delta) > 5 and len(stable_delta) > 5:
            from scipy.stats import f_oneway
            f_stat, p_val = f_oneway(septic_delta, stable_delta)
            print(f"ANOVA on Δ: F = {f_stat:.2f}, p = {p_val:.4f}")
            print(f"  Septic mean Δ: {np.mean(septic_delta):+.3f}")
            print(f"  Stable mean Δ: {np.mean(stable_delta):+.3f}")
            print()

            if np.mean(septic_delta) < np.mean(stable_delta):
                print("  ** Septic patients show MORE NEGATIVE Δ (decoupling over time) **")
            else:
                print("  Septic patients show less negative / more positive Δ")
        print()

        # By pair
        print("=" * 70)
        print("TRAJECTORY BY VITAL PAIR")
        print("=" * 70)
        print()

        print(f"{'Vital Pair':<45} {'Septic Δ':>12} {'Stable Δ':>12} {'Difference':>12}")
        print("-" * 85)

        for pair in paired["pair"].unique().to_list():
            pair_data = paired.filter(pl.col("pair") == pair)

            septic_d = pair_data.filter(pl.col("regime") == "septic")["delta"].mean()
            stable_d = pair_data.filter(pl.col("regime") == "stable")["delta"].mean()

            if septic_d is not None and stable_d is not None:
                diff = septic_d - stable_d
                marker = "←" if diff < -0.05 else ""
                print(f"{pair:<45} {septic_d:>+12.3f} {stable_d:>+12.3f} {diff:>+12.3f} {marker}")

        print()
        print("← = septic shows more decoupling (more negative Δ)")

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("If decoupling is PROGRESSIVE, we expect:")
    print("  - Septic Δ < 0 (correlation decreases over time)")
    print("  - Stable Δ ≈ 0 (correlation stays stable)")
    print("  - Septic Δ < Stable Δ (septic decouples more)")
    print()
    print("This would indicate that vital sign coupling degrades")
    print("DURING the ICU stay in septic patients.")
    print()


if __name__ == "__main__":
    main()
