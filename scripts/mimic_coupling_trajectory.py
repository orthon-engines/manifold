#!/usr/bin/env python3
"""
MIMIC Coupling Trajectory Study

THE CORRECT QUESTION:
    Does the RATE OF CHANGE in vital sign coupling predict sepsis
    BEFORE traditional thresholds trigger?

NOT: "Is HR high?"                    (absolute value)
NOT: "Is BP low?"                     (absolute value)
NOT: "Is HR-BP correlation weak?"     (absolute coupling)

BUT: "Is HR-BP correlation DECREASING over time?"  (first derivative)
     "Is the decoupling ACCELERATING?"             (second derivative)

METHODOLOGY (from literature):
1. Ground truth = Sepsis-3 onset time (SOFA +2 AND suspicion of infection)
2. Compute coupling in sliding windows BEFORE onset
3. Measure the SLOPE of coupling over time (rate of change)
4. Compare: Do patients who develop sepsis show coupling decay
   BEFORE their vitals individually breach thresholds?

References:
- PhysioNet/CinC 2019 Challenge (Reyna et al.)
- Bloch et al. 2019: Second-order derived features achieve AUC 0.94
- Seymour et al. 2019: Vital sign trajectory subphenotyping
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr, linregress, ttest_ind
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

# Vitals (no GCS)
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220052: "arterial_bp_mean",
    220181: "nibp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
}
VITALS = list(VITAL_ITEMIDS.values())


def compute_window_coupling(vital_data, window_start, window_end):
    """Compute mean |correlation| for all vital pairs in a time window."""
    window_data = vital_data.filter(
        (pl.col("charttime") >= window_start) &
        (pl.col("charttime") < window_end)
    )

    if len(window_data) < 10:
        return None, None

    # Get series for each vital
    vital_series = {}
    for vital in VITALS:
        v_data = window_data.filter(pl.col("vital_name") == vital)
        if len(v_data) >= 3:
            vital_series[vital] = v_data["valuenum"].drop_nulls().to_numpy()

    if len(vital_series) < 2:
        return None, None

    # Compute correlations
    correlations = []
    for v1, v2 in combinations(vital_series.keys(), 2):
        vals1 = vital_series[v1]
        vals2 = vital_series[v2]
        min_len = min(len(vals1), len(vals2))

        if min_len >= 3:
            try:
                r, _ = pearsonr(vals1[:min_len], vals2[:min_len])
                if not np.isnan(r):
                    correlations.append(abs(r))
            except:
                pass

    if len(correlations) == 0:
        return None, None

    return np.mean(correlations), len(correlations)


def compute_coupling_trajectory(vital_data, reference_time, icu_intime,
                                 window_hours=4, step_hours=2, lookback_hours=24):
    """
    Compute coupling values in sliding windows leading up to reference_time.
    Returns list of (hours_before_reference, mean_coupling) tuples.
    """
    trajectory = []

    # Start from lookback_hours before reference, step forward
    current_end = reference_time - timedelta(hours=lookback_hours)

    while current_end <= reference_time:
        window_start = current_end - timedelta(hours=window_hours)
        window_end = current_end

        # Skip if window is before ICU admission
        if window_start < icu_intime:
            current_end += timedelta(hours=step_hours)
            continue

        coupling, n_pairs = compute_window_coupling(vital_data, window_start, window_end)

        if coupling is not None:
            hours_before = (reference_time - window_end).total_seconds() / 3600
            trajectory.append({
                "hours_before": hours_before,
                "coupling": coupling,
                "n_pairs": n_pairs,
            })

        current_end += timedelta(hours=step_hours)

    return trajectory


def compute_trajectory_slope(trajectory):
    """
    Compute the slope of coupling over time.
    Negative slope = decoupling (correlation decreasing as sepsis approaches)
    """
    if len(trajectory) < 3:
        return None, None

    # x = hours before (negative direction: -24, -22, -20, ..., -2, 0)
    # We want to measure: as time progresses toward event, does coupling decrease?
    # So x should be ordered chronologically

    # Sort by hours_before descending (earliest first)
    sorted_traj = sorted(trajectory, key=lambda x: -x["hours_before"])

    x = np.array([t["hours_before"] for t in sorted_traj])
    y = np.array([t["coupling"] for t in sorted_traj])

    # Flip x so it's in natural time order (approaching 0)
    # x goes from high (far from event) to low (close to event)
    # If coupling decreases as we approach event, slope of y vs (-x) should be negative
    # Or equivalently: slope of y vs x should be positive (since x decreases)

    try:
        slope, intercept, r_value, p_value, std_err = linregress(-x, y)
        # slope > 0 means coupling INCREASES as we approach event (unusual)
        # slope < 0 means coupling DECREASES as we approach event (decoupling)
        return slope, r_value ** 2
    except:
        return None, None


def main():
    data_dir = Path("data/mimic_demo")
    base = data_dir / "mimic-iv-clinical-database-demo-2.2"

    print("=" * 70)
    print("COUPLING TRAJECTORY STUDY")
    print("=" * 70)
    print()
    print("QUESTION: Does the RATE OF CHANGE in vital sign coupling")
    print("          predict sepsis BEFORE traditional thresholds?")
    print()
    print("METHODOLOGY:")
    print("  1. Identify sepsis onset time (Sepsis-3 proxy)")
    print("  2. Compute coupling in sliding windows BEFORE onset")
    print("  3. Measure SLOPE of coupling trajectory")
    print("  4. Negative slope = decoupling (early warning signal)")
    print()

    # Load data
    print("Loading data...")

    chartevents = pl.read_csv(
        base / "icu" / "chartevents.csv.gz",
        columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        schema_overrides={"valuenum": pl.Float64}
    )
    chartevents = chartevents.filter(pl.col("itemid").is_in(list(VITAL_ITEMIDS.keys())))
    vital_map = pl.DataFrame({
        "itemid": list(VITAL_ITEMIDS.keys()),
        "vital_name": list(VITAL_ITEMIDS.values())
    })
    chartevents = chartevents.join(vital_map, on="itemid", how="left")
    chartevents = chartevents.with_columns(
        pl.col("charttime").str.to_datetime().alias("charttime")
    )

    icustays = pl.read_csv(base / "icu" / "icustays.csv.gz")
    micro = pl.read_csv(base / "hosp" / "microbiologyevents.csv.gz")
    prescriptions = pl.read_csv(
        base / "hosp" / "prescriptions.csv.gz",
        infer_schema_length=10000,
        ignore_errors=True
    )
    diagnoses = pl.read_csv(base / "hosp" / "diagnoses_icd.csv.gz")

    # Get sepsis patients (ICD codes as proxy for Sepsis-3)
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]
    sepsis_hadms = diagnoses.filter(
        pl.col("icd_code").is_in(sepsis_codes)
    )["hadm_id"].unique().to_list()

    # Find infection onset time for septic stays
    # (Using culture/antibiotic time as Sepsis-3 proxy)
    print("\nIdentifying infection onset times...")

    infection_times = {}
    sepsis_stays = icustays.filter(pl.col("hadm_id").is_in(sepsis_hadms))

    for row in sepsis_stays.iter_rows(named=True):
        stay_id = row["stay_id"]
        hadm_id = row["hadm_id"]
        icu_intime = row["intime"]

        if isinstance(icu_intime, str):
            icu_intime = datetime.fromisoformat(icu_intime.replace(" ", "T"))

        onset_time = None

        # Check microbiology
        patient_micro = micro.filter(pl.col("hadm_id") == hadm_id)
        if len(patient_micro) > 0 and "charttime" in patient_micro.columns:
            culture_times = patient_micro["charttime"].drop_nulls().to_list()
            if culture_times:
                first_culture = min(culture_times)
                if isinstance(first_culture, str):
                    first_culture = datetime.fromisoformat(first_culture.replace(" ", "T"))
                onset_time = first_culture

        # Check prescriptions for antibiotics
        if prescriptions is not None:
            patient_rx = prescriptions.filter(pl.col("hadm_id") == hadm_id)
            if len(patient_rx) > 0 and "starttime" in patient_rx.columns and "drug" in patient_rx.columns:
                abx_terms = ["cillin", "mycin", "floxacin", "cef", "vanco", "metro", "zosyn", "meropenem"]
                for term in abx_terms:
                    try:
                        abx_rx = patient_rx.filter(pl.col("drug").str.to_lowercase().str.contains(term))
                        if len(abx_rx) > 0:
                            abx_times = abx_rx["starttime"].drop_nulls().to_list()
                            if abx_times:
                                first_abx = min(abx_times)
                                if isinstance(first_abx, str):
                                    first_abx = datetime.fromisoformat(first_abx.replace(" ", "T"))
                                if onset_time is None or first_abx < onset_time:
                                    onset_time = first_abx
                    except:
                        pass

        if onset_time is not None:
            hours_after_icu = (onset_time - icu_intime).total_seconds() / 3600
            infection_times[stay_id] = {
                "onset_time": onset_time,
                "icu_intime": icu_intime,
                "hours_after_icu": hours_after_icu,
            }

    # Get non-septic stays
    non_septic_hadms = set(icustays["hadm_id"].to_list()) - set(sepsis_hadms)
    non_septic_stays = icustays.filter(pl.col("hadm_id").is_in(list(non_septic_hadms)))

    # Categorize septic patients
    early_onset = []  # < 6h after ICU
    late_onset = []   # >= 6h after ICU

    for stay_id, info in infection_times.items():
        if info["hours_after_icu"] >= 6:
            late_onset.append((stay_id, info))
        else:
            early_onset.append((stay_id, info))

    print(f"\nSeptic stays with timing: {len(infection_times)}")
    print(f"  Early onset (<6h): {len(early_onset)}")
    print(f"  Late onset (>=6h): {len(late_onset)} <- THESE ARE ANALYZABLE")
    print(f"Non-septic stays: {len(non_septic_stays)}")

    print()
    print("=" * 70)
    print("COUPLING TRAJECTORY ANALYSIS")
    print("=" * 70)
    print()
    print("For late-onset septic patients, computing coupling trajectory")
    print("in 4-hour windows, stepping by 2 hours, up to 24h before onset.")
    print()

    # Compute trajectories for late-onset patients
    septic_trajectories = []
    septic_slopes = []

    for stay_id, info in late_onset:
        stay_vitals = chartevents.filter(pl.col("stay_id") == stay_id)

        if len(stay_vitals) < 30:
            continue

        trajectory = compute_coupling_trajectory(
            stay_vitals,
            info["onset_time"],
            info["icu_intime"],
            window_hours=4,
            step_hours=2,
            lookback_hours=24
        )

        if len(trajectory) >= 3:
            slope, r2 = compute_trajectory_slope(trajectory)
            if slope is not None:
                septic_trajectories.append({
                    "stay_id": stay_id,
                    "trajectory": trajectory,
                    "slope": slope,
                    "r2": r2,
                    "label": "septic",
                })
                septic_slopes.append(slope)

    print(f"Late-onset septic patients with valid trajectories: {len(septic_trajectories)}")

    # For comparison: compute trajectories for non-septic patients
    # Use ICU discharge as reference time (since they don't have onset)
    stable_trajectories = []
    stable_slopes = []

    sample_stable = non_septic_stays.head(100)  # Sample for comparison

    for row in sample_stable.iter_rows(named=True):
        stay_id = row["stay_id"]
        icu_intime = row["intime"]
        icu_outtime = row["outtime"]

        if isinstance(icu_intime, str):
            icu_intime = datetime.fromisoformat(icu_intime.replace(" ", "T"))
        if isinstance(icu_outtime, str):
            icu_outtime = datetime.fromisoformat(icu_outtime.replace(" ", "T"))

        # Need at least 24h of data
        stay_duration = (icu_outtime - icu_intime).total_seconds() / 3600
        if stay_duration < 24:
            continue

        stay_vitals = chartevents.filter(pl.col("stay_id") == stay_id)

        if len(stay_vitals) < 30:
            continue

        # Use midpoint of stay as reference
        reference_time = icu_intime + timedelta(hours=stay_duration / 2)

        trajectory = compute_coupling_trajectory(
            stay_vitals,
            reference_time,
            icu_intime,
            window_hours=4,
            step_hours=2,
            lookback_hours=24
        )

        if len(trajectory) >= 3:
            slope, r2 = compute_trajectory_slope(trajectory)
            if slope is not None:
                stable_trajectories.append({
                    "stay_id": stay_id,
                    "trajectory": trajectory,
                    "slope": slope,
                    "r2": r2,
                    "label": "stable",
                })
                stable_slopes.append(slope)

    print(f"Non-septic patients with valid trajectories: {len(stable_trajectories)}")

    print()
    print("=" * 70)
    print("RATE OF CHANGE COMPARISON")
    print("=" * 70)
    print()
    print("Slope interpretation:")
    print("  Negative slope = coupling DECREASING as time approaches reference")
    print("                 = DECOUPLING (expected early warning signal)")
    print("  Positive slope = coupling INCREASING")
    print()

    if len(septic_slopes) > 0 and len(stable_slopes) > 0:
        septic_mean = np.mean(septic_slopes)
        stable_mean = np.mean(stable_slopes)
        septic_std = np.std(septic_slopes)
        stable_std = np.std(stable_slopes)

        t_stat, p_value = ttest_ind(septic_slopes, stable_slopes)

        print(f"Septic (n={len(septic_slopes)}):  Mean slope = {septic_mean:+.4f} (SD: {septic_std:.4f})")
        print(f"Stable (n={len(stable_slopes)}):  Mean slope = {stable_mean:+.4f} (SD: {stable_std:.4f})")
        print()
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")
        print()

        if septic_mean < stable_mean:
            print("RESULT: Septic patients show MORE NEGATIVE slopes (more decoupling)")
        else:
            print("RESULT: No clear difference in coupling trajectory")

        # Count how many show negative slope (decoupling)
        septic_decoupling = sum(1 for s in septic_slopes if s < 0)
        stable_decoupling = sum(1 for s in stable_slopes if s < 0)

        print()
        print(f"Patients showing decoupling (negative slope):")
        print(f"  Septic: {septic_decoupling}/{len(septic_slopes)} ({100*septic_decoupling/len(septic_slopes):.1f}%)")
        print(f"  Stable: {stable_decoupling}/{len(stable_slopes)} ({100*stable_decoupling/len(stable_slopes):.1f}%)")

    else:
        print("Insufficient data for statistical comparison.")

    print()
    print("=" * 70)
    print("INDIVIDUAL TRAJECTORIES (Septic Late-Onset)")
    print("=" * 70)
    print()

    for traj in septic_trajectories[:5]:  # Show first 5
        print(f"Stay {traj['stay_id']} - Slope: {traj['slope']:+.4f}, R²: {traj['r2']:.3f}")
        for point in sorted(traj['trajectory'], key=lambda x: -x['hours_before']):
            bar = "█" * int(point['coupling'] * 20)
            print(f"  t={-point['hours_before']:+5.1f}h: {point['coupling']:.3f} {bar}")
        print()

    print()
    print("=" * 70)
    print("STUDY LIMITATIONS")
    print("=" * 70)
    print()
    print("1. MIMIC Demo has very few late-onset patients (most infected at admission)")
    print("2. Sepsis-3 timing is approximated from culture/antibiotic orders")
    print("3. True early warning requires SOFA scoring (not available in demo)")
    print("4. Full MIMIC-IV with PhysioNet 2019 methodology needed for validation")
    print()
    print("REFERENCES:")
    print("  - PhysioNet/CinC 2019 Challenge (Reyna et al.)")
    print("  - Bloch et al. 2019: Second-order derived features, AUC 0.94")
    print("  - Seymour et al. 2019: Vital sign trajectory subphenotyping")
    print()


if __name__ == "__main__":
    main()
