#!/usr/bin/env python3
"""
MIMIC Early Warning Proof of Concept

For patients with LATE-ONSET infection (>6h after ICU admission):
Track decoupling score trajectory BEFORE infection time.

Question: Does decoupling increase BEFORE sepsis diagnosis?

Expected trajectory (if predictive):
    t=-12h: Score = 0.2 (coupled)
    t=-8h:  Score = 0.3 (starting to decouple)
    t=-4h:  Score = 0.5 (decoupling)
    t=0h:   Score = 0.7 (decoupled at diagnosis)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

# Vitals
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220052: "arterial_bp_mean",
    220181: "nibp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
}
VITALS = list(VITAL_ITEMIDS.values())
WEAK_THRESHOLD = 0.25


def compute_window_decoupling(vital_data, window_start, window_end):
    """Compute decoupling score for a time window."""
    window_data = vital_data.filter(
        (pl.col("charttime") >= window_start) &
        (pl.col("charttime") < window_end)
    )

    if len(window_data) < 20:
        return None

    # Get series for each vital
    vital_series = {}
    for vital in VITALS:
        v_data = window_data.filter(pl.col("vital_name") == vital)
        if len(v_data) >= 5:
            vital_series[vital] = v_data["valuenum"].drop_nulls().to_numpy()

    if len(vital_series) < 2:
        return None

    # Compute correlations
    n_weak = 0
    n_valid = 0
    sum_corr = 0

    for v1, v2 in combinations(vital_series.keys(), 2):
        vals1 = vital_series[v1]
        vals2 = vital_series[v2]
        min_len = min(len(vals1), len(vals2))

        if min_len >= 5:
            try:
                r, _ = pearsonr(vals1[:min_len], vals2[:min_len])
                r_abs = abs(r)
                n_valid += 1
                sum_corr += r_abs
                if r_abs < WEAK_THRESHOLD:
                    n_weak += 1
            except:
                pass

    if n_valid == 0:
        return None

    return {
        "decoupling_score": n_weak / n_valid,
        "mean_coupling": sum_corr / n_valid,
        "n_weak": n_weak,
        "n_valid": n_valid,
    }


def main():
    data_dir = Path("data/mimic_demo")
    base = data_dir / "mimic-iv-clinical-database-demo-2.2"

    print("=" * 70)
    print("MIMIC Early Warning Proof of Concept")
    print("=" * 70)
    print()
    print("Tracking decoupling score BEFORE infection onset")
    print("for patients with late-onset infection (>6h after ICU admission)")
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

    # Get sepsis patients
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]
    sepsis_hadms = diagnoses.filter(
        pl.col("icd_code").is_in(sepsis_codes)
    )["hadm_id"].unique().to_list()

    # Find infection onset time for each septic stay
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

    # Categorize by timing
    late_onset_stays = []
    for stay_id, info in infection_times.items():
        if info["hours_after_icu"] > 6:
            late_onset_stays.append((stay_id, info))

    print(f"\nTotal septic stays with timing: {len(infection_times)}")
    print(f"Late-onset (>6h after ICU): {len(late_onset_stays)}")

    if len(late_onset_stays) == 0:
        print("\nNo late-onset patients available for trajectory analysis.")
        print("This is a limitation of the demo dataset.")
        return

    print()

    # Track decoupling trajectory for late-onset patients
    print("=" * 70)
    print("TRAJECTORY ANALYSIS: Decoupling Before Infection")
    print("=" * 70)
    print()

    # Time bins relative to infection onset
    time_bins = [
        (-24, -20, "t=-22h"),
        (-20, -16, "t=-18h"),
        (-16, -12, "t=-14h"),
        (-12, -8, "t=-10h"),
        (-8, -4, "t=-6h"),
        (-4, 0, "t=-2h"),
        (0, 4, "t=+2h"),
    ]

    all_trajectories = []

    for stay_id, info in late_onset_stays:
        onset_time = info["onset_time"]
        icu_intime = info["icu_intime"]

        # Get vital data for this stay
        stay_vitals = chartevents.filter(pl.col("stay_id") == stay_id)

        if len(stay_vitals) < 50:
            continue

        trajectory = {"stay_id": stay_id, "hours_after_icu": info["hours_after_icu"]}

        for start_offset, end_offset, label in time_bins:
            window_start = onset_time + timedelta(hours=start_offset)
            window_end = onset_time + timedelta(hours=end_offset)

            # Skip if window is before ICU admission
            if window_start < icu_intime:
                continue

            result = compute_window_decoupling(stay_vitals, window_start, window_end)

            if result is not None:
                trajectory[f"{label}_score"] = result["decoupling_score"]
                trajectory[f"{label}_coupling"] = result["mean_coupling"]

        if len(trajectory) > 2:  # Has at least some time points
            all_trajectories.append(trajectory)

    print(f"Trajectories computed for {len(all_trajectories)} late-onset patients")
    print()

    if len(all_trajectories) == 0:
        print("Insufficient data for trajectory analysis.")
        return

    # Print individual trajectories
    print("Individual Patient Trajectories (Decoupling Score):")
    print()
    print(f"{'Stay':<15} {'t=-22h':>8} {'t=-18h':>8} {'t=-14h':>8} {'t=-10h':>8} {'t=-6h':>8} {'t=-2h':>8} {'t=+2h':>8}")
    print("-" * 85)

    for traj in all_trajectories:
        row = f"{traj['stay_id']:<15}"
        for label in ["t=-22h", "t=-18h", "t=-14h", "t=-10h", "t=-6h", "t=-2h", "t=+2h"]:
            key = f"{label}_score"
            if key in traj:
                row += f" {traj[key]:>7.2f}"
            else:
                row += f" {'---':>7}"
        print(row)

    print()

    # Aggregate trajectory
    print("=" * 70)
    print("AGGREGATE TRAJECTORY (Mean Across Patients)")
    print("=" * 70)
    print()

    print(f"{'Time':>10} {'Mean Score':>12} {'Mean |r|':>12} {'n patients':>12}")
    print("-" * 50)

    for _, _, label in time_bins:
        scores = [t[f"{label}_score"] for t in all_trajectories if f"{label}_score" in t]
        couplings = [t[f"{label}_coupling"] for t in all_trajectories if f"{label}_coupling" in t]

        if scores:
            print(f"{label:>10} {np.mean(scores):>12.3f} {np.mean(couplings):>12.3f} {len(scores):>12}")

    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Look for INCREASING decoupling score approaching t=0 (infection onset):")
    print()
    print("  If predictive: t=-22h < t=-14h < t=-6h < t=0")
    print("                 (score increases as infection approaches)")
    print()
    print("  If not predictive: scores are flat or random")
    print()
    print("NOTE: n is very small in demo data. Full MIMIC-IV needed for")
    print("      statistically robust early warning validation.")
    print()


if __name__ == "__main__":
    main()
