#!/usr/bin/env python3
"""
MIMIC Decoupling Score

Operationalizes the decoupling hypothesis into a concrete risk metric:

    Decoupling Score = # pairs with correlation drop > threshold
                       ─────────────────────────────────────────
                                    total pairs

Rule: IF ≥4 of 10 vital pairs show correlation drop >15% over 4 hours
      THEN: Sepsis risk elevated

Tests whether this score discriminates septic vs stable patients.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr, f_oneway
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# =============================================================================
# Configuration
# =============================================================================

# 5 vitals → 10 pairs
VITALS = ["heart_rate", "arterial_bp_mean", "nibp_mean", "respiratory_rate", "spo2"]

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220052: "arterial_bp_mean",
    220181: "nibp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
}

# Thresholds
CORRELATION_DROP_THRESHOLD = 0.15  # 15% drop
MIN_PAIRS_FOR_RISK = 4  # ≥4 of 10 pairs
WINDOW_HOURS = 12  # Compare 12-hour windows (first 12h vs 12-24h)


# =============================================================================
# Score Computation
# =============================================================================

def compute_decoupling_score(chartevents, stay_id, icu_intime, window_hours=4, drop_threshold=0.15):
    """
    Compute decoupling score for a single ICU stay.

    TWO SCORES:
    1. Temporal: Compares early vs late window correlations
    2. Absolute: Counts pairs with |correlation| < 0.25 (weak coupling)

    Returns:
        dict with scores
    """
    # Get stay's vital data
    stay_vitals = chartevents.filter(pl.col("stay_id") == stay_id)

    if len(stay_vitals) < 50:
        return None

    # Define time windows
    early_start = icu_intime
    early_end = icu_intime + timedelta(hours=window_hours)
    late_start = icu_intime + timedelta(hours=window_hours)
    late_end = icu_intime + timedelta(hours=window_hours * 2)

    # Get vital series for each window
    def get_vital_series(data, start, end):
        window_data = data.filter(
            (pl.col("charttime") >= start) &
            (pl.col("charttime") < end)
        )

        vital_series = {}
        for vital in VITALS:
            v_data = window_data.filter(pl.col("vital_name") == vital)
            if len(v_data) >= 5:
                vital_series[vital] = v_data["valuenum"].drop_nulls().to_numpy()
        return vital_series

    early_series = get_vital_series(stay_vitals, early_start, early_end)
    late_series = get_vital_series(stay_vitals, late_start, late_end)

    # Also get full stay data for absolute score
    full_series = {}
    for vital in VITALS:
        v_data = stay_vitals.filter(pl.col("vital_name") == vital)
        if len(v_data) >= 10:
            full_series[vital] = v_data["valuenum"].drop_nulls().to_numpy()

    # Compute correlations for all pairs
    all_pairs = list(combinations(VITALS, 2))
    pair_results = []
    n_dropped = 0
    n_valid = 0
    n_weak_coupling = 0
    n_full_valid = 0
    sum_abs_corr = 0

    WEAK_THRESHOLD = 0.25  # Pairs with |r| < 0.25 are "weakly coupled"

    for v1, v2 in all_pairs:
        # Early correlation
        if v1 in early_series and v2 in early_series:
            e1, e2 = early_series[v1], early_series[v2]
            min_len = min(len(e1), len(e2))
            if min_len >= 5:
                try:
                    early_corr, _ = pearsonr(e1[:min_len], e2[:min_len])
                    early_corr = abs(early_corr)
                except:
                    early_corr = np.nan
            else:
                early_corr = np.nan
        else:
            early_corr = np.nan

        # Late correlation
        if v1 in late_series and v2 in late_series:
            l1, l2 = late_series[v1], late_series[v2]
            min_len = min(len(l1), len(l2))
            if min_len >= 5:
                try:
                    late_corr, _ = pearsonr(l1[:min_len], l2[:min_len])
                    late_corr = abs(late_corr)
                except:
                    late_corr = np.nan
            else:
                late_corr = np.nan
        else:
            late_corr = np.nan

        # Full stay correlation (for absolute score)
        if v1 in full_series and v2 in full_series:
            f1, f2 = full_series[v1], full_series[v2]
            min_len = min(len(f1), len(f2))
            if min_len >= 10:
                try:
                    full_corr, _ = pearsonr(f1[:min_len], f2[:min_len])
                    full_corr_abs = abs(full_corr)
                    n_full_valid += 1
                    sum_abs_corr += full_corr_abs
                    if full_corr_abs < WEAK_THRESHOLD:
                        n_weak_coupling += 1
                except:
                    pass

        # Check for temporal drop
        if not np.isnan(early_corr) and not np.isnan(late_corr):
            n_valid += 1

            if early_corr > 0.01:
                relative_drop = (early_corr - late_corr) / early_corr
            else:
                relative_drop = 0

            dropped = relative_drop > drop_threshold
            if dropped:
                n_dropped += 1

            pair_results.append({
                "pair": f"{v1}___{v2}",
                "early_corr": early_corr,
                "late_corr": late_corr,
                "relative_drop": relative_drop,
                "dropped": dropped,
            })

    # Temporal score (original)
    temporal_score = n_dropped / n_valid if n_valid > 0 else np.nan
    temporal_risk = n_dropped >= MIN_PAIRS_FOR_RISK if n_valid > 0 else False

    # Absolute score (NEW): fraction of pairs with weak coupling
    absolute_score = n_weak_coupling / n_full_valid if n_full_valid > 0 else np.nan
    mean_coupling = sum_abs_corr / n_full_valid if n_full_valid > 0 else np.nan
    absolute_risk = n_weak_coupling >= MIN_PAIRS_FOR_RISK if n_full_valid > 0 else False

    return {
        "temporal_score": temporal_score,
        "n_dropped_pairs": n_dropped,
        "n_valid_pairs": n_valid,
        "temporal_risk": temporal_risk,
        "absolute_score": absolute_score,
        "n_weak_pairs": n_weak_coupling,
        "n_full_valid": n_full_valid,
        "mean_coupling": mean_coupling,
        "absolute_risk": absolute_risk,
    }


def main():
    data_dir = Path("data/mimic_demo")

    print("=" * 70)
    print("MIMIC Decoupling Score Analysis")
    print("=" * 70)
    print()
    print("Definition:")
    print(f"  Decoupling Score = # pairs with |corr| drop > {CORRELATION_DROP_THRESHOLD*100:.0f}%")
    print(f"                     ─────────────────────────────────────────")
    print(f"                                  total pairs")
    print()
    print(f"  Risk Rule: IF ≥{MIN_PAIRS_FOR_RISK} of 10 pairs drop > {CORRELATION_DROP_THRESHOLD*100:.0f}%")
    print(f"             THEN: Sepsis risk elevated")
    print()
    print(f"  Window: Compare {WINDOW_HOURS}h early vs {WINDOW_HOURS}h late")
    print()

    # Load data
    print("Loading data...")
    base = data_dir / "mimic-iv-clinical-database-demo-2.2"

    chartevents = pl.read_csv(
        base / "icu" / "chartevents.csv.gz",
        columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        schema_overrides={"valuenum": pl.Float64}
    )

    # Filter to vitals
    chartevents = chartevents.filter(pl.col("itemid").is_in(list(VITAL_ITEMIDS.keys())))

    # Add vital name
    vital_map = pl.DataFrame({
        "itemid": list(VITAL_ITEMIDS.keys()),
        "vital_name": list(VITAL_ITEMIDS.values())
    })
    chartevents = chartevents.join(vital_map, on="itemid", how="left")
    chartevents = chartevents.with_columns(
        pl.col("charttime").str.to_datetime().alias("charttime")
    )

    # Load ICU stays
    icustays = pl.read_csv(base / "icu" / "icustays.csv.gz")

    # Load sepsis status
    diagnoses = pl.read_csv(base / "hosp" / "diagnoses_icd.csv.gz")
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]
    sepsis_hadms = diagnoses.filter(
        pl.col("icd_code").is_in(sepsis_codes)
    )["hadm_id"].unique().to_list()

    # Compute decoupling score for each stay
    print(f"\nComputing decoupling scores for {len(icustays)} stays...")

    results = []
    stays = icustays.to_dicts()

    for i, stay in enumerate(stays):
        if i % 20 == 0:
            print(f"  {i}/{len(stays)}...")

        stay_id = stay["stay_id"]
        hadm_id = stay["hadm_id"]
        intime = stay["intime"]

        if isinstance(intime, str):
            intime = datetime.fromisoformat(intime.replace(" ", "T"))

        regime = "septic" if hadm_id in sepsis_hadms else "stable"

        score_result = compute_decoupling_score(
            chartevents, stay_id, intime,
            window_hours=WINDOW_HOURS,
            drop_threshold=CORRELATION_DROP_THRESHOLD
        )

        if score_result is not None:
            results.append({
                "stay_id": stay_id,
                "regime": regime,
                "temporal_score": score_result["temporal_score"],
                "n_dropped_pairs": score_result["n_dropped_pairs"],
                "temporal_risk": score_result["temporal_risk"],
                "absolute_score": score_result["absolute_score"],
                "n_weak_pairs": score_result["n_weak_pairs"],
                "mean_coupling": score_result["mean_coupling"],
                "absolute_risk": score_result["absolute_risk"],
            })

    if not results:
        print("No results computed.")
        return

    df = pl.DataFrame(results)
    print(f"\nComputed scores for {len(df)} stays")

    # Save
    output_dir = data_dir / "geometry"
    output_dir.mkdir(exist_ok=True)
    df.write_parquet(output_dir / "decoupling_scores.parquet")
    print(f"Saved to {output_dir / 'decoupling_scores.parquet'}")
    print()

    # ==========================================================================
    # RESULTS: ABSOLUTE SCORE (Mean Coupling)
    # ==========================================================================
    print("=" * 70)
    print("RESULTS: ABSOLUTE Decoupling Score (Mean Coupling)")
    print("=" * 70)
    print()
    print("Absolute Score = fraction of pairs with |r| < 0.25 (weak coupling)")
    print()

    # Summary by regime
    summary = df.group_by("regime").agg([
        pl.col("mean_coupling").mean().round(3).alias("mean_coupling"),
        pl.col("absolute_score").mean().round(3).alias("mean_abs_score"),
        pl.col("n_weak_pairs").mean().round(2).alias("mean_weak"),
        pl.col("absolute_risk").mean().round(3).alias("abs_risk_rate"),
        pl.len().alias("n"),
    ])

    print("Mean Coupling & Absolute Score by regime:")
    print(summary)
    print()

    # ANOVA on mean coupling
    septic_coupling = df.filter(pl.col("regime") == "septic")["mean_coupling"].drop_nulls().to_numpy()
    stable_coupling = df.filter(pl.col("regime") == "stable")["mean_coupling"].drop_nulls().to_numpy()

    if len(septic_coupling) > 5 and len(stable_coupling) > 5:
        f_stat, p_val = f_oneway(septic_coupling, stable_coupling)
        print(f"ANOVA (Mean Coupling): F = {f_stat:.2f}, p = {p_val:.4f}")
        print(f"  Septic mean: {np.mean(septic_coupling):.3f}")
        print(f"  Stable mean: {np.mean(stable_coupling):.3f}")

        if np.mean(septic_coupling) < np.mean(stable_coupling):
            print("\n  ** Septic shows LOWER coupling (more decoupled) **")
    print()

    # ==========================================================================
    # ABSOLUTE RISK RULE
    # ==========================================================================
    print("=" * 70)
    print(f"ABSOLUTE RISK RULE: ≥{MIN_PAIRS_FOR_RISK} pairs with |r|<0.25 → Risk")
    print("=" * 70)
    print()

    septic_risk = df.filter((pl.col("regime") == "septic") & (pl.col("absolute_risk") == True)).height
    septic_no_risk = df.filter((pl.col("regime") == "septic") & (pl.col("absolute_risk") == False)).height
    stable_risk = df.filter((pl.col("regime") == "stable") & (pl.col("absolute_risk") == True)).height
    stable_no_risk = df.filter((pl.col("regime") == "stable") & (pl.col("absolute_risk") == False)).height

    total_septic = septic_risk + septic_no_risk
    total_stable = stable_risk + stable_no_risk

    print("Confusion Matrix (Absolute Score):")
    print()
    print(f"                    Predicted Risk    Predicted No Risk")
    print(f"  Actual Septic     {septic_risk:^15}   {septic_no_risk:^17}")
    print(f"  Actual Stable     {stable_risk:^15}   {stable_no_risk:^17}")
    print()

    if total_septic > 0 and total_stable > 0:
        sensitivity = septic_risk / total_septic if total_septic > 0 else 0
        specificity = stable_no_risk / total_stable if total_stable > 0 else 0
        ppv = septic_risk / (septic_risk + stable_risk) if (septic_risk + stable_risk) > 0 else 0
        npv = stable_no_risk / (stable_no_risk + septic_no_risk) if (stable_no_risk + septic_no_risk) > 0 else 0

        print(f"Performance (Absolute Score):")
        print(f"  Sensitivity: {sensitivity:.1%}")
        print(f"  Specificity: {specificity:.1%}")
        print(f"  PPV:         {ppv:.1%}")
        print(f"  NPV:         {npv:.1%}")
    print()

    # ==========================================================================
    # RESULTS: TEMPORAL SCORE
    # ==========================================================================
    print("=" * 70)
    print("RESULTS: TEMPORAL Decoupling Score (Correlation Drop)")
    print("=" * 70)
    print()
    print(f"Temporal Score = fraction of pairs with >15% correlation drop")
    print(f"                 (comparing first {WINDOW_HOURS}h vs next {WINDOW_HOURS}h)")
    print()

    # Filter to valid temporal scores
    temporal_df = df.filter(pl.col("temporal_score").is_not_null())

    temporal_summary = temporal_df.group_by("regime").agg([
        pl.col("temporal_score").mean().round(3).alias("mean_temporal"),
        pl.col("n_dropped_pairs").mean().round(2).alias("mean_dropped"),
        pl.col("temporal_risk").mean().round(3).alias("temp_risk_rate"),
        pl.len().alias("n"),
    ])

    print("Temporal Score by regime:")
    print(temporal_summary)
    print()

    # ANOVA on temporal score
    septic_temp = temporal_df.filter(pl.col("regime") == "septic")["temporal_score"].drop_nulls().to_numpy()
    stable_temp = temporal_df.filter(pl.col("regime") == "stable")["temporal_score"].drop_nulls().to_numpy()

    if len(septic_temp) > 5 and len(stable_temp) > 5:
        f_stat, p_val = f_oneway(septic_temp, stable_temp)
        print(f"ANOVA (Temporal Score): F = {f_stat:.2f}, p = {p_val:.4f}")
        print(f"  Septic mean: {np.mean(septic_temp):.3f}")
        print(f"  Stable mean: {np.mean(stable_temp):.3f}")
    print()

    # ==========================================================================
    # INTERPRETATION
    # ==========================================================================
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Two decoupling metrics:")
    print()
    print("1. ABSOLUTE: Mean |correlation| across all vital pairs")
    print("   - Lower = more decoupled (expected in sepsis)")
    print("   - Based on static geometry analysis")
    print()
    print("2. TEMPORAL: Fraction of pairs showing >15% correlation drop")
    print("   - Higher = more decoupling over time")
    print("   - Tests if decoupling is progressive")
    print()


if __name__ == "__main__":
    main()
