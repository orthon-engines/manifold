# Claude AI Memory - PRISM/ORTHON Architecture

**Last Updated:** 2026-02-05
**Session Summary:** Created all 27 PRISM entry points (14 core + 13 granular) + normalization engine + stability pipeline

---

## Architecture Principle

```
PRISM = Muscle (pure computation, no decisions, no classification)
ORTHON = Brain (orchestration, typology, classification, interpretation)

PRISM computes numbers. ORTHON classifies.
```

---

## PRISM Entry Points (Complete: 27 stages)

Location: `/Users/jasonrudder/prism/prism/entry_points/`

### Core Stages (14)

| Stage | File | Output | Engine |
|-------|------|--------|--------|
| 01 | `stage_01_signal_vector.py` | `signal_vector.parquet` | `engines/signal/*` |
| 02 | `stage_02_state_vector.py` | `state_vector.parquet` | `engines/state/centroid.py` |
| 03 | `stage_03_state_geometry.py` | `state_geometry.parquet` | `engines/state/eigendecomp.py` |
| 04 | `stage_04_cohorts.py` | `cohorts.parquet` | Pure aggregation |
| 05 | `stage_05_signal_geometry.py` | `signal_geometry.parquet` | `engines/signal_geometry.py` |
| 06 | `stage_06_signal_pairwise.py` | `signal_pairwise.parquet` | `engines/signal_pairwise.py` |
| 07 | `stage_07_geometry_dynamics.py` | `geometry_dynamics.parquet` | `engines/geometry_dynamics.py` |
| 08 | `stage_08_lyapunov.py` | `lyapunov.parquet` | `engines/dynamics/lyapunov.py` |
| 09 | `stage_09_dynamics.py` | `dynamics.parquet` | `engines/dynamics/*` |
| 10 | `stage_10_information_flow.py` | `information_flow.parquet` | `engines/pairwise/causality.py` |
| 11 | `stage_11_topology.py` | `topology.parquet` | Basic topology |
| 12 | `stage_12_zscore.py` | `zscore.parquet` | `engines/sql/zscore.sql` |
| 13 | `stage_13_statistics.py` | `statistics.parquet` | `engines/sql/statistics.sql` |
| 14 | `stage_14_correlation.py` | `correlation.parquet` | `engines/sql/correlation.sql` |

### Granular Substages (13)

| Stage | File | Output | Description |
|-------|------|--------|-------------|
| 02a | `stage_02a_observations_windowed.py` | `observations_windowed.parquet` | Windowed observations |
| 03a | `stage_03a_signal_statistics.py` | `signal_statistics.parquet` | kurtosis, skewness, crest_factor |
| 03b | `stage_03b_signal_temporal.py` | `signal_temporal.parquet` | trend, rate_of_change |
| 03c | `stage_03c_signal_spectral.py` | `signal_spectral.parquet` | dominant_freq, spectral_entropy |
| 03d | `stage_03d_signal_complexity.py` | `signal_complexity.parquet` | sample_entropy, perm_entropy |
| 03e | `stage_03e_signal_stationarity.py` | `signal_stationarity.parquet` | ADF, variance_ratio, hurst |
| 05a | `stage_05a_state_correlation.py` | `state_correlation.parquet` | State component correlations |
| 05b | `stage_05b_signal_pairwise_detail.py` | `signal_pairwise_summary.parquet`, `signal_pairwise_detail.parquet` | Cross-correlation, lead/lag |
| 05c | `stage_05c_state_aggregate.py` | `state_aggregate.parquet` | Per-cohort state stats |
| 08a | `stage_08a_cohort_discovery.py` | `cohort_discovery.parquet` | Clustering signals |
| 08b | `stage_08b_cohort_membership.py` | `cohort_membership.parquet`, `signal_cohort_waterfall.parquet` | Membership tracking |
| 08c | `stage_08c_cohort_evolution.py` | `cohort_evolution.parquet` | Cohort dynamics |
| 09a | `stage_09a_cohort_thermodynamics.py` | `cohort_thermodynamics.parquet` | Entropy, energy, temperature |

---

## ORTHON Entry Points (Complete: 5 stages)

Location: `/Users/jasonrudder/orthon/orthon/entry_points/`

| Stage | File | Output | Module |
|-------|------|--------|--------|
| 01 | `stage_01_validate.py` | `observations_validated.parquet` | `core/validation.py` |
| 02 | `stage_02_typology.py` | `typology_raw.parquet` | `ingest/typology_raw.py` |
| 03 | `stage_03_classify.py` | `typology.parquet` | `typology/discrete_sparse.py`, `typology/level2_corrections.py` |
| 04 | `stage_04_manifest.py` | `manifest.yaml` | `manifest/generator.py` |
| 05 | `stage_05_diagnostic.py` | `diagnostic_report.txt` | `engines/*` |

---

## Pipeline Flow

```
ORTHON Pipeline:
observations.parquet
    → stage_01_validate → observations_validated.parquet
    → stage_02_typology → typology_raw.parquet
    → stage_03_classify → typology.parquet
    → stage_04_manifest → manifest.yaml

PRISM Pipeline (Core):
observations.parquet + typology.parquet + manifest.yaml
    → stage_01_signal_vector → signal_vector.parquet
    → stage_02_state_vector → state_vector.parquet
    → stage_03_state_geometry → state_geometry.parquet
    → stage_04_cohorts → cohorts.parquet
    → stage_05_signal_geometry → signal_geometry.parquet
    → stage_06_signal_pairwise → signal_pairwise.parquet
    → stage_07_geometry_dynamics → geometry_dynamics.parquet
    → stage_08_lyapunov → lyapunov.parquet
    → stage_09_dynamics → dynamics.parquet
    → stage_10_information_flow → information_flow.parquet
    → stage_11_topology → topology.parquet
    → stage_12_zscore → zscore.parquet
    → stage_13_statistics → statistics.parquet
    → stage_14_correlation → correlation.parquet

PRISM Pipeline (Granular - runs in parallel with core):
    → stage_02a_observations_windowed → observations_windowed.parquet
    → stage_03a_signal_statistics → signal_statistics.parquet
    → stage_03b_signal_temporal → signal_temporal.parquet
    → stage_03c_signal_spectral → signal_spectral.parquet
    → stage_03d_signal_complexity → signal_complexity.parquet
    → stage_03e_signal_stationarity → signal_stationarity.parquet
    → stage_05a_state_correlation → state_correlation.parquet
    → stage_05b_signal_pairwise_detail → signal_pairwise_*.parquet
    → stage_05c_state_aggregate → state_aggregate.parquet
    → stage_08a_cohort_discovery → cohort_discovery.parquet
    → stage_08b_cohort_membership → cohort_membership.parquet
    → stage_08c_cohort_evolution → cohort_evolution.parquet
    → stage_09a_cohort_thermodynamics → cohort_thermodynamics.parquet

Back to ORTHON:
PRISM outputs → stage_05_diagnostic → diagnostic_report.txt
```

---

## Normalization (v2.5)

**Documentation:** `prism/docs/NORMALIZATION.md`

| Method | Robustness | Use When |
|--------|------------|----------|
| zscore | Low | Clean Gaussian data |
| robust | Medium | Some outliers |
| **mad** | High | Industrial, unknown distribution |
| none | N/A | Preserve variance dynamics |

**Files:**
- `prism/engines/normalization.py` - Core engine
- `prism/engines/sql/mad_anomaly.sql` - Robust anomaly detection
- `prism/entry_points/stage_03_state_geometry.py` - `--norm` option

---

## Stability Pipeline

**File:** `prism/entry_points/stability_pipeline.py`

Combines multiple stages for full dynamical systems assessment:
- Signal geometry (cross-signal eigenstructure)
- Lyapunov exponents (stability)
- Critical slowing down (early warning)
- Formal assessment (classification)

```bash
python -m prism.entry_points.stability_pipeline observations.parquet -o results/
```

---

## Key Concepts

- **Structure = Geometry × Mass** - Both can fail independently
- **B-tipping** (geometry→mass): CSD provides early warning
- **R-tipping** (mass→geometry): NO early warning
- **effective_dim**: Participation ratio - 63% importance in RUL prediction

---

## Commands

```bash
# Run individual PRISM core stages
python -m prism.entry_points.stage_01_signal_vector manifest.yaml
python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet --norm mad
python -m prism.entry_points.stage_08_lyapunov observations.parquet -o lyapunov.parquet

# Run granular substages
python -m prism.entry_points.stage_03a_signal_statistics observations.parquet
python -m prism.entry_points.stage_03b_signal_temporal observations.parquet
python -m prism.entry_points.stage_03c_signal_spectral observations.parquet
python -m prism.entry_points.stage_03d_signal_complexity observations.parquet
python -m prism.entry_points.stage_03e_signal_stationarity observations.parquet
python -m prism.entry_points.stage_08a_cohort_discovery signal_vector.parquet
python -m prism.entry_points.stage_09a_cohort_thermodynamics state_geometry.parquet

# Run ORTHON stages
python -m orthon.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m orthon.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
```
