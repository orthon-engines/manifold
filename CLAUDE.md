# CLAUDE.md — Manifold

Manifold is a domain-agnostic dynamical systems computation engine.
It reads `observations.parquet` and a `manifest.yaml`, then produces parquet files in organized directories.

**Design principle:** Stages orchestrate. Engines compute. pmtvs does math.
Each layer only calls the layer below it.

**Manifold computes. It never classifies, labels, or interprets.**
No `trajectory_type`. No `stability_class`. No `is_chaotic`.
If you find yourself writing `if signal_type == 'PERIODIC'`, STOP. That belongs in Prime.

---

## Repositories

GitHub org: `rudder-framework`

```
rudder-framework/prime       → interpreter, classification, SQL analysis
rudder-framework/manifold    → THIS REPO — blind compute engine
rudder-framework/pmtvs       → Rust+Python math functions (PyPI: pmtvs)
```

Prime depends on Manifold's output. Manifold never depends on Prime.
Both depend on pmtvs (`pip install pmtvs`). pmtvs depends on nothing.

```
pmtvs   (leaf — no dependencies)
 ↑   ↑
 |   |
Prime  Manifold
 |   ↑
 └───┘    (Prime calls Manifold's pipeline)
```

---

## Architecture: Two Layers

```
manifold/stages/       RUNNERS — orchestrate I/O. Read parquet, call engines, write parquet.
manifold/core/         ENGINES — compute. DataFrames in, DataFrames out. No file I/O.
```

ALL math imports come directly from pmtvs:
```python
from pmtvs import sample_entropy, hurst_exponent, eigendecomposition
```

There is no `manifold/primitives/` directory. It was removed.
Never recreate it. Never create wrapper functions for pmtvs.
If a function name doesn't match, fix the call site or add an alias in pmtvs.

Each layer ONLY calls the layer below it.

| Layer | Input | Returns | Does I/O |
|-------|-------|---------|----------|
| `stages/group/run.py` (runner) | manifest, file paths | writes parquet files | YES |
| `core/*.py` (engine) | config dict, DataFrames | DataFrames | NO |
| `pmtvs` (external package) | numpy arrays | numpy arrays / floats | NO |

Also:
- `manifold/io/` — reader.py, writer.py, manifest.py (all parquet I/O)
- `manifold/validation/` — input checks (sequential signal_0, no nulls, signal_id present)
- `manifold/run.py` — top-level sequencer (dependency-ordered)

### Layer enforcement

```bash
# These should return NOTHING:
grep -rn "from manifold.stages" manifold/core/
grep -rn "manifold.primitives" manifold/
```

Nobody reaches up. Nobody skips a layer.

---

## How to Run

### CLI

```bash
cd ~/manifold
./venv/bin/python -m manifold ~/domains/rossler/train
./venv/bin/python -m manifold ~/domains/cmapss/FD_004/train
./venv/bin/python -m manifold ~/domains/pendulum/train
```

Point at a directory containing `time_observations.parquet` and `manifest.yaml`.
All stages run. No flags needed.

### Library API (how Prime calls Manifold)

```python
from manifold import run

run(
    observations_path="~/domains/rossler/train/time_observations.parquet",
    manifest_path="~/domains/rossler/train/manifest.yaml",
    output_dir="~/domains/rossler/train/time_output",
)
```

Three explicit paths. No guessing. No path discovery.

**Virtual environment:** `./venv/` — always use it. Never create a new one.

**Parallel workers:** `MANIFOLD_WORKERS` env var (default: `0` = auto-detect).

---

## Import Conventions

```python
# Math: always direct from pmtvs
from pmtvs import sample_entropy, hurst_exponent, eigendecomposition
from pmtvs import svd_decomposition as svd
from pmtvs import optimal_delay, optimal_dimension

# Core engines: module imports
from manifold.core.state_geometry import compute_state_geometry
from manifold.core.geometry_dynamics import compute_geometry_dynamics

# Stages import from core, never from other stages
```

---

## Data Model

### Input: observations.parquet (READ ONLY)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Grouping key (engine_1, pump_A). User-defined, never used inside engines. |
| signal_id | String | Required | Which signal (temp, pressure, vibration) |
| signal_0 | Float64 | Required | Canonical sequential index: 0, 1, 2, 3... per signal_id. **NOT a timestamp.** |
| value | Float64 | Required | The measurement |

**signal_0 is sacred.** Manifold groups by `(signal_id, signal_0)` where signal_0 is always sequential integers starting at 0.
Never reindex. Never use timestamps. Never skip values.

**observations.parquet is sacred.** Never modify it. Never normalize it in place.
Never add columns. It is READ ONLY.

### Output: Parquet files in organized directories

All output goes to `<prefix>_output/` matching the observations file prefix
(e.g., `time_observations.parquet` → `time_output/`):

#### `signal/` — "What does each signal look like?"

| File | Stage | Description |
|------|-------|-------------|
| typology_windows.parquet | 00a | Per-window typology metrics (hurst, entropy, spectral, trend) |
| typology_vector.parquet | 00a | Per-signal typology summary (mean, std, cv, varies) |
| signal_vector.parquet | 01 | Per-signal windowed features (FFT, entropy, hurst, kurtosis...) |
| signal_geometry.parquet | 05 | Distance-to-centroid, coherence, projections |
| signal_stability.parquet | 33 | Hilbert + Wavelet analysis |

#### `cohort/` — "What is the system's geometric structure and signal relationships?"

| File | Stage | Description |
|------|-------|-------------|
| cohort_geometry.parquet | 03 | Eigenvalues, effective_dim, condition_number — **THE core output** |
| cohort_vector.parquet | 02 | Cross-signal centroid per window |
| cohort_signal_positions.parquet | 03 | Signal loadings on PCs |
| cohort_feature_loadings.parquet | 03 | Feature loadings on PC1 |
| cohort_pairwise.parquet | 06 | Covariance matrix within each window |
| cohort_information_flow.parquet | 10 | Granger causality, transfer entropy, copula |

#### `cohort/cohort_dynamics/` — "How is each cohort changing over time?"

| File | Stage | Description |
|------|-------|-------------|
| breaks.parquet | 00 | Change-point detection |
| geometry_dynamics.parquet | 07 | Velocity/jerk of geometry evolution |
| ftle.parquet | 08 | Finite-Time Lyapunov Exponents |
| lyapunov.parquet | 08_lyapunov | Largest Lyapunov exponent |
| thermodynamics.parquet | 09a | Temperature, entropy from eigenvalue spectra |
| ftle_field.parquet | 15 | Local FTLE grid |
| ftle_backward.parquet | 17 | Backward FTLE |
| velocity_field.parquet | 21 | State-space velocity field |
| ftle_rolling.parquet | 22 | Time-varying FTLE |
| ridge_proximity.parquet | 23 | Urgency = v · grad(FTLE) |
| persistent_homology.parquet | 36 | Topological persistence (Betti numbers) |

#### `system/` — "How do cohorts compare across the fleet?"

| File | Stage | Description |
|------|-------|-------------|
| system_geometry.parquet | 26 | SVD at cohort scale (Scale 2) |
| system_vector.parquet | 25 | Cohort-level feature vector (Scale 2 input) |
| system_cohort_positions.parquet | 26 | Cohort loadings on PCs |
| system_pairwise.parquet | 27 | Pairwise cohort comparison |
| system_information_flow.parquet | 28 | Granger at system scale |
| trajectory_signatures.parquet | 32 | Per-cohort geometric fingerprint (joined upstream features) |
| trajectory_library.parquet | 32 | DTW-clustered trajectory types (medoids, silhouette) |
| trajectory_match.parquet | 32 | Per-cohort nearest-cluster match + confidence |

#### `system/system_dynamics/` — "How is the fleet evolving over time?"

| File | Stage | Description |
|------|-------|-------------|
| ftle.parquet | 30 | FTLE on cohort trajectories |
| velocity_field.parquet | 31 | Velocity field at fleet scale |

#### `parameterization/` — "Which signals matter most?"

| File | Stage | Description |
|------|-------|-------------|
| signal_dominance.parquet | 03b | Mean/final PC1 loading, dominance rank |

---

## Stages and Groups

Stages live in `manifold/stages/<group>/` and run in dependency order:

| Group | Stages | What it computes |
|-------|--------|-----------------|
| **vector/** | 00a, 00, 01, 33 | Per-signal features (typology, breaks, signal_vector, stability) |
| **geometry/** | 02, 03, 03b, 05, 07 | State vectors, eigendecomposition, signal geometry, dynamics |
| **dynamics/** | 08, 08_lyapunov, 09a, 15, 17, 21, 22, 23, 36 | FTLE, Lyapunov, velocity, topology, thermodynamics |
| **information/** | 06, 10 | Pairwise coupling, Granger causality, transfer entropy |
| **energy/** | 25, 26, 27, 28, 30, 31, 32 | Cohort vectors, system geometry, fleet-scale analysis, trajectory signatures |

---

## Two-Scale Architecture

Same engines run at two scales:

```
Scale 1:  signals → signal_vector → cohort_geometry     (per cohort)
Scale 2:  system_vector → system_geometry               (across cohorts)
```

Eigendecomposition at Scale 1 tells you about individual system geometry.
Eigendecomposition at Scale 2 tells you about fleet-wide patterns.

---

## Engine Minimum Sample Requirements

FFT-based engines require larger windows. This is physics, not a bug.

| Engine | Min Samples | Reason |
|--------|-------------|--------|
| spectral | 64 | FFT resolution |
| hurst | 128 | Long-range dependence |
| sample_entropy | 64 | Statistical validity |
| perm_entropy | 8 | Permutation patterns |
| acf_decay | 16 | Lag structure |
| kurtosis | 4 | 4th moment |

Do NOT lower minimums to fit small windows. The math doesn't work.
Insufficient data → return NaN, never skip.

---

## Rules

### 1. TWO LAYERS, ONE DIRECTION
Runners call engines. Engines call pmtvs. Nobody reaches up.

### 2. ALL MATH FROM PMTVS
`from pmtvs import function_name` — always. No wrappers. No shims.
If a parameter name doesn't match, fix the call site.

### 3. NO CLASSIFICATION IN MANIFOLD
Manifold computes numbers. It never decides what those numbers mean.
No trajectory_type, stability_class, is_chaotic, regime labels.
Classification lives in Prime via SQL.

### 4. OBSERVATIONS ARE SACRED
Never modify observations.parquet. Never normalize it in place.
Never add columns. It is READ ONLY.

### 5. signal_0 IS SEQUENTIAL
signal_0 = 0, 1, 2, 3... per signal_id. Always. No timestamps. No gaps.

### 6. COHORT IS USER-OPTIONAL
`cohort` is a grouping key the user provides for their own bookkeeping.
Engines never filter or classify by cohort. The math is blind to it.

### 7. COMPUTE ONCE, QUERY FOREVER
Manifold runs once and writes parquet files. Every question gets answered
by Prime running SQL against those files.

### 8. NO HACKS
- No `sys.modules` aliases
- No shim classes or compatibility layers
- No `manifold/primitives/` directory
- No wrapper functions around pmtvs
- If imports are broken, fix them properly

---

## Where New Code Goes

| Type of Code | Location | Pattern |
|--------------|----------|---------|
| New math function | `pmtvs` (separate repo, PyPI) | Pure function, numpy in/out |
| New engine | `manifold/core/` | DataFrame in, DataFrame out, no I/O |
| New stage runner | `manifold/stages/<group>/` | Follow existing runner pattern |
| New I/O helper | `manifold/io/` | Reader or writer only |

---

## Sibling Repos

- **Prime** (`~/prime/`) — The interpreter. Reads Manifold's parquet files via DuckDB SQL. Handles typology, classification, analysis, canary detection, brittleness scoring, ML features. Static HTML + DuckDB-WASM explorer for browser-based analysis.
- **pmtvs** (`~/pmtvs/`, PyPI: `pmtvs`) — Rust+Python math functions. 244 functions: entropy, fractal, dynamical systems, spectral, embedding, topology, information theory, statistical tests. Shared by both Prime and Manifold.
- **Manifold** (this repo) — The compute engine. `observations.parquet` + `manifest.yaml` in, parquet files out.

---

## Known Domains

```
~/domains/
├── rossler/             # Chaotic system (Rössler attractor)
├── pendulum/            # Pendulum dynamics
├── cmapss/              # C-MAPSS turbofan (FD001–FD004)
├── hydraulic/           # Hydraulic condition monitoring
├── calce/               # Battery calendar aging
├── lorenz/              # Chaotic system (Lorenz attractor)
├── Bearings_IMS/        # IMS bearing degradation
├── building_vibration/  # Building vibration monitoring
└── lumo/                # Lumo dataset
```

---

## What NOT to Touch

- Mathematical logic inside pmtvs (separate repo)
- Output file schemas (downstream SQL depends on column names)
- The `main` branch (work on feature branches)
- Stage execution order in `run.py` (dependencies are real)
- observations.parquet (READ ONLY, always)
