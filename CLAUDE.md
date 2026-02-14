# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## ⛔ STOP: MANDATORY RULES — READ BEFORE EVERY ACTION

### Rule 0: SEARCH BEFORE YOU CREATE

**Before writing ANY new code, you MUST search the repo for existing implementations.**

```bash
# Find existing files
find . -name "*.py" | xargs grep -l "function_name"

# Find existing patterns
grep -r "def compute" manifold/core/

# Find how similar things are done
grep -r "sample_rate" manifold/
```

**If you think something doesn't exist, ASK THE USER before creating it.**

### Rule 1: USE EXISTING CODE

If a function, engine, or pattern exists in the repo, **USE IT**. Do not recreate.

```
WRONG: "I'll write a quick FFT function..."
RIGHT: "I see manifold/primitives/individual/spectral.py has psd() — using that."

WRONG: "Let me create a runner to orchestrate this..."
RIGHT: "I see manifold/stages/vector/ handles this — adding to it."
```

### Rule 2: NO ROGUE FILE CREATION

| Location | Allowed? |
|----------|----------|
| `/tmp/` | ❌ NEVER |
| `~/` | ❌ NEVER |
| Random standalone scripts | ❌ NEVER |
| Inside existing repo structure | ✅ With approval |

**If you create something in /tmp, you are hiding evidence. This is forbidden.**

### Rule 3: SHOW YOUR WORK BEFORE CHANGES

Before modifying any file, show:
1. The existing file/function you're modifying
2. The existing pattern you're following
3. Get explicit approval before creating NEW files

```
WRONG: *silently creates new_runner.py*
RIGHT: "I found breaks.py at manifold/stages/vector/breaks.py.
        I'll add the new function there following the existing pattern.
        Here's the current structure: [shows code]
        Okay to proceed?"
```

### Rule 4: MANIFOLD COMPUTES, ORTHON CLASSIFIES

- Do NOT create typology logic, classification rules, or signal labels in MANIFOLD
- Do NOT modify observations.parquet or typology.parquet
- Do NOT second-guess manifest.yaml — execute what it says
- If MANIFEST_CONTRACT.md doesn't answer your question, ASK THE USER

**If you find yourself writing `if signal_type == 'PERIODIC'` in MANIFOLD, STOP.
That is classification. Classification belongs in ORTHON.**

---

## 🚫 EXPLICITLY FORBIDDEN BEHAVIORS

| Behavior | Why It's Forbidden |
|----------|-------------------|
| Create scripts in /tmp | Hides work, not verifiable |
| Create "one-off" runners | Bypasses established patterns |
| Inline compute in stages | Core computes, stages orchestrate |
| Duplicate existing functionality | Creates inconsistency |
| Create new venv | Use existing `./venv/` |
| Guess at implementations | Ask if unsure |
| Generate code without showing existing | Must show what exists first |

### The /tmp Rule (CRITICAL)

```
/tmp is where code goes to avoid accountability.

You will NEVER:
- Write scripts to /tmp
- Write data to /tmp
- Write anything to /tmp

EVERYTHING stays in the repository where it can be reviewed.
```

### The One-Off Runner Rule (CRITICAL)

```
WRONG:
  "I'll create a quick script to do this..."
  "Let me write a standalone runner..."
  "Here's a temporary solution..."

RIGHT:
  "I found the existing stage at [path]. Adding to it."
  "This matches the pattern in [existing file]. Following that."
  "The manifest contract says [X]. Implementing exactly that."
```

---

## ✅ ALLOWED BEHAVIORS

| Behavior | How To Do It |
|----------|--------------|
| Call existing engines | `engine_registry[name](window)` |
| Sequence operations | Chain existing stages via `manifold/run.py` |
| Pass config from manifest | Read and forward, don't interpret |
| Add to existing files | Show the file first, get approval |
| Create new engine | Follow existing engine pattern, propose location first |

---

## Architecture

Manifold is a domain-agnostic dynamical systems computation engine.
- **orthon** — dynamical systems analysis interpreter (brain)
- **manifold** — dynamical systems computation (muscle)

### Four Layers

```
manifold/primitives/   Pure math (numpy → float). No I/O, no DataFrames.
manifold/core/         Compute engines (DataFrames → DataFrames). signal/, state/, dynamics/, pairwise/.
manifold/stages/       Stage runners (orchestration only). 5 groups, 27 stages.
manifold/io/           Parquet reader, writer, manifest loader.
```

### Five Stage Groups

| Group | Directory | Purpose | Output Dirs |
|-------|-----------|---------|-------------|
| vector | `manifold/stages/vector/` | Signal features | 1_signal_features |
| geometry | `manifold/stages/geometry/` | System state | 2_system_state, 3_health_scoring |
| dynamics | `manifold/stages/dynamics/` | Trajectory evolution | 5_evolution |
| information | `manifold/stages/information/` | Pairwise relationships | 4_signal_relationships |
| energy | `manifold/stages/energy/` | Fleet analysis | 6_fleet |

### Pipeline (27 stages)

All stages always run. No opt-in, no tiers.

```
observations.parquet + manifest.yaml
        │
        ▼
  vector:
    00 breaks              01 signal_vector       33 signal_stability

  geometry:
    02 state_vector        03 state_geometry       05 signal_geometry
    07 geometry_dynamics   20 sensor_eigendecomp   34 cohort_baseline
    35 observation_geometry

  dynamics:
    08 ftle                08_lyapunov             09a cohort_thermodynamics
    15 ftle_field          17 ftle_backward        21 velocity_field
    22 ftle_rolling        23 ridge_proximity

  information:
    06 signal_pairwise     10 information_flow     18 segment_comparison
    19 info_flow_delta

  energy (requires n_cohorts >= 2):
    26 system_geometry     27 cohort_pairwise      28 cohort_information_flow
    30 cohort_ftle         31 cohort_velocity_field
```

### Output: 6 Named Directories

```
output/
├── 1_signal_features/        signal_vector, signal_geometry, signal_stability
├── 2_system_state/           state_vector, state_geometry, geometry_dynamics, sensor_eigendecomp
├── 3_health_scoring/         breaks, cohort_baseline, observation_geometry
├── 4_signal_relationships/   signal_pairwise, information_flow, segment_comparison, info_flow_delta
├── 5_evolution/              ftle, lyapunov, ftle_field, ftle_backward, velocity_field, ftle_rolling, ridge_proximity
└── 6_fleet/                  system_geometry, cohort_pairwise, cohort_information_flow, cohort_ftle, cohort_velocity_field
```

### Running

```bash
python -m manifold data/rossler               # Full pipeline
python -m manifold data/rossler --stages 01,02,03   # Subset
python -m manifold data/rossler --skip 08,09a       # Skip specific
```

---

## Engine Minimum Sample Requirements

**FFT-based engines require larger windows. This is physics, not a bug.**

| Engine | Minimum Samples | Reason |
|--------|-----------------|--------|
| spectral | 64 | FFT resolution |
| harmonics | 64 | FFT resolution |
| fundamental_freq | 64 | FFT resolution |
| thd | 64 | FFT resolution |
| sample_entropy | 64 | Statistical validity |
| hurst | 128 | Long-range dependence |
| crest_factor | 4 | Simple ratio |
| kurtosis | 4 | 4th moment |
| skewness | 4 | 3rd moment |
| perm_entropy | 8 | Permutation patterns |
| acf_decay | 16 | Lag structure |
| snr | 32 | Power estimation |
| phase_coherence | 32 | Phase estimation |

**When system window < engine minimum:**
- Manifest specifies `engine_window_overrides`
- Manifold uses expanded window for that engine
- I (window end index) alignment is preserved

**Do NOT lower engine minimums to fit small windows. The math doesn't work.**

---

## Directory Structure

```
~/manifold/
├── CLAUDE.md                     # This file
├── README.md                     # Project docs
├── LICENSE.md                    # PolyForm Noncommercial 1.0.0
├── pyproject.toml                # Package config
├── requirements.txt              # Dev dependencies
├── config/                       # Domain & environment YAML configs
├── tests/                        # Test suite
├── docs/                         # Documentation
│
├── manifold/                     # Main Python package
│   ├── __init__.py               # Package docstring
│   ├── __main__.py               # python -m manifold entry point
│   ├── run.py                    # Pipeline sequencer (27 stages)
│   │
│   ├── io/                       # I/O layer (all parquet reads/writes)
│   │   ├── reader.py             # STAGE_DIRS mapping, load_observations
│   │   ├── writer.py             # write_output, write_sidecar
│   │   └── manifest.py           # load_manifest, get_observations_path
│   │
│   ├── stages/                   # Stage runners (orchestration, no math)
│   │   ├── vector/               # breaks, signal_vector, signal_stability
│   │   ├── geometry/             # state_vector, state_geometry, signal_geometry,
│   │   │                         #   geometry_dynamics, sensor_eigendecomp,
│   │   │                         #   cohort_baseline, observation_geometry
│   │   ├── dynamics/             # ftle, lyapunov, cohort_thermodynamics,
│   │   │                         #   ftle_field, ftle_backward, velocity_field,
│   │   │                         #   ftle_rolling, ridge_proximity
│   │   ├── information/          # signal_pairwise, information_flow,
│   │   │                         #   segment_comparison, info_flow_delta
│   │   └── energy/               # system_geometry, cohort_pairwise,
│   │                             #   cohort_information_flow, cohort_ftle,
│   │                             #   cohort_velocity_field
│   │
│   ├── core/                     # Compute engines (DataFrames → DataFrames)
│   │   ├── signal/               # 38 per-signal engines (spectral, hurst, entropy, etc.)
│   │   ├── state/                # Centroid, eigendecomposition (SVD)
│   │   ├── dynamics/             # FTLE, Lyapunov, attractor, saddle detection
│   │   ├── geometry/             # Feature group config
│   │   ├── pairwise/             # Correlation, Granger causality, copula
│   │   ├── parallel/             # Parallel runners (dynamics, info flow, topology)
│   │   ├── sql/                  # SQL engines (zscore, statistics, correlation)
│   │   ├── signal_geometry.py    # Per-signal → state relationships
│   │   ├── state_geometry.py     # Eigenvalues and shape metrics
│   │   ├── geometry_dynamics.py  # Derivatives, curvature, collapse detection
│   │   ├── signal_pairwise.py   # Signal-to-signal pairwise
│   │   ├── registry.py           # Engine discovery & loading
│   │   ├── base.py               # BaseEngine class
│   │   ├── rolling.py            # Rolling window wrapper
│   │   └── normalization.py      # Z-score, robust, MAD, min-max
│   │
│   ├── primitives/               # Pure math (numpy → float, no I/O)
│   │   ├── individual/           # Statistics, spectral, entropy, derivatives, etc.
│   │   ├── embedding/            # Delay embedding, Cao's method, AMI
│   │   ├── dynamical/            # FTLE, Lyapunov, RQA, saddle
│   │   ├── pairwise/             # Correlation, causality, distance
│   │   ├── matrix/               # SVD, covariance, graph Laplacian
│   │   ├── information/          # Transfer entropy, mutual info, divergence
│   │   ├── network/              # Centrality, community, paths
│   │   ├── topology/             # Persistent homology, distance
│   │   └── tests/                # Bootstrap, hypothesis, null models
│   │
│   └── validation/               # Input validation & prerequisites
│
└── data/                         # Domain datasets (rossler, etc.)
```

### Where New Code Goes

| Type of Code | Location | Pattern to Follow |
|--------------|----------|-------------------|
| New signal engine | `manifold/core/signal/` | Copy `kurtosis.py` pattern |
| New rolling engine | `manifold/core/rolling.py` | Add to existing rolling wrapper |
| New primitive | `manifold/primitives/individual/` | Pure function, no I/O |
| New stage | ASK FIRST | Probably doesn't need new stage |
| New pairwise primitive | `manifold/primitives/pairwise/` | Pure function, no I/O |

### Import Conventions

```python
# Primitives: direct function imports
from manifold.primitives.individual.spectral import psd, spectral_entropy
from manifold.primitives.individual.geometry import eigendecomposition

# Core engines: module or function imports
from manifold.core.signal_geometry import compute_signal_geometry
from manifold.core import signal, state

# Stages import from core, never from other stages
from manifold.core.geometry_dynamics import compute_geometry_dynamics
```

---

## Input: manifest.yaml (from ORTHON)

**Read MANIFEST_CONTRACT.md for the full specification.**

Quick reference — for each signal, the manifest tells Manifold:
- `engines`: which signal-level engines to run
- `rolling_engines`: which rolling engines to run
- `window_size`: samples per window (system default)
- `stride`: samples between windows
- `engine_window_overrides`: per-engine window sizes (when different from system)
- `derivative_depth`: max derivative order (0, 1, or 2)
- `eigenvalue_budget`: max eigenvalues to compute
- `output_hints`: how to format engine output (per_bin vs summary, etc.)

Manifold executes what the manifest says. No more, no less.

---

## Input: observations.parquet Schema (v2.4)

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| signal_id | str | What signal (temp, pressure, etc.) |
| I | UInt32 | Sequential index 0,1,2,3... per signal_id |
| value | Float64 | The measurement |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| cohort | str | Grouping key (engine_1, pump_A) - cargo only |

### I is Canonical
- Sequential integers per signal_id
- NOT timestamps
- Starts at 0, no gaps

### cohort is Cargo
- ZERO effect on compute
- Never in groupby
- Passes through for reporting

---

## Key Rules Summary

1. **SEARCH BEFORE CREATE** — find existing code first
2. **USE EXISTING PATTERNS** — don't reinvent
3. **NO /tmp** — everything in repo
4. **NO ONE-OFF RUNNERS** — use established stages
5. **MANIFOLD computes, ORTHON classifies** — no labels in MANIFOLD
6. **state_vector = centroid, state_geometry = eigenvalues** — separate concerns
7. **Scale-invariant features only** — no absolute values
8. **I is canonical** — sequential per signal_id
9. **ASK IF UNSURE** — don't guess

---

## Checklist Before Any Change

```
□ Did I search for existing implementations?
□ Am I using existing patterns/files?
□ Did I show the user what I'm modifying?
□ Is this going in the repo (not /tmp)?
□ Am I following MANIFEST_CONTRACT.md?
□ Did I get approval for new files?
```

**If any answer is NO, stop and fix it.**

---

## Error Handling

**Engines must fail loudly, not silently.**

```python
# WRONG - silent failure
def compute(y):
    try:
        result = complex_math(y)
    except:
        pass  # Silent! BAD!
    return result

# RIGHT - loud failure
def compute(y):
    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")
    return complex_math(y)
```

If an engine can't run (insufficient samples, bad data), it should:
1. Raise an exception, OR
2. Return explicit NaN with logged warning

Never silently return garbage.

---

## Credits

- **Avery Rudder** — "Laplace transform IS the state engine" — eigenvalue insight
