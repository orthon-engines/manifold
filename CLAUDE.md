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
grep -r "def compute" prism/engines/

# Find how similar things are done
grep -r "sample_rate" prism/
```

**If you think something doesn't exist, ASK THE USER before creating it.**

### Rule 1: USE EXISTING CODE

If a function, engine, or pattern exists in the repo, **USE IT**. Do not recreate.

```
WRONG: "I'll write a quick FFT function..."
RIGHT: "I see prism/primitives/spectral.py has psd() — using that."

WRONG: "Let me create a runner to orchestrate this..."
RIGHT: "I see prism/signal_vector/runner.py handles this — adding to it."
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
RIGHT: "I found runner.py at prism/signal_vector/runner.py.
        I'll add the new function there following the existing pattern.
        Here's the current structure: [shows code]
        Okay to proceed?"
```

### Rule 4: PRISM COMPUTES, ORTHON CLASSIFIES

- Do NOT create typology logic, classification rules, or signal labels in PRISM
- Do NOT modify observations.parquet or typology.parquet
- Do NOT second-guess manifest.yaml — execute what it says
- If MANIFEST_CONTRACT.md doesn't answer your question, ASK THE USER

**If you find yourself writing `if signal_type == 'PERIODIC'` in PRISM, STOP.
That is classification. Classification belongs in ORTHON.**

---

## 🚫 EXPLICITLY FORBIDDEN BEHAVIORS

| Behavior | Why It's Forbidden |
|----------|-------------------|
| Create scripts in /tmp | Hides work, not verifiable |
| Create "one-off" runners | Bypasses established patterns |
| Inline compute in orchestrators | Engines compute, orchestrators orchestrate |
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
  "I found the existing runner at [path]. Adding to it."
  "This matches the pattern in [existing file]. Following that."
  "The manifest contract says [X]. Implementing exactly that."
```

---

## ✅ ALLOWED BEHAVIORS

| Behavior | How To Do It |
|----------|--------------|
| Call existing engines | `engine_registry[name](window)` |
| Sequence operations | Chain existing entry points |
| Pass config from manifest | Read and forward, don't interpret |
| Add to existing files | Show the file first, get approval |
| Create new engine | Follow existing engine pattern, propose location first |

---

## Architecture

```
observations.parquet   (from ORTHON)
typology.parquet       (from ORTHON)
manifest.yaml          (from ORTHON)
        │
        ▼
┌─────────────────────────────────────────────┐
│        signal_vector/  (PRISM Stage 1)      │
│  Per-signal features at each I              │
│  signal/ = kurtosis, spectral, entropy...   │
│  rolling/ = rolling_kurtosis, rolling_vol...│
│  sql/ = zscore, statistics                  │
│  Output: signal_vector.parquet              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        state_vector/  (PRISM Stage 2)       │
│  state_vector.py = centroid (WHERE)         │
│  state_geometry.py = eigenvalues (SHAPE)    │
│  signal_geometry.py = signal-to-centroid    │
│  Output: state_vector.parquet               │
│          state_geometry.parquet             │
│          signal_geometry.parquet            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│    geometry_pairwise/  (PRISM Stage 3)      │
│  signal_pairwise.py = between-signal        │
│  granger, transfer_entropy = directional    │
│  correlation, mutual_info = symmetric       │
│  Output: signal_pairwise.parquet            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│    geometry_laplace/  (PRISM Stage 4)       │
│  geometry_dynamics.py = velocity, accel     │
│  lyapunov_engine.py = chaos measurement     │
│  dynamics_runner.py = RQA, attractor        │
│  information_flow_runner.py = TE, causal    │
│  Output: geometry_dynamics.parquet          │
│          lyapunov.parquet                   │
│          dynamics.parquet                   │
│          information_flow.parquet           │
└─────────────────────────────────────────────┘
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
- PRISM uses expanded window for that engine
- I (window end index) alignment is preserved

**Do NOT lower engine minimums to fit small windows. The math doesn't work.**

---

## Directory Structure

```
prism/
├── primitives/                # CANONICAL MATH FUNCTIONS
│   └── individual/
│       └── spectral.py        # psd, dominant_frequency, etc.
│
├── engines/                   # COMPUTE ENGINES (call primitives)
│   ├── signal/                # Per-signal engines
│   │   ├── kurtosis.py
│   │   ├── spectral.py        # Imports from primitives/
│   │   ├── harmonics.py
│   │   └── ...
│   ├── rolling/               # Rolling window engines
│   └── sql/                   # SQL-based engines
│
├── signal_vector/             # Stage 1 orchestration
│   └── runner.py              # USE THIS — don't create new runners
│
├── state_vector/              # Stage 2
├── geometry_pairwise/         # Stage 3
├── geometry_laplace/          # Stage 4
│
├── cli.py                     # CLI entry point
├── MANIFEST_CONTRACT.md       # What ORTHON delivers (READ THIS)
└── CLAUDE.md                  # YOU ARE HERE
```

### Where New Code Goes

| Type of Code | Location | Pattern to Follow |
|--------------|----------|-------------------|
| New signal engine | `prism/engines/signal/` | Copy `kurtosis.py` pattern |
| New rolling engine | `prism/engines/rolling/` | Copy existing rolling pattern |
| New primitive | `prism/primitives/individual/` | Pure function, no I/O |
| New stage | ASK FIRST | Probably doesn't need new stage |

---

## Input: manifest.yaml (from ORTHON)

**Read MANIFEST_CONTRACT.md for the full specification.**

Quick reference — for each signal, the manifest tells PRISM:
- `engines`: which signal-level engines to run
- `rolling_engines`: which rolling engines to run
- `window_size`: samples per window (system default)
- `stride`: samples between windows
- `engine_window_overrides`: per-engine window sizes (when different from system)
- `derivative_depth`: max derivative order (0, 1, or 2)
- `eigenvalue_budget`: max eigenvalues to compute
- `output_hints`: how to format engine output (per_bin vs summary, etc.)

PRISM executes what the manifest says. No more, no less.

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
4. **NO ONE-OFF RUNNERS** — use established orchestrators
5. **PRISM computes, ORTHON classifies** — no labels in PRISM
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
