# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## STOP: DATA COMPLIANCE IS ORTHON'S RESPONSIBILITY

**ORTHON delivers data. PRISM computes. The contract is non-negotiable.**

If `observations.parquet` does not comply with PRISM's schema:

1. **STOP** - Do not proceed
2. **REJECT** - Return error explaining what's wrong
3. **DO NOT** adjust PRISM to accommodate bad data

**Bad data from ORTHON is ORTHON's problem.**

---

## CRITICAL: PRISM - ORTHON Architecture

**PRISM computes numbers. ORTHON classifies.**

```
┌─────────────────┐   observations.parquet   ┌─────────────────┐
│     ORTHON      │ ────────────────────────>│      PRISM      │
│                 │   typology.parquet       │                 │
│   Classifies    │   manifest.yaml          │   Computes      │
│                 │ <────────────────────────│   Numbers only  │
└─────────────────┘   {parquet outputs}      └─────────────────┘
```

---

## Architecture

```
observations.parquet
        │
        ▼
┌─────────────────────────────────────────────┐
│            TYPOLOGY (ORTHON)                │
│  Signal classification (ORTHON's only calc) │
│  Output: typology.parquet, manifest.yaml    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        SIGNAL_VECTOR (PRISM)                │
│  Per-signal features at each I              │
│  Output: signal_vector.parquet              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         STATE_VECTOR (PRISM)                │
│  Centroid = mean position in feature space  │
│  NO eigenvalues here                        │
│  Output: state_vector.parquet               │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        STATE_GEOMETRY (PRISM)               │
│  SVD → eigenvalues, effective_dim           │
│  SHAPE of signal cloud lives HERE           │
│  Output: state_geometry.parquet             │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  SIGNAL_GEOMETRY + PAIRWISE (PRISM)         │
│  Output: signal_geometry.parquet            │
│          signal_pairwise.parquet            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│      GEOMETRY_DYNAMICS (PRISM)              │
│  Derivatives: velocity, acceleration, jerk  │
│  Output: geometry_dynamics.parquet          │
│          signal_dynamics.parquet            │
│          pairwise_dynamics.parquet          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         DYNAMICS LAYER (PRISM)              │
│  Output: lyapunov.parquet                   │
│          dynamics.parquet                   │
│          information_flow.parquet           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│          SQL LAYER (PRISM)                  │
│  Output: zscore.parquet                     │
│          statistics.parquet                 │
│          correlation.parquet                │
└─────────────────────────────────────────────┘
```

---

## Input: observations.parquet (Schema v2.0)

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| signal_id | str | What signal (temp, pressure, etc.) |
| I | UInt32 | Sequential index 0,1,2,3... per signal_id |
| value | Float64 | The measurement |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| unit_id | str | Pass-through label (cargo only) |

### I is Canonical

```
CORRECT:
signal_id | I | value
----------|---|------
temp      | 0 | 45.2
temp      | 1 | 45.4
temp      | 2 | 45.6

WRONG (timestamps):
signal_id | I          | value
----------|------------|------
temp      | 1596760568 | 45.2
```

### unit_id is Cargo

**unit_id has ZERO effect on compute.**

- DO NOT include unit_id in groupby operations
- Group by I or signal_id only
- unit_id passes through for ORTHON reporting

---

## Output Files (14 total)

### From ORTHON
- `typology.parquet` - Signal characterization
- `manifest.yaml` - Engine selection

### From PRISM
- `signal_vector.parquet` - Per-signal features
- `state_vector.parquet` - Centroids (position)
- `state_geometry.parquet` - Eigenvalues (shape)
- `signal_geometry.parquet` - Signal-to-centroid
- `signal_pairwise.parquet` - Pairwise relationships
- `geometry_dynamics.parquet` - State derivatives
- `signal_dynamics.parquet` - Signal derivatives
- `pairwise_dynamics.parquet` - Pairwise derivatives
- `lyapunov.parquet` - Lyapunov exponents
- `dynamics.parquet` - RQA, attractor
- `information_flow.parquet` - Transfer entropy
- `zscore.parquet` - Normalized
- `statistics.parquet` - Summary stats
- `correlation.parquet` - Correlation matrix

---

## Key Separation: state_vector vs state_geometry

| File | Computes | Contains |
|------|----------|----------|
| state_vector.py | Centroid (mean) | centroid_*, mean_distance |
| state_geometry.py | SVD on features | eigenvalue_*, effective_dim |

**Eigenvalues ONLY in state_geometry.py.**

---

## CLI Commands

```bash
# Run full pipeline
python -m prism data/cmapss

# Run individual stages
python -m prism signal-vector-temporal data/cmapss
python -m prism state-vector data/cmapss
python -m prism geometry data/cmapss
python -m prism geometry-dynamics data/cmapss
python -m prism lyapunov data/cmapss
python -m prism dynamics data/cmapss
python -m prism sql data/cmapss
```

---

## Rules

1. **PRISM computes, ORTHON classifies**
2. **Typology lives in ORTHON** (only calc there)
3. **state_vector = centroid, state_geometry = eigenvalues**
4. **Scale-invariant features only**
5. **I is canonical** (sequential, not timestamps)
6. **unit_id is cargo** (never in groupby)
7. **No classification in PRISM** (no labels, no thresholds)

---

## Do NOT

- Compute eigenvalues in state_vector.py
- Put classification logic in PRISM
- Create typology in PRISM
- Include unit_id in groupby
- Use scale-dependent engines (rms, peak, mean, std)
- Return labels from PRISM ("chaotic", "stable", etc.)

---

## Directory Structure

```
prism/
├── prism/
│   ├── cli.py
│   ├── signal_vector_temporal.py
│   ├── sql_runner.py
│   │
│   └── engines/
│       ├── state_vector.py       # Centroid only
│       ├── state_geometry.py     # Eigenvalues here
│       ├── signal_geometry.py
│       ├── signal_pairwise.py
│       ├── geometry_dynamics.py
│       ├── lyapunov_engine.py
│       ├── dynamics_runner.py
│       ├── information_flow_runner.py
│       └── sql/
│
└── data/
```

---

## Session Recovery

```bash
cd ~/prism
./venv/bin/python -m prism data/cmapss
```

---

## Credits

- **Avery Rudder** - "Laplace transform IS the state engine"
