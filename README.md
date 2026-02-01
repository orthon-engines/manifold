# PRISM

**Persistent Relational Inference & Structural Measurement**

A behavioral geometry engine for signal topology analysis. PRISM transforms raw observations into eigenvalue-based state representations that capture the SHAPE of signal distributions.

---

## Quick Start

```bash
# Run full pipeline on a data directory
python -m prism data/cmapss

# Run individual stages
python -m prism typology data/cmapss
python -m prism signal-vector data/cmapss
python -m prism state-vector data/cmapss
python -m prism geometry data/cmapss
python -m prism dynamics data/cmapss

# Run temporal signal vector (for dynamics)
python -m prism signal-vector-temporal data/cmapss

# Run legacy 53-engine pipeline
python -m prism --legacy data/manifest.yaml
```

---

## Architecture (v2)

```
observations.parquet
       ↓
┌──────────────────────────────────────────────────────────────┐
│                    TYPOLOGY ENGINE                           │
│  Signal characterization: smoothness, periodicity, tails     │
│  Output: typology.parquet                                    │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                   SIGNAL VECTOR                              │
│  Per-signal features (scale-invariant only)                  │
│  Aggregate: signal_vector.parquet (one row per signal)       │
│  Temporal:  signal_vector_temporal.parquet (with I column)   │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                    STATE VECTOR                              │
│  System state via eigenvalues (SVD)                          │
│  - Eigenvalues capture SHAPE of signal cloud                 │
│  - effective_dim from participation ratio                    │
│  - Multi-mode detection                                      │
│  - Engine disagreement (shape vs spectral views)             │
│  Output: state_vector.parquet                                │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                  GEOMETRY LAYER                              │
│  state_geometry.parquet   - per-engine eigenvalues over time │
│  signal_geometry.parquet  - per-signal distance to state     │
│  signal_pairwise.parquet  - pairwise signal relationships    │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│               GEOMETRY DYNAMICS LAYER                        │
│  "You have position. You have shape. Here are derivatives."  │
│  geometry_dynamics.parquet - velocity, acceleration, jerk    │
│  signal_dynamics.parquet   - per-signal trajectory analysis  │
│  Collapse detection, trajectory classification               │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                  DYNAMICS LAYER                              │
│  dynamics.parquet        - Lyapunov, RQA, attractor          │
│  information_flow.parquet - Transfer entropy, Granger        │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                   SQL LAYER                                  │
│  zscore.parquet          - Normalized metrics                │
│  statistics.parquet      - Summary statistics                │
│  correlation.parquet     - Correlation matrix                │
│  regime_assignment.parquet - State labels                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Typology-Guided Engine Selection

Not all engines run on all signals. The typology engine classifies each signal:

| Classification | Engines Selected |
|----------------|------------------|
| SMOOTH | rolling_kurtosis, rolling_entropy, rolling_crest_factor |
| NOISY | kurtosis, entropy, crest_factor (larger window) |
| IMPULSIVE | kurtosis, crest_factor, peak_ratio |
| PERIODIC | harmonics_ratio, band_ratios, spectral_centroid |
| APERIODIC | entropy, hurst |
| NON_STATIONARY | rolling engines only (global stats meaningless) |

### Scale-Invariant Features Only

All features are scale-invariant (ratios, entropy, kurtosis). Deprecated:
- rms, peak, mean, std (absolute values)
- rolling_rms, rolling_mean, rolling_std, rolling_range
- envelope, total_power

### Eigenvalue-Based State

The state vector uses SVD to compute eigenvalues of the signal distribution:

```python
# Centroid (position in feature space)
centroid = mean(signal_matrix, axis=0)

# Eigenvalues (shape of signal cloud)
U, S, Vt = svd(signal_matrix - centroid)
eigenvalues = S² / (N - 1)

# Effective dimension (participation ratio)
effective_dim = (Σλ)² / Σλ²

# Multi-mode detection
is_multimode = (λ₂/λ₁ > 0.5) and (n_significant_modes >= 2)
```

**Key insight:** For two signals to occupy the same state, they must match across ALL feature dimensions. Eigenvalues capture this shape.

### Geometry Dynamics (Differential Geometry)

The geometry dynamics engine computes derivatives of the geometry evolution:

```python
# First derivative (velocity/tangent)
dx/dt = (x[t+1] - x[t-1]) / (2*dt)

# Second derivative (acceleration/curvature)
d²x/dt² = (x[t+1] - 2*x[t] + x[t-1]) / dt²

# Third derivative (jerk/torsion)
d³x/dt³ = derivative of acceleration

# Curvature
κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
```

**Trajectory Classification:**
| Type | Meaning |
|------|---------|
| STABLE | Low velocity and acceleration |
| CONVERGING | Moving toward equilibrium |
| DIVERGING | Moving away from equilibrium |
| OSCILLATING | Velocity changes sign periodically |
| CHAOTIC | High variability in derivatives |
| COLLAPSING | Sustained loss of effective dimension |
| EXPANDING | Sustained gain in effective dimension |

**Collapse Detection:** Identifies when effective_dim has sustained negative velocity - the system is losing degrees of freedom.

---

## Output Files

| File | Description | Rows |
|------|-------------|------|
| typology.parquet | Signal characterization | units × signals |
| signal_vector.parquet | Per-signal features | units × signals |
| signal_vector_temporal.parquet | Features with I column | units × signals × time |
| state_vector.parquet | System state | units |
| state_geometry.parquet | Per-engine eigenvalues over time | units × engines × time |
| signal_geometry.parquet | Signal-to-state relationships | units × signals |
| signal_pairwise.parquet | Pairwise relationships | units × N²/2 pairs |
| geometry_dynamics.parquet | Derivatives of geometry evolution | units × engines × time |
| signal_dynamics.parquet | Per-signal trajectory dynamics | units × signals × time |
| dynamics.parquet | Lyapunov, RQA, attractor | units × signals |
| information_flow.parquet | Transfer entropy, Granger | units × N² pairs |
| zscore.parquet | Normalized metrics | observations |
| statistics.parquet | Summary statistics | units × signals |
| correlation.parquet | Correlation matrix | units × signals² |
| regime_assignment.parquet | State labels | observations |

---

## Directory Structure

```
prism/
├── prism/
│   ├── cli.py                    # Main CLI
│   ├── signal_vector.py          # Aggregate signal features
│   ├── signal_vector_temporal.py # Temporal signal features
│   ├── sql_runner.py             # SQL engine runner
│   │
│   ├── engines/
│   │   ├── engine_manifest.yaml  # Scale-invariant engine config
│   │   ├── typology_engine.py    # Signal characterization
│   │   ├── state_vector.py       # Eigenvalue-based state
│   │   ├── state_geometry.py     # Per-engine eigenvalues
│   │   ├── signal_geometry.py    # Signal-to-state relationships
│   │   ├── signal_pairwise.py    # Pairwise relationships
│   │   ├── geometry_dynamics.py  # Differential geometry (derivatives)
│   │   ├── dynamics_runner.py    # Lyapunov, RQA
│   │   ├── information_flow_runner.py  # Transfer entropy, Granger
│   │   ├── signal/               # Individual signal engines
│   │   ├── rolling/              # Rolling window engines
│   │   └── sql/                  # SQL engines
│   │
│   └── _legacy/                  # Legacy 53-engine pipeline
│       ├── runner.py
│       ├── python_runner.py
│       └── ram_manager.py
│
├── data/
│   └── cmapss/                   # C-MAPSS turbofan dataset
│       ├── observations.parquet
│       ├── typology.parquet
│       ├── signal_vector.parquet
│       └── state_vector.parquet
│
└── ENGINE_INVENTORY.md           # Full engine inventory
```

---

## Credits

- **Avery Rudder** - "Laplace transform IS the state engine" - eigenvalue insight
- Architecture: Typology-guided, scale-invariant, eigenvalue-based

---

## License

MIT
