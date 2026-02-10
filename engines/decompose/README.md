# engines/decompose/

Eigendecomposition operation module. Decomposes ANY feature matrix
(N entities x M features) into eigenvalues, eigenvectors, and loadings.

**Scale-agnostic**: this module does not know whether entities are signals,
cohorts, or anything else. The same math applies at every scale.

## Layout

```
decompose/
  __init__.py         # exports run, run_system, compute_decomposition
  run.py              # orchestrator (delegates to stage_03 / stage_26)
  README.md           # this file
  engines/
    __init__.py       # (empty)
    eigen.py          # full eigendecomposition via SVD
    effective_dim.py  # participation ratio + eigenvalue entropy
    condition.py      # condition number, spectral gap, eigenvalue ratios
    thermodynamics.py # temperature (dS/dI), free energy, heat capacity
```

## Engines

| Engine | Input | Output |
|--------|-------|--------|
| `eigen` | (N, D) feature matrix | eigenvalues, explained_ratio, effective_dim, condition_number, principal_components, signal_loadings |
| `effective_dim` | 1-D eigenvalue array | effective_dim, eigenvalue_entropy, eigenvalue_entropy_norm |
| `condition` | 1-D eigenvalue array | condition_number, ratio_2_1, spectral_gap, ratio_3_1 |
| `thermodynamics` | time-series of effective_dim + total_variance | temperature, free_energy, heat_capacity |

## Usage

```python
from engines.decompose import run, run_system, compute_decomposition
import numpy as np

# --- Full eigendecomposition of a raw matrix ---
matrix = np.random.randn(10, 5)  # 10 entities, 5 features
result = compute_decomposition(matrix, norm_method="zscore")
print(result['effective_dim'])
print(result['eigenvalues'])

# --- Signal-scale orchestration (reads/writes parquet) ---
run("signal_vector.parquet", "state_vector.parquet", "state_geometry.parquet")

# --- Cohort-scale orchestration (reads/writes parquet) ---
run_system("cohort_vector.parquet", "system_geometry.parquet")
```

## Relationship to Existing Code

- `eigen.py` wraps `engines.manifold.state.eigendecomp.compute()` -- the single
  source of truth for SVD-based eigendecomposition.
- `run.py` delegates to `stage_03_state_geometry` (signal scale) and
  `stage_26_system_geometry` (cohort scale).
- `effective_dim.py`, `condition.py`, and `thermodynamics.py` are standalone
  compute functions that operate on eigenvalue arrays, independent of I/O.
