# engines/fingerprint/

Gaussian fingerprint operation module. Computes per-entity probabilistic
fingerprints and pairwise Bhattacharyya similarity for ANY set of entity
vectors over time.

**Scale-agnostic**: this module does not know whether entities are signals,
cohorts, or anything else. The same math applies at every scale.

## Layout

```
fingerprint/
  __init__.py         # exports run, run_system, compute_fingerprint, compute_similarity
  run.py              # orchestrator (delegates to stage_24 / stage_32, or runs generic)
  README.md           # this file
  engines/
    __init__.py       # (empty)
    gaussian.py       # per-entity Gaussian fingerprint (mean, std, volatility)
    similarity.py     # pairwise Bhattacharyya distance between fingerprints
```

## Engines

| Engine | Input | Output |
|--------|-------|--------|
| `gaussian` | DataFrame of entity vectors over time | mean\_\*, std\_\*, n\_windows, volatility per entity |
| `similarity` | Gaussian fingerprint DataFrame | bhattacharyya\_distance, n\_features, normalized\_distance, similarity, volatility\_diff per pair |

## Usage

```python
from engines.fingerprint import run, run_system, compute_fingerprint, compute_similarity
import polars as pl

# --- Signal-scale orchestration (reads/writes parquet via stage_24) ---
run("signal_vector.parquet", "gaussian_fingerprint.parquet", "gaussian_similarity.parquet")

# --- Cohort-scale orchestration (reads/writes parquet via stage_32) ---
run_system("cohort_vector.parquet", "cohort_fingerprint.parquet", "cohort_similarity.parquet")

# --- Direct compute on a DataFrame ---
data = pl.read_parquet("any_vector.parquet")
features = ['spectral_entropy', 'hurst', 'kurtosis']

fp = compute_fingerprint(data, features, entity_col='signal_id')
sim = compute_similarity(fp, features, entity_col='signal_id')

# --- Generic orchestration (any entity type, auto-detect features) ---
from engines.fingerprint.run import run_generic
fp, sim = run_generic(
    "custom_vector.parquet",
    "custom_fingerprint.parquet",
    "custom_similarity.parquet",
    entity_col='device_id',
)
```

## Math

### Gaussian Fingerprint

For each entity *e* and each feature *f*, given windows w_1 ... w_N:

- mean_f = (1/N) * sum(f(w_i))
- std_f  = sample standard deviation of f(w_i)
- volatility = (1/F) * sum(std_f) across all F features

### Bhattacharyya Distance

For two entities *a* and *b* with Gaussian fingerprints, the per-feature
Bhattacharyya distance is:

    D_B(f) = 0.25 * (mu_a - mu_b)^2 / (var_a + var_b)
           + 0.5 * ln((var_a + var_b) / (2 * sigma_a * sigma_b))

The total distance is summed across features. Similarity is exp(-D_B_total).

## Relationship to Existing Code

- `run()` delegates to `stage_24_gaussian_fingerprint` (signal scale, DuckDB SQL).
- `run_system()` delegates to `stage_32_cohort_fingerprint` (cohort scale, DuckDB SQL).
- `run_generic()` uses the pure-Polars/NumPy compute engines directly for
  arbitrary entity types without DuckDB.
- `gaussian.py` and `similarity.py` are standalone compute functions that
  operate on Polars DataFrames, independent of I/O or SQL.
