# engines/pairwise/

Pairwise operation module. Computes relationship metrics between ANY two vectors.

## Architecture

This module is a **thin delegation layer** over existing engines:

```
engines/pairwise/
    __init__.py           exports run, compute_pairwise
    run.py                orchestrator (compute_pairwise for vectors, run for pipeline)
    engines/
        __init__.py
        distance.py       -> engines.primitives.pairwise.distance
        correlation.py    -> engines.manifold.pairwise.correlation
        information.py    -> engines.manifold.pairwise.causality + primitives.information
        cointegration.py  -> engines.manifold.pairwise.cointegration
        copula.py         -> engines.manifold.pairwise.copula
        topology.py       -> engines.entry_points.stage_11_topology
```

## Usage

### Direct vector-to-vector (any scale)

```python
from engines.pairwise import compute_pairwise

results = compute_pairwise(x, y)
# returns flat dict with all metrics
```

### Full pipeline (signal-level, parquet I/O)

```python
from engines.pairwise import run

df = run("signal_vector.parquet", "state_vector.parquet", "signal_pairwise.parquet")
```

### Individual engines

```python
from engines.pairwise.engines.distance import compute
from engines.pairwise.engines.correlation import compute
from engines.pairwise.engines.information import compute_granger
```

## Engines

| Engine | Metrics | Delegate |
|--------|---------|----------|
| distance | euclidean, manhattan, cosine, DTW | primitives.pairwise.distance |
| correlation | pearson, spearman, xcorr, MI | manifold.pairwise.correlation |
| information | granger, TE, KL, JS | manifold.pairwise.causality + primitives.information |
| cointegration | ADF, hedge ratio, half-life | manifold.pairwise.cointegration |
| copula | gaussian, clayton, gumbel, frank | manifold.pairwise.copula |
| topology | density, degree, edges | entry_points.stage_11_topology |

## Scale Agnosticism

`compute_pairwise(x, y)` does not know whether x and y are:
- Raw signal windows (signal-level pairwise)
- Cohort centroid vectors (cohort-level pairwise)
- System-level summary vectors

The math is identical at every scale. The caller decides the semantics.
