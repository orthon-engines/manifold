# engines/vector/ — Vector Operation

Extracts per-signal (or per-entity) features from raw time series windows.

## What it does

The vector operation takes a signal window (a 1D numpy array of values) and
produces a feature vector — a flat dictionary of named numeric features that
describe the window's statistical shape, complexity, spectral content, and
harmonic structure.

## Module layout

```
engines/vector/
    __init__.py          # exports run, compute_vector
    run.py               # orchestrator + single-window compute
    engines/
        __init__.py      # (empty)
        shape.py         # kurtosis, skewness, crest_factor
        complexity.py    # sample_entropy, perm_entropy, hurst, acf_decay
        spectral.py      # spectral_slope, dominant_freq, spectral_entropy
        harmonic.py      # harmonics, fundamental_freq, thd
```

## Engine groups

| Group      | Wraps                                       | Key outputs                                     |
|------------|---------------------------------------------|-------------------------------------------------|
| shape      | `manifold.signal.statistics`                | kurtosis, skewness, crest_factor                |
| complexity | `manifold.signal.complexity`, `.memory`     | sample_entropy, hurst, acf_lag1, acf_half_life  |
| spectral   | `manifold.signal.spectral`                  | spectral_slope, dominant_freq, spectral_entropy |
| harmonic   | `manifold.signal.harmonics`, `.fundamental_freq`, `.thd` | fundamental_freq, thd_percent, harmonic_2x |

## Usage

### Single window (lightweight)

```python
import numpy as np
from engines.vector import compute_vector

y = np.random.randn(256)
features = compute_vector(y, engines=['shape', 'spectral'])
# {'kurtosis': -0.12, 'skewness': 0.03, 'crest_factor': 3.1, ...}
```

### Full pipeline step

```python
from engines.vector import run

df = run(
    observations_path='output/observations.parquet',
    output_path='output/signal_vector.parquet',
    manifest=manifest_dict,
    verbose=True,
)
```

The `run()` function delegates to `stage_01_signal_vector.run()`, which handles
windowing, parallel dispatch across signals, per-engine window sizing, and
parquet I/O. This module is a clean facade — no duplicated logic.

## Scale agnosticism

The vector operation does not know whether it is processing individual signals
(Scale 1) or cohort-level aggregates (Scale 2). The caller determines the
entity granularity; this module just computes features on whatever array it
receives.
