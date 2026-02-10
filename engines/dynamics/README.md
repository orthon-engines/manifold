# engines/dynamics/

Dynamical systems operation module. Computes FTLE, velocity fields,
break detection, and ridge proximity on ANY trajectory.

**Scale-agnostic**: this module does not know whether the trajectory comes
from a single signal time series, a cohort centroid path, or any other source.
The same math applies at every scale.

## Layout

```
dynamics/
  __init__.py             # exports run_*, compute_ftle, compute_velocity
  run.py                  # orchestrator (delegates to stage_00/08/21/22/23)
  README.md               # this file
  engines/
    __init__.py           # (empty)
    ftle.py               # FTLE via Rosenstein/Kantz (forward + backward)
    ftle_rolling.py       # rolling FTLE with trend statistics
    velocity.py           # speed, acceleration, curvature from trajectory
    ridge.py              # urgency = velocity toward FTLE ridge
    breaks.py             # Heaviside + Dirac break detection
```

## Engines

| Engine | Input | Output |
|--------|-------|--------|
| `ftle` | 1-D signal array | ftle, ftle_std, embedding_dim, embedding_tau, confidence, is_deterministic |
| `ftle_rolling` | 1-D signal array | rolling_ftle, rolling_ftle_std, rolling_ftle_confidence |
| `velocity` | (T, D) trajectory matrix | speed, acceleration_magnitude, curvature |
| `ridge` | ftle_values + speeds arrays | ftle_gradient, urgency, time_to_ridge |
| `breaks` | 1-D signal array | list of break dicts (I, magnitude, direction, sharpness, ...) |

## Usage

```python
from engines.dynamics import (
    run_ftle, run_ftle_rolling, run_velocity, run_ridge, run_breaks,
    compute_ftle, compute_velocity,
)
import numpy as np

# --- Compute engines (arrays in, dicts out) ---
y = np.random.randn(1000)
result = compute_ftle(y, direction='forward', min_samples=200)
print(result['ftle'], result['confidence'])

trajectory = np.random.randn(500, 3)  # 500 timesteps, 3 dimensions
vel = compute_velocity(trajectory)
print(vel['speed'].shape, vel['curvature'].shape)

# --- Pipeline orchestration (reads/writes parquet) ---
run_breaks("observations.parquet", "breaks.parquet", sensitivity=1.0)
run_ftle("observations.parquet", "ftle.parquet", direction='forward')
run_ftle_rolling("observations.parquet", "ftle_rolling.parquet", window_size=200)
run_velocity("observations.parquet", "velocity_field.parquet")
run_ridge("ftle_rolling.parquet", "velocity_field.parquet", "ridge_proximity.parquet")
```

## Relationship to Existing Code

- `ftle.py` wraps `engines.manifold.dynamics.ftle.compute()` -- the single
  source of truth for FTLE computation via Rosenstein/Kantz.
- `ftle_rolling.py` wraps `engines.manifold.dynamics.ftle.compute_rolling()` and
  `compute_trend()` for rolling window analysis.
- `velocity.py` implements the same math as `stage_21_velocity_field` but at the
  array level, without parquet I/O or per-cohort iteration.
- `ridge.py` implements the core urgency math from `stage_23_ridge_proximity`
  at the array level, without parquet I/O.
- `breaks.py` wraps `engines.manifold.breaks.compute()` and `summarize_breaks()`.
- `run.py` delegates to `stage_00`, `stage_08`, `stage_21`, `stage_22`, and
  `stage_23` for full pipeline-scale orchestration.
