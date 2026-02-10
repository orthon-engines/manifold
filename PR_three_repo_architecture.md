# PR: Three-Repo Architecture — orthon-engines → orthon → orthon-ml

## Summary

Establish the canonical repository structure for the ORTHON ecosystem.
Three repos, one dependency direction, zero duplication.

## The Problem

Currently all code lives in one repo. As the project splits into core
framework, compute engines, and ML integration, we need a structure where:

- Engines never get duplicated across repos
- Updates to an engine propagate everywhere automatically
- Each repo has a clear audience and clear boundaries
- `pip install` just works at every level

## Architecture

```
orthon-engines          (foundation — pure math, zero dependencies)
       ↑
    orthon              (framework — orchestration, SQL, pipelines)
       ↑
    orthon-ml           (shell — ML-friendly API, notebooks, demos)
```

**Single dependency direction. No cycles. No diamonds.**

orthon-ml depends on orthon, which depends on orthon-engines.
Installing orthon-ml pulls in everything. Installing orthon pulls in engines.
Installing orthon-engines gives you just the math.

Both orthon and orthon-ml import engines from the same installed package.
No file copying. No version management headaches. One `pip install` resolves
the entire tree.

---

## Repo 1: `orthon-engines`

**Audience:** Mathematicians, researchers who want raw compute  
**Install:** `pip install orthon-engines`  
**Dependencies:** numpy (that's it, except copula needs scipy.special.ndtri)

### What's in it

```
orthon-engines/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── orthon_engines/
│       ├── __init__.py          # exports all compute() functions
│       ├── hurst.py
│       ├── acf.py
│       ├── permutation_entropy.py
│       ├── sample_entropy.py
│       ├── fft.py
│       ├── spectral_entropy.py
│       ├── garch.py
│       ├── rolling_variance.py
│       ├── wavelet_decomposition.py
│       ├── derivative_stats.py
│       ├── rqa.py
│       ├── cusum.py
│       ├── level_shift.py
│       ├── runs_test.py
│       ├── phase_space.py
│       ├── embedding.py
│       ├── correlation_dimension.py
│       ├── density.py
│       ├── mode_detection.py
│       ├── range_analysis.py
│       ├── extreme_clustering.py
│       ├── path_complexity.py
│       ├── directional_bias.py
│       ├── rolling_correlation.py
│       ├── dynamic_coherence.py
│       ├── phase_sync.py
│       ├── sync_index.py
│       ├── multivariate_changepoint.py
│       ├── lyapunov.py
│       ├── basin_stability.py
│       ├── cross_recurrence.py
│       ├── joint_recurrence.py
│       ├── transfer_entropy.py
│       ├── directed_info.py
│       ├── granger.py
│       ├── ccm.py
│       ├── impulse_response.py
│       ├── local_projections.py
│       ├── pcmci.py
│       ├── var_decomposition.py
│       ├── cross_correlation.py
│       ├── wavelet_coherence.py
│       ├── cointegration.py
│       ├── copula.py
│       ├── lasso_select.py
│       └── mutual_info.py       # if split from lasso_select
```

### Rules

- Every file has a `compute()` function
- Arrays in, dict of numbers out
- No framework imports, no SQL, no orchestration
- No domain vocabulary (no "stock", "patient", "turbofan")
- Engines don't know each other exist (no cross-imports between engines)

### pyproject.toml

```toml
[project]
name = "orthon-engines"
version = "0.1.0"
description = "Pure mathematical compute engines for signal analysis"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21",
]

[project.optional-dependencies]
full = ["scipy>=1.7"]   # for copula's ndtri
```

### Usage (standalone)

```python
from orthon_engines import hurst, fft, lasso_select

signal = np.random.randn(1000)
h = hurst.compute(signal)
print(h['hurst'])  # 0.50 ± noise (random walk)

spectrum = fft.compute(signal)
print(spectrum['dominant_frequency'])
```

---

## Repo 2: `orthon`

**Audience:** Signal analysis researchers, data scientists  
**Install:** `pip install orthon`  
**Dependencies:** orthon-engines, duckdb

### What's in it

```
orthon/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── orthon/
│       ├── __init__.py
│       ├── typology/               # Signal Typology layer
│       │   ├── __init__.py
│       │   ├── classifier.py       # signal classification logic
│       │   └── schema.sql          # typology output tables
│       ├── geometry/               # Behavioral Geometry layer
│       │   ├── __init__.py
│       │   ├── orchestrator.py     # pure orchestrator, no inline compute
│       │   └── schema.sql
│       ├── dynamics/               # Dynamical Systems layer
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   └── schema.sql
│       ├── causal/                 # Causal Mechanics layer
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   └── schema.sql
│       ├── registry/               # engine registry + config
│       │   ├── __init__.py
│       │   ├── engines.yaml        # maps engines to layers
│       │   └── loader.py           # discovers + loads engines from orthon-engines
│       ├── pipeline/               # end-to-end orchestration
│       │   ├── __init__.py
│       │   ├── runner.py           # full pipeline: typology → geometry → dynamics → causal
│       │   └── manifest.py         # feature manifest (locked selections)
│       ├── db/                     # DuckDB layer
│       │   ├── __init__.py
│       │   ├── connection.py
│       │   └── queries/            # .sql files — the "brain"
│       │       ├── typology.sql
│       │       ├── geometry.sql
│       │       ├── dynamics.sql
│       │       ├── causal.sql
│       │       └── feature_selection.sql
│       └── io/                     # data ingest (CSV, arrays, etc.)
│           ├── __init__.py
│           └── loaders.py
```

### Rules

- Orchestrators are pure orchestrators — NO inline compute
- All computation lives in orthon-engines (imported, never copied)
- SQL handles all logic: layers, joins, windows, aggregations, classifications
- ORTHON = dumb muscle (engines), SQL = brain (queries)
- No domain vocabulary in framework code

### pyproject.toml

```toml
[project]
name = "orthon"
version = "0.1.0"
description = "Domain-agnostic signal analysis framework"
requires-python = ">=3.9"
dependencies = [
    "orthon-engines>=0.1.0",
    "duckdb>=0.9",
]
```

### How orthon imports engines

```python
# orthon/registry/loader.py
from orthon_engines import hurst, fft, garch, lasso_select  # etc.

ENGINE_REGISTRY = {
    'hurst': hurst.compute,
    'fft': fft.compute,
    'garch': garch.compute,
    'lasso_select': lasso_select.compute,
    # ... all 41+
}

def run_engine(name: str, *args, **kwargs) -> dict:
    """Call an engine by name. Returns dict of numbers."""
    return ENGINE_REGISTRY[name](*args, **kwargs)
```

### Usage

```python
import orthon

# Full pipeline on raw signals
results = orthon.analyze(signals_df)

# Access specific layer
typology = orthon.typology.classify(signal)
print(typology['memory_class'])  # 'persistent', 'antipersistent', 'random'

# Feature selection on engine outputs
manifest = orthon.select_features(feature_matrix, target)
print(manifest['nonzero_names'])  # ['hurst', 'spectral_entropy', 'garch_persistence']
```

---

## Repo 3: `orthon-ml`

**Audience:** ML engineers, data scientists doing prediction  
**Install:** `pip install orthon-ml`  
**Dependencies:** orthon (which pulls in orthon-engines)

### What's in it

```
orthon-ml/
├── pyproject.toml
├── README.md                       # ML-focused: "extract 200+ features in one line"
├── LICENSE
├── src/
│   └── orthon_ml/
│       ├── __init__.py
│       ├── featurizer.py           # DataFrame in, feature matrix out
│       ├── selector.py             # scikit-learn compatible transformer
│       └── compat.py               # adapters for sklearn, xgboost, lightgbm
├── notebooks/
│   ├── quickstart.ipynb            # 10-line feature engineering replacement
│   ├── cmapss_benchmark.ipynb      # CMAPSS RUL demo
│   ├── vs_tsfresh.ipynb            # benchmark vs alternatives
│   └── vs_manual.ipynb             # before/after manual feature engineering
├── benchmarks/
│   └── feature_extraction_speed.py
└── tests/
    └── test_featurizer.py
```

### Rules

- NO engines (imports from orthon-engines via orthon)
- NO SQL orchestration (imports from orthon)
- NO mathematical theory in docs — practical ML vocabulary only
- This repo is a translation layer + demo notebooks + benchmarks
- Entire codebase should be < 500 lines

### pyproject.toml

```toml
[project]
name = "orthon-ml"
version = "0.1.0"
description = "ML feature engineering powered by ORTHON signal analysis"
requires-python = ">=3.9"
dependencies = [
    "orthon>=0.1.0",
    "pandas>=1.3",
]

[project.optional-dependencies]
sklearn = ["scikit-learn>=1.0"]
all = ["scikit-learn>=1.0", "xgboost", "lightgbm"]
```

### Core API (entire file)

```python
# orthon_ml/featurizer.py

import orthon
import pandas as pd
import numpy as np


class OrthonFeaturizer:
    """
    One-line feature engineering for time series.
    
    Usage:
        featurizer = OrthonFeaturizer()
        features = featurizer.fit_transform(df, target='RUL')
    """
    
    def __init__(self, select_features=True):
        self.select_features = select_features
        self.manifest_ = None
        self.feature_names_ = None
    
    def fit(self, df: pd.DataFrame, target: str = None, y: np.ndarray = None):
        """Run ORTHON engines, optionally select features via LASSO."""
        self.feature_matrix_, self.feature_names_ = orthon.featurize(df)
        
        if self.select_features and (target is not None or y is not None):
            target_values = df[target].values if target else y
            self.manifest_ = orthon.select_features(
                self.feature_matrix_, target_values, self.feature_names_
            )
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply locked feature manifest to new data."""
        features, _ = orthon.featurize(df)
        if self.manifest_ is not None:
            return features[:, self.manifest_['nonzero_indices']]
        return features
    
    def fit_transform(self, df, target=None, y=None) -> np.ndarray:
        return self.fit(df, target, y).transform(df)
    
    def get_feature_names(self) -> list:
        """Return selected feature names."""
        if self.manifest_ is not None:
            return self.manifest_['nonzero_names']
        return self.feature_names_
```

```python
# orthon_ml/selector.py  (scikit-learn compatible)

import numpy as np

class OrthonTransformer:
    """Drop-in scikit-learn transformer.
    
    from orthon_ml import OrthonTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    
    pipe = Pipeline([
        ('features', OrthonTransformer()),
        ('model', RandomForestRegressor())
    ])
    pipe.fit(X_train, y_train)
    """
    
    def __init__(self):
        self.featurizer = None
    
    def fit(self, X, y=None):
        from orthon_ml.featurizer import OrthonFeaturizer
        self.featurizer = OrthonFeaturizer(select_features=(y is not None))
        self.featurizer.fit_raw(X, y=y)
        return self
    
    def transform(self, X):
        return self.featurizer.transform_raw(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self):
        return self.featurizer.get_feature_names()
```

---

## Dependency Resolution

### What happens at install time

```bash
pip install orthon-ml
# → installs orthon (framework)
#   → installs orthon-engines (math)
#   → installs duckdb
# → installs pandas
# Done. Everything works.

pip install orthon
# → installs orthon-engines (math)
# → installs duckdb
# Done. ML layer not installed, not needed.

pip install orthon-engines
# → installs numpy
# Done. Just the raw compute.
```

### What happens at import time

```python
# In orthon (framework):
from orthon_engines import hurst       # direct import, same installed package

# In orthon-ml (shell):
import orthon                          # gets framework + engines transitively
from orthon_engines import lasso_select  # can also import engines directly
```

**No file copying. No version drift. One source of truth.**

When you update `hurst.py` in orthon-engines and bump the version,
`pip install --upgrade orthon-engines` updates it everywhere.

---

## File Management Benefits

| Before (monorepo) | After (three repos) |
|---|---|
| Copy engine files between directories | Import from one package |
| "Which version of hurst.py is current?" | Only one version exists |
| PR touches framework + engines + ML code | PRs scoped to one concern |
| README tries to speak to everyone | Each README speaks to its audience |
| One giant test suite | Focused tests per repo |
| Breaking an engine breaks everything | Engine tests run independently |

---

## Migration Plan

### Phase 1: Extract orthon-engines
1. Create `orthon-engines` repo
2. Move all engine `.py` files from current `engines/` directory
3. Add `pyproject.toml`, package as `orthon_engines`
4. Publish to PyPI (or test PyPI first)
5. Update current repo to `pip install orthon-engines` instead of local imports

### Phase 2: Clean up orthon core
1. Remove engine files from orthon repo (now imported)
2. Restructure into `typology/`, `geometry/`, `dynamics/`, `causal/`
3. Move SQL queries into `db/queries/`
4. Add `pyproject.toml` with orthon-engines dependency
5. Publish as `orthon`

### Phase 3: Create orthon-ml shell
1. Create `orthon-ml` repo
2. Write featurizer.py + selector.py (< 200 lines total)
3. Create demo notebooks (this IS the marketing)
4. Add benchmarks vs tsfresh, tsfel, catch22
5. Publish as `orthon-ml`

### Phase 4: CMAPSS benchmark
1. orthon-ml notebook: load CMAPSS FD001 training data
2. `OrthonFeaturizer().fit_transform(train_df, target='RUL')`
3. LASSO selects features, locks manifest
4. Apply to test data, predict RUL, report RMSE
5. Document everything. Publish results.

---

## Version Strategy

All three repos start at `0.1.0`. Use semantic versioning:

- **orthon-engines:** Bump minor for new engines, patch for bug fixes.
  Engine contract (arrays in, dict out) is the API — changing that is a major bump.
- **orthon:** Bump minor for new pipeline features, patch for SQL fixes.
  Pinned to orthon-engines `>=0.x` (compatible minor versions).
- **orthon-ml:** Bump minor for new notebooks/adapters, patch for fixes.
  Pinned to orthon `>=0.x`.

---

## Files to Create

### New repos
- `orthon-engines/` — new repo, receives all engine .py files
- `orthon-ml/` — new repo, shell + notebooks

### Modified
- `orthon/` — current repo, restructured to import from orthon-engines

### No files deleted
Engines are moved, not deleted. Git history preserved via `git mv` or
noted in commit message for cross-repo moves.
