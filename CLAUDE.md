# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRISM (Persistent Relational Inference & Structural Measurement) is a behavioral geometry engine for time-series analysis. It measures intrinsic properties, relational structure, and temporal dynamics. **The math interprets; we don't add narrative.**

**Version 2.0 Architecture: Pure Polars + Parquet**
- All storage via Parquet files (no database)
- All I/O via Polars DataFrames (no pandas for I/O)
- Pandas only at engine boundaries (scipy/sklearn compatibility)
- No DuckDB dependency

**Core Philosophy:**
- Record reality faithfully
- Let math speak
- The geometry interprets - we don't add opinion
- Parquet is truth (all measurements persist to Parquet files)
- Explicit time (nothing inferred between steps)
- No implicit execution (importing does nothing)

**Academic Research Standards:**
- **NO SHORTCUTS** - All engines use complete data (no subsampling)
- **NO APPROXIMATIONS** - Peer-reviewed algorithms (antropy, pyrqa)
- **NO SPEED HACKS** - 2-3 hour runs acceptable, 2-3 week runs expected
- **VERIFIED QUALITY** - All 21 engines audited for data integrity
  - Vector: antropy (entropy), pyrqa (RQA), full R/S (Hurst)
  - Geometry: Complete matrices, no dimensionality shortcuts
  - State: Full time series, proper cointegration tests
- **Publication-grade** - Suitable for peer-reviewed research

## Essential Commands

### Initialize Data Directories
```bash
# Create data directories
python -m prism.db.parquet_store --init

# Validate structure
python -m prism.db.parquet_store --validate
```

### Data Fetching
```bash
# Fetch climate data
python -m prism.entry_points.fetch --climate

# Fetch C-MAPSS turbofan data
python -m prism.entry_points.fetch --cmapss
```

### Vector Computation
```bash
# Run vector engines on all indicators
python -m prism.entry_points.indicator_vector

# Specific domain with testing mode
python -m prism.entry_points.indicator_vector --domain cmapss --testing

# Parallel execution
python -m prism.entry_points.indicator_vector --workers 4 --testing
```

### Code Quality
```bash
# Format code
black prism/ --line-length 100

# Type checking
mypy prism/

# Linting
ruff check prism/
```

### Testing
```bash
# Run tests
pytest

# With coverage
pytest --cov=prism --cov-report=html
```

## Architecture

### Storage: Pure Polars + Parquet

PRISM uses **Parquet files** for all storage with **Polars** for all I/O operations.

**Key Design Decisions (v2.0):**
- **No DuckDB** - Removed entirely from codebase
- **Polars for I/O** - All file reads/writes use Polars
- **Pandas at boundaries** - Only used where scipy/sklearn require it
- **Atomic writes** - Temp files merged atomically for parallel safety

**Benefits:**
- No database file locking (Parquet files are immutable)
- No schema migrations (Parquet has embedded schema)
- Parallel-safe writes (each worker writes separate temp file, merge after)
- Simpler deployment (just files, no database server)

**Directory Structure:**
```
data/
  raw/
    observations.parquet      # Time series data
    indicators.parquet        # Indicator metadata
    characterization.parquet  # 6-axis classification
  config/
    cohort_members.parquet    # Cohort assignments
    cohorts.parquet           # Cohort definitions
    domain_members.parquet    # Domain → cohort mapping
  vector/
    indicator.parquet         # Layer 0: 51 metrics per indicator
    cohort.parquet            # Layer 2: Aggregated cohort fingerprint (long format)
    cohort_wide.parquet       # Layer 2: Cohort fingerprint (wide format for geometry)
  geometry/
    cohort.parquet            # Layer 1: Cohort-level structural metrics
    indicator_pair.parquet    # Layer 1: Pairwise indicator geometry
    domain.parquet            # Layer 3: Domain-level structural metrics
    cohort_pair.parquet       # Layer 3: Pairwise cohort geometry
  state/
    indicators.parquet        # Layer 4: Query-time state
  filter/
    deep_curated.parquet      # Curated indicators (non-redundant)
    deep_redundant.parquet    # Redundant indicators to exclude
    deep_pairs.parquet        # Full pair analysis
```

### Database Layer Usage

```python
import polars as pl
from prism.db.parquet_store import get_parquet_path, ensure_directories
from prism.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic

# Read data
observations = pl.read_parquet(get_parquet_path('raw', 'observations'))

# Filtered read
filtered = observations.filter(pl.col('indicator_id') == 'sensor_1')

# Write with upsert (deduplicates by key columns)
upsert_parquet(
    df,
    get_parquet_path('vector', 'indicators'),
    key_cols=['indicator_id', 'obs_date', 'engine', 'metric_name']
)

# Atomic write (overwrites entire file)
write_parquet_atomic(df, get_parquet_path('geometry', 'cohorts'))
```

### Seven-Layer Pipeline Architecture

PRISM: Progressive Regime Identification through Structural Mathematics

```
Layer 0: OBSERVATIONS (raw/observations.parquet)
         Source data → time series
         "What are the raw measurements?"
         Output: raw/observations.parquet

Layer 1: INDICATOR VECTOR (indicator_vector.py)
         Raw observations → 51 behavioral metrics
         "What is this indicator doing in isolation?"
         Output: vector/indicator.parquet

Layer 2: COHORT GEOMETRY (cohort_geometry.py)
         Indicator vectors → pairwise relationships + cohort structure
         "How do indicators relate within their cohort?"
         Output: geometry/cohort.parquet, geometry/indicator_pair.parquet

Layer 3: COHORT VECTOR (cohort_vector.py)
         Aggregate indicator vectors → cohort behavioral fingerprint
         "What is this cohort doing as a unit?"
         Output: vector/cohort.parquet, vector/cohort_wide.parquet

Layer 4: DOMAIN GEOMETRY (domain_geometry.py)
         Cohort vectors → pairwise relationships + domain structure
         "How do cohorts relate within the domain?"
         Output: geometry/domain.parquet, geometry/cohort_pair.parquet

Layer 5: STATE (indicator_state.py, cohort_state.py)
         For any indicator at time t:
         ├─ indicator_vector[t]           (from Layer 1)
         ├─ position_in_cohort[t]         (from Layer 2)
         ├─ cohort_vector[t]              (from Layer 3)
         ├─ cohort_position_in_domain[t]  (from Layer 4)
         └─ DERIVED: indicator_position_in_domain[t]

         "Where does this indicator sit in the full hierarchy?"
         Output: state/indicator.parquet, state/cohort.parquet

Layer 6: PHYSICS (physics.py) [EXPERIMENTAL]
         State trajectories → physics metrics
         Tests: Energy conservation, entropy increase, least action
         "Do universal laws hold in behavioral space?"
         Output: physics/indicator.parquet, physics/cohort.parquet, physics/conservation.parquet

REGIME CHANGE = geometric deformation at any layer
```

### Engine Types

**Vector Engines (7)** - Intrinsic properties of single series
- Engines: Hurst, Entropy, GARCH, Wavelet, Spectral, Lyapunov, RQA

**Geometry** - Computed directly in runners (PCA, MST, clustering, LOF, distance, etc.)

**State Engines (6)** - Temporal dynamics over time
- Engines: Granger, Cross-Correlation, Cointegration, DTW, DMD, Transfer Entropy

### Modules (prism/modules/)

**Reusable computation modules called inline by runners:**
- `characterize.py`: Inline characterization of indicators
- `laplace.py`: Laplace field computation (gradient, divergence)
- `modes.py`: Behavioral mode discovery from Laplace signatures
- `wavelet_microscope.py`: Frequency-band degradation detection

**Key principle**: Modules are NOT standalone entrypoints - they're building blocks imported by runners.

```python
# Runners import modules inline
from prism.modules.modes import discover_modes, compute_affinity_weighted_features
from prism.modules.wavelet_microscope import run_wavelet_microscope
```

### Entrypoint Architecture

**CLI Entrypoints (python -m prism.entry_points.*):**
- `fetch.py`: Fetch external data (USGS, climate, etc.)
- `indicator_vector.py`: Compute vector metrics per indicator
- `geometry.py`: Compute cohort geometry (calls modes/wavelet modules inline)
- `characterize.py`: One-time indicator characterization
- `state.py`: Query-time state derivation
- `report.py`: Generate reports

**NOT entrypoints (use prism.entry_points instead):**
- Domain-specific scripts in `scripts/` are for evaluation/testing only
- Modules in `prism/modules/` are imported by entrypoints, not run directly

## Key Patterns & Conventions

### Reading Data
```python
import polars as pl
from prism.db.parquet_store import get_parquet_path

# Simple read
observations = pl.read_parquet(get_parquet_path('raw', 'observations'))

# Lazy scan for large files
obs_lazy = pl.scan_parquet(get_parquet_path('raw', 'observations'))
filtered = obs_lazy.filter(pl.col('indicator_id') == 'sensor_1').collect()

# Read with column selection
df = pl.read_parquet(path, columns=['indicator_id', 'obs_date', 'value'])
```

### Writing Data
```python
from prism.db.polars_io import upsert_parquet, write_parquet_atomic

# Upsert (preserves existing rows, updates by key)
upsert_parquet(df, target_path, key_cols=['indicator_id', 'obs_date'])

# Atomic write (replaces entire file)
write_parquet_atomic(df, target_path)
```

### Batch Processing Pattern
```python
# Collect results in list, write once at end
rows = []
for item in items:
    result = compute_something(item)
    rows.append(result)

# Single write operation
df = pl.DataFrame(rows, infer_schema_length=None)
upsert_parquet(df, target_path, key_cols)
```

### Parallel Processing
```python
from prism.db.scratch import TempParquet, merge_temp_files

# Worker writes to isolated temp file
with TempParquet(prefix='worker_0') as temp:
    results_df = compute_metrics(...)
    temp.write(results_df)

# After all workers complete, merge results
merge_temp_files(temp_paths, target_path, key_cols)
```

## Directory Structure

```
prism/                      # CORE PACKAGE (pip installable)
├── db/                     # Database layer (Parquet-based)
│   ├── __init__.py         # Exports all DB functions
│   ├── parquet_store.py    # Path management, directory structure
│   ├── polars_io.py        # Atomic writes, upsert operations
│   ├── query.py            # Table introspection utilities
│   └── scratch.py          # TempParquet for parallel workers
│
├── engines/                # ALL 21 ENGINES (~7,300 lines)
│   ├── __init__.py         # Engine registry
│   ├── engine_base.py      # BaseEngine, EngineResult contracts
│   └── [engine files]      # hurst.py, pca.py, granger.py, etc.
│
├── entry_points/           # CLI entrypoints (python -m prism.entry_points.*)
│   ├── fetch.py            # Data fetching to Parquet
│   ├── characterize.py     # 6-axis indicator classification
│   ├── indicator_vector.py # Layer 1: Vector metrics (51 per indicator)
│   ├── geometry.py         # Layer 2-3: Geometry + modes + wavelet
│   ├── state.py            # Layer 4: Query-time state derivation
│   └── report.py           # Generate reports
│
├── modules/                # Reusable computation (NOT entrypoints)
│   ├── characterize.py     # Inline characterization
│   ├── laplace.py          # Laplace field computation
│   ├── modes.py            # Mode discovery from signatures
│   └── wavelet_microscope.py # Frequency-band degradation
│
├── cohorts/                # Cohort definitions
│   └── climate.py          # Climate cohorts
│
└── utils/                  # Shared utilities
    ├── stride.py           # Window/stride configuration
    └── bisection.py        # Regime analysis

fetchers/                   # DATA FETCHERS (standalone)
├── usgs_fetcher.py         # USGS earthquake data
├── climate_fetcher.py      # Climate data (NOAA, NASA)
├── cmapss_fetcher.py       # NASA C-MAPSS turbofan data
├── tep_fetcher.py          # Tennessee Eastman Process data
└── yaml/                   # Fetch configurations

config/                     # YAML configurations
├── registry_*.yaml         # Indicator registries by domain
├── stride.yaml             # Window/stride configuration
└── geometry.yaml           # Engine configurations

data/                       # Parquet storage (gitignored)
├── raw/                    # Layer 1: Raw observations
├── config/                 # Configuration tables
├── vector/                 # Layer 2: Vector metrics
├── geometry/               # Layer 3: Structural snapshots
├── state/                  # Layer 4: Temporal dynamics
└── filter/                 # Redundancy analysis results
```

## Parallel Execution (No Lock Issues)

Parquet files have **no locking**. Parallel processing is safe:

```bash
# Parallel workers write to isolated temp files
python -m prism.entry_points.indicator_vector --workers 4 --testing

# Each worker:
# 1. Reads from shared Parquet (no lock needed)
# 2. Writes to isolated temp Parquet
# 3. Results merged atomically after all complete
```

**Timing expectations:**
- Small datasets (~100 indicators): ~5-10 minutes
- Medium datasets (1000+ indicators): 2-3 hours
- Large datasets (5000+ indicators): 2-3 weeks

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary I/O), Pandas (engine compatibility)
- **Parquet:** PyArrow (via Polars)
- **Core:** NumPy, SciPy, scikit-learn
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx
- **Testing:** pytest
- **Code Quality:** black, ruff, mypy

## Environment Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/prism-engines/prism-core.git
   cd prism-core
   pip install -e ".[dev,all]"
   ```

2. **Initialize data directories:**
   ```bash
   python -m prism.db.parquet_store --init
   ```

3. **Fetch initial data:**
   ```bash
   python -m prism.entry_points.fetch --cmapss
   ```

## What PRISM Does and Does NOT Do

**PRISM interprets with math, not narrative.** The geometry speaks for itself.

PRISM does NOT:
- Predict timing or outcomes
- Recommend actions
- Add opinion or spin
- Generate narrative explanations

PRISM DOES:
- Show you the shape of structural stress
- Identify when current geometry matches historical patterns
- Reveal which indicators belong together
- Detect regime boundaries mathematically

## Migration from DuckDB (v1.x -> v2.0)

Version 2.0 completely removes DuckDB in favor of pure Polars + Parquet:

| v1.x (DuckDB) | v2.0 (Polars + Parquet) |
|---------------|-------------------------|
| `duckdb.connect(db_path)` | `pl.read_parquet(path)` |
| `conn.execute(sql)` | `df.filter(...).select(...)` |
| `INSERT INTO table` | `upsert_parquet(df, path, keys)` |
| `ScratchDB` temporary tables | `TempParquet` temporary files |
| Schema migrations | Parquet schema embedded |
| Database locking | No locking (immutable files) |

**Removed:**
- `prism/db/open.py` - DuckDB connections
- `prism/db/migrate.py` - Schema migrations
- `prism/db/reset.py` - Database reset
- All `--db` CLI arguments
- All `duckdb` imports

**New:**
- `prism/db/parquet_store.py` - Path management
- `prism/db/polars_io.py` - Atomic I/O operations
- `prism/db/scratch.py` - Parallel temp files
