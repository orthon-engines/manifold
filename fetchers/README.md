# PRISM Fetchers

Standalone data fetchers for acquiring raw observations.

**These are workspace scripts, NOT part of the prism package.**

## Design Principles

- Fetchers do NOT import from `prism`
- Each fetcher exposes one function: `fetch(config) -> list[dict]`
- Fetchers return raw data only - no database writes
- YAML files define what to fetch

## Available Fetchers

| Fetcher | Source | Requires |
|---------|--------|----------|
| `usgs_fetcher.py` | USGS Earthquake Catalog | (none) |
| `climate_fetcher.py` | Climate Data (NOAA, NASA) | (none) |
| `cmapss_fetcher.py` | NASA C-MAPSS Turbofan | (none) |
| `tep_fetcher.py` | Tennessee Eastman Process | (none) |

## Usage

### Direct Python

```python
from fetchers.usgs_fetcher import fetch

config = {
    "start_date": "2020-01-01",
    "min_magnitude": 5.0,
}
observations = fetch(config)
```

### Via PRISM Fetch Runner

```bash
python -m prism.entry_points.fetch fetchers/yaml/usgs.yaml
python -m prism.entry_points.fetch --cmapss
```

## Return Format

All fetchers return `list[dict]` with these keys:
- `indicator_id`: str
- `observed_at`: datetime
- `value`: float
- `source`: str

## YAML Configuration

Put fetch configs in `fetchers/yaml/`:

```yaml
# fetchers/yaml/usgs.yaml
source: usgs
start_date: "2010-01-01"
min_magnitude: 4.5
```

## Adding a New Fetcher

1. Create `{source}_fetcher.py` in this directory
2. Implement `fetch(config) -> list[dict]`
3. Create `yaml/{source}.yaml` with indicator list
4. Do NOT import anything from `prism`
