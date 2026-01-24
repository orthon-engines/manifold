# ORTHON Dashboard

Streamlit dashboard for ORTHON signal analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Data Directory

Place parquet files in `data/`:

```
data/
  observations.parquet          # Raw signal data (from UCI Hydraulic fetch)
  signal_typology_profile.parquet   # Normalized 0-1 scores per axis
  signal_typology_metrics.parquet   # Raw engine outputs
```

### observations.parquet

Expected columns:
- Signal columns (sensor readings, measurements)
- Optional: `timestamp`, `signal_id`

### signal_typology_profile.parquet

| Column | Type |
|--------|------|
| signal_id | str |
| memory | float |
| information | float |
| frequency | float |
| volatility | float |
| wavelet | float |
| derivatives | float |
| recurrence | float |
| discontinuity | float |
| momentum | float |

### signal_typology_metrics.parquet

Raw engine outputs (hurst_exponent, acf_decay_rate, etc.)

## Features

- **Signal tab**: Time series visualization, basic stats
- **Typology tab**: Radar chart, classification table
- **Metrics tab**: Raw engine outputs
- **Compare tab**: Overlay multiple signals

## Configuration

Thresholds are configurable in `app.py`:

```python
DEFAULT_THRESHOLDS = [0.25, 0.4, 0.6, 0.75]
```

Future: User-adjustable thresholds via sidebar.
