# Ørthon Analysis Dashboard

Interactive analysis dashboard for regime-aware behavioral geometry.

## Quick Start

### 1. Install dependencies

```bash
cd orthon-app
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 3. Load your data

- Upload a Parquet file via the sidebar, OR
- Enter the path to a local Parquet file

---

## Features

| Tab | Description |
|-----|-------------|
| **Data Summary** | Row counts, column types, sample data |
| **Signal Typology** | Classify signals (deterministic, stochastic, mixed) |
| **Behavioral Geometry** | Pairwise sensor relationships, correlation matrices |
| **State** | Transfer entropy, regime detection, coherence tracking |
| **Derivatives** | Rate-of-change features, Laplace transforms |
| **Advanced Analysis** | Cohort discovery, failure mode classification |
| **Machine Learning** | Train XGBoost/CatBoost/LightGBM, export predictions |

## Sidebar Controls

- **Filters**: SQL-driven dropdowns populated from your data
- **Section Visibility**: Toggle which analysis layers to include
- **Window Settings**: Configure window size and stride for rolling computations

## Outputs

- **Parquet**: Processed features and predictions
- **HTML**: Publication-ready reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit App (app.py)                                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Sidebar   │  │    Tabs     │  │   Graphs    │         │
│  │   Filters   │  │   Content   │  │   Display   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │     DuckDB      │                         │
│                 │   (SQL Engine)  │                         │
│                 └────────┬────────┘                         │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Local Parquet  │
                  │     Files       │
                  └─────────────────┘
```

## Next Steps

This is a scaffold. To make it fully functional:

1. **Connect to your processing pipeline** — Import your Vector/Geometry/State layer code
2. **Add actual computations** — Replace placeholders with real analysis
3. **Wire up ML training** — Connect to scikit-learn/XGBoost pipelines
4. **Add export functions** — Generate Parquet outputs and HTML reports

---

## File Structure

```
orthon-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

Future structure:

```
orthon-app/
├── app.py
├── requirements.txt
├── components/         # Reusable UI components
│   ├── filters.py
│   ├── charts.py
│   └── exporters.py
├── analysis/           # Analysis logic
│   ├── vector.py
│   ├── geometry.py
│   └── state.py
├── ml/                 # ML training pipelines
│   ├── train.py
│   └── predict.py
└── templates/          # HTML report templates
    └── report.html
```
