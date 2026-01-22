"""
Ørthon Analysis Dashboard
Regime-Aware Behavioral Geometry for Predictive Diagnostics
"""

import streamlit as st
import duckdb
import polars as pl
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for orthon imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import orthon
    from orthon.cohort import discover_cohorts, ConfidenceMetrics
    from orthon.report import compare_cohorts, generate_report
    ORTHON_AVAILABLE = True
except ImportError:
    ORTHON_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Ørthon",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DuckDB connection
@st.cache_resource
def get_db():
    return duckdb.connect(":memory:")

conn = get_db()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_parquet(file_path: str) -> pd.DataFrame:
    """Load parquet file into DuckDB and return as DataFrame"""
    conn.execute(f"CREATE OR REPLACE TABLE data AS SELECT * FROM read_parquet('{file_path}')")
    return conn.execute("SELECT * FROM data LIMIT 1000").df()

@st.cache_data
def get_columns(file_path: str) -> list:
    """Get column names from parquet file"""
    return conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").df()['column_name'].tolist()

@st.cache_data
def get_unique_values(file_path: str, column: str) -> list:
    """Get unique values for a column (for dropdowns)"""
    result = conn.execute(f"SELECT DISTINCT {column} FROM read_parquet('{file_path}') ORDER BY {column}").df()
    return result[column].tolist()

def run_query(query: str) -> pd.DataFrame:
    """Run arbitrary SQL query"""
    return conn.execute(query).df()

# ============================================================
# SIDEBAR - FILTERS
# ============================================================

st.sidebar.title("🔬 Ørthon")
st.sidebar.caption("Behavioral Geometry Analysis")

st.sidebar.markdown("---")

# File upload or path
st.sidebar.subheader("📁 Data Source")
data_source = st.sidebar.radio("Load data from:", ["Upload", "File Path"], horizontal=True)

data_loaded = False
file_path = None

if data_source == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Parquet", type=['parquet'])
    if uploaded_file:
        # Save to temp location
        temp_path = Path("/tmp/uploaded_data.parquet")
        temp_path.write_bytes(uploaded_file.read())
        file_path = str(temp_path)
        data_loaded = True
else:
    file_path = st.sidebar.text_input("Parquet file path:", placeholder="/path/to/data.parquet")
    if file_path and Path(file_path).exists():
        data_loaded = True
    elif file_path:
        st.sidebar.error("File not found")

st.sidebar.markdown("---")

# SQL Filters (only show if data loaded)
if data_loaded:
    st.sidebar.subheader("🔍 Filters")
    
    columns = get_columns(file_path)
    
    # Entity filter
    entity_col = st.sidebar.selectbox("Entity Column:", columns, index=0)
    entities = get_unique_values(file_path, entity_col)
    selected_entities = st.sidebar.multiselect("Select Entities:", entities, default=entities[:5] if len(entities) > 5 else entities)
    
    # Time/Cycle filter
    time_col = st.sidebar.selectbox("Time Column:", columns, index=min(1, len(columns)-1))
    
    # Additional filter columns
    st.sidebar.markdown("**Additional Filters:**")
    filter_col = st.sidebar.selectbox("Filter by:", ["None"] + columns)
    if filter_col != "None":
        filter_vals = get_unique_values(file_path, filter_col)
        selected_filter = st.sidebar.multiselect(f"Select {filter_col}:", filter_vals)
    
    st.sidebar.markdown("---")
    
    # Section visibility
    st.sidebar.subheader("📊 Show Sections")
    show_vector = st.sidebar.checkbox("Vector Layer", value=True)
    show_geometry = st.sidebar.checkbox("Geometry Layer", value=True)
    show_state = st.sidebar.checkbox("State Layer", value=True)
    show_derivatives = st.sidebar.checkbox("Derivatives", value=False)
    
    st.sidebar.markdown("---")
    
    # Window parameters
    st.sidebar.subheader("⚙️ Window Settings")
    window_size = st.sidebar.slider("Window Size:", 5, 100, 20)
    stride = st.sidebar.slider("Stride:", 1, 50, 5)
    
    # Build SQL WHERE clause
    where_clauses = []
    if selected_entities:
        entities_str = ", ".join([f"'{e}'" for e in selected_entities])
        where_clauses.append(f"{entity_col} IN ({entities_str})")
    if filter_col != "None" and selected_filter:
        filter_str = ", ".join([f"'{f}'" for f in selected_filter])
        where_clauses.append(f"{filter_col} IN ({filter_str})")
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

# ============================================================
# MAIN CONTENT - TABS
# ============================================================

tabs = st.tabs([
    "📋 Data Summary",
    "📈 Signal Typology", 
    "🔷 Behavioral Geometry",
    "🔄 State",
    "⚡ Derivatives",
    "🎯 Advanced Analysis",
    "🤖 Machine Learning"
])

# ------------------------------------------------------------
# TAB 1: Data Summary
# ------------------------------------------------------------
with tabs[0]:
    st.header("Data Summary")
    
    if not data_loaded:
        st.info("👈 Load a Parquet file from the sidebar to begin.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        # Get summary stats via SQL
        summary = run_query(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT {entity_col}) as unique_entities,
                MIN({time_col}) as min_time,
                MAX({time_col}) as max_time
            FROM read_parquet('{file_path}')
            WHERE {where_sql}
        """)
        
        col1.metric("Total Rows", f"{summary['total_rows'][0]:,}")
        col2.metric("Unique Entities", summary['unique_entities'][0])
        col3.metric("Time Range Start", summary['min_time'][0])
        col4.metric("Time Range End", summary['max_time'][0])
        
        st.markdown("---")
        
        # Column info
        st.subheader("Column Summary")
        col_info = run_query(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')")
        st.dataframe(col_info, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        sample = run_query(f"""
            SELECT * FROM read_parquet('{file_path}')
            WHERE {where_sql}
            LIMIT 100
        """)
        st.dataframe(sample, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: Signal Typology
# ------------------------------------------------------------
with tabs[1]:
    st.header("Signal Typology")
    st.caption("Classify signals: Deterministic, Stochastic, Mixed")
    
    if not data_loaded:
        st.info("👈 Load data to analyze signal types.")
    else:
        # Graph controls
        gcol1, gcol2, gcol3 = st.columns([2, 1, 1])
        with gcol1:
            sensor_cols = [c for c in columns if c not in [entity_col, time_col]]
            selected_sensor = st.selectbox("Select Sensor:", sensor_cols)
        with gcol2:
            chart_type = st.selectbox("Chart Type:", ["Line", "Histogram", "Box Plot"])
        with gcol3:
            single_entity = st.selectbox("Entity:", selected_entities if selected_entities else entities[:1])
        
        # Fetch data for visualization
        viz_data = run_query(f"""
            SELECT {time_col}, {selected_sensor}
            FROM read_parquet('{file_path}')
            WHERE {entity_col} = '{single_entity}'
            ORDER BY {time_col}
        """)
        
        # Display chart
        if chart_type == "Line":
            st.line_chart(viz_data, x=time_col, y=selected_sensor)
        elif chart_type == "Histogram":
            st.bar_chart(viz_data[selected_sensor].value_counts())
        else:
            st.bar_chart(viz_data[[selected_sensor]])
        
        # Three graphs below
        st.markdown("---")
        g1, g2, g3 = st.columns(3)
        
        with g1:
            st.markdown("**Autocorrelation**")
            st.caption("Coming soon...")
            
        with g2:
            st.markdown("**Spectral Density**")
            st.caption("Coming soon...")
            
        with g3:
            st.markdown("**Stationarity Test**")
            st.caption("Coming soon...")

# ------------------------------------------------------------
# TAB 3: Behavioral Geometry
# ------------------------------------------------------------
with tabs[2]:
    st.header("Behavioral Geometry")
    st.caption("Pairwise relationships between signals")
    
    if not data_loaded:
        st.info("👈 Load data to analyze geometry.")
    else:
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            sensor_x = st.selectbox("Sensor X:", sensor_cols, key="geo_x")
        with gcol2:
            sensor_y = st.selectbox("Sensor Y:", sensor_cols, index=min(1, len(sensor_cols)-1), key="geo_y")
        
        chart_type_geo = st.selectbox("Chart Type:", ["Scatter", "Heatmap", "Line Overlay"], key="geo_chart")
        
        # Fetch pairwise data
        pair_data = run_query(f"""
            SELECT {time_col}, {sensor_x}, {sensor_y}
            FROM read_parquet('{file_path}')
            WHERE {entity_col} = '{single_entity}'
            ORDER BY {time_col}
        """)
        
        if chart_type_geo == "Scatter":
            st.scatter_chart(pair_data, x=sensor_x, y=sensor_y)
        else:
            st.line_chart(pair_data, x=time_col, y=[sensor_x, sensor_y])
        
        # Correlation matrix
        st.markdown("---")
        st.subheader("Correlation Matrix")
        st.caption("Pairwise correlations across all sensors")
        
        if st.button("Compute Correlations"):
            corr_data = run_query(f"""
                SELECT {', '.join(sensor_cols[:10])}
                FROM read_parquet('{file_path}')
                WHERE {where_sql}
            """)
            corr_matrix = corr_data.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))

# ------------------------------------------------------------
# TAB 4: State
# ------------------------------------------------------------
with tabs[3]:
    st.header("State Analysis")
    st.caption("Transfer Entropy, Regime Detection, Coherence Tracking")
    
    if not data_loaded:
        st.info("👈 Load data to analyze state dynamics.")
    else:
        analysis_type = st.selectbox("Analysis Type:", [
            "Transfer Entropy",
            "Regime Detection", 
            "Coherence Tracking (hd_slope)",
            "Phase Space"
        ])
        
        st.markdown("---")
        
        if analysis_type == "Coherence Tracking (hd_slope)":
            st.markdown("""
            **hd_slope** measures the velocity of coherence loss — how fast 
            a system is moving away from baseline behavior.
            """)
            
            # Placeholder for actual computation
            st.info("Connect to your Vector/Geometry layer outputs to compute hd_slope")
            
        elif analysis_type == "Transfer Entropy":
            te_col1, te_col2 = st.columns(2)
            with te_col1:
                source_sensor = st.selectbox("Source:", sensor_cols, key="te_source")
            with te_col2:
                target_sensor = st.selectbox("Target:", sensor_cols, key="te_target")
            
            st.info("Transfer entropy computation placeholder")

# ------------------------------------------------------------
# TAB 5: Derivatives
# ------------------------------------------------------------
with tabs[4]:
    st.header("Engine Derivatives")
    st.caption("Computed derivatives and rate-of-change features")
    
    if not data_loaded:
        st.info("👈 Load data to compute derivatives.")
    else:
        derivative_type = st.selectbox("Derivative Type:", [
            "First Derivative (velocity)",
            "Second Derivative (acceleration)",
            "Rolling Rate of Change",
            "Laplace Transform Features"
        ])
        
        deriv_sensor = st.selectbox("Sensor:", sensor_cols, key="deriv_sensor")
        
        st.markdown("---")
        st.info(f"Derivative computation for {deriv_sensor} — connect to processing pipeline")

# ------------------------------------------------------------
# TAB 6: Advanced Analysis
# ------------------------------------------------------------
with tabs[5]:
    st.header("Advanced Analysis")
    st.caption("Cohort Discovery, Failure Mode Classification")
    
    if not data_loaded:
        st.info("👈 Load data for advanced analysis.")
    else:
        analysis_mode = st.selectbox("Analysis Mode:", [
            "Cohort Discovery",
            "Failure Mode Classification",
            "Anomaly Detection",
            "RUL Estimation"
        ])
        
        st.markdown("---")
        
        if analysis_mode == "Cohort Discovery":
            st.markdown("""
            Identify groups of entities that exhibit similar behavioral patterns.

            - **Signal Types**: Cluster signals by behavioral fingerprint
            - **Structural Groups**: Cluster entities by correlation patterns
            - **Temporal Cohorts**: Cluster entities by trajectory dynamics
            """)

            if st.button("Run Cohort Analysis"):
                if not ORTHON_AVAILABLE:
                    st.error("Orthon module not found. Run `pip install -e .` from repo root.")
                else:
                    # Check for required parquet files
                    data_dir = Path(file_path).parent
                    vector_path = data_dir / "vector.parquet"
                    geometry_path = data_dir / "geometry.parquet"
                    state_path = data_dir / "state.parquet"

                    if not vector_path.exists():
                        st.warning(f"vector.parquet not found in {data_dir}")
                    elif not geometry_path.exists():
                        st.warning(f"geometry.parquet not found in {data_dir}")
                    elif not state_path.exists():
                        st.warning(f"state.parquet not found in {data_dir}")
                    else:
                        with st.spinner("Running 3-layer cohort discovery..."):
                            result = discover_cohorts(
                                vector_path=str(vector_path),
                                geometry_path=str(geometry_path),
                                state_path=str(state_path),
                            )

                        st.success("Cohort discovery complete!")

                        # Display results
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Signal Types", result.n_signal_types)
                        col2.metric("Structural Groups", result.n_structural_groups)
                        col3.metric("Temporal Cohorts", result.n_temporal_cohorts)

                        # Confidence scores
                        st.subheader("Confidence Metrics")
                        if result.signal_type_confidence:
                            st.write(f"**Signal Types**: {result.signal_type_confidence.composite_score:.3f} ({result.signal_type_confidence.interpretation})")
                        if result.structural_confidence:
                            st.write(f"**Structural Groups**: {result.structural_confidence.composite_score:.3f} ({result.structural_confidence.interpretation})")

                        # Show cohort assignments
                        st.subheader("Cohort Assignments")
                        st.dataframe(result.to_dataframe())

        elif analysis_mode == "Failure Mode Classification":
            st.markdown("""
            Compare an experimental entity against a control cohort to identify
            structural and temporal divergences.
            """)

            if ORTHON_AVAILABLE and data_loaded:
                data_dir = Path(file_path).parent
                geometry_path = data_dir / "geometry.parquet"
                state_path = data_dir / "state.parquet"

                if geometry_path.exists() and state_path.exists():
                    # Get entities for selection
                    exp_entity = st.selectbox("Experiment Entity:", entities, key="exp_entity")
                    ctrl_entities = st.multiselect(
                        "Control Entities:",
                        [e for e in entities if e != exp_entity],
                        default=[e for e in entities if e != exp_entity][:3],
                        key="ctrl_entities"
                    )

                    if st.button("Generate Comparison Report"):
                        if not ctrl_entities:
                            st.warning("Select at least one control entity")
                        else:
                            with st.spinner("Comparing cohorts..."):
                                result = compare_cohorts(
                                    experiment_ids=[exp_entity],
                                    control_ids=ctrl_entities,
                                    geometry_path=str(geometry_path),
                                    state_path=str(state_path),
                                )
                                report_html = generate_report(result, output_format="html")

                            st.success("Comparison complete!")

                            # Display report
                            st.components.v1.html(report_html, height=600, scrolling=True)

                            # Download button
                            st.download_button(
                                "📥 Download Report (HTML)",
                                report_html,
                                file_name="comparison_report.html",
                                mime="text/html"
                            )
                else:
                    st.warning("geometry.parquet and state.parquet required for comparison")
            elif not ORTHON_AVAILABLE:
                st.error("Orthon module not found")

# ------------------------------------------------------------
# TAB 7: Machine Learning
# ------------------------------------------------------------
with tabs[6]:
    st.header("Machine Learning")
    
    if not data_loaded:
        st.info("👈 Load data to train models.")
    else:
        st.subheader("1. Select Features")
        
        feature_sources = st.multiselect(
            "Feature Sources:",
            ["Vector Layer", "Geometry Layer", "State Layer", "Derivatives", "Cohort Features"],
            default=["Vector Layer", "Geometry Layer"]
        )
        
        st.markdown("---")
        st.subheader("2. Select Model")
        
        model_type = st.selectbox("Model:", [
            "XGBoost",
            "CatBoost", 
            "LightGBM",
            "Random Forest",
            "Logistic Regression"
        ])
        
        st.markdown("---")
        st.subheader("3. Train/Test Split")
        
        split_col1, split_col2 = st.columns(2)
        with split_col1:
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        with split_col2:
            random_state = st.number_input("Random State:", value=42)
        
        st.markdown("---")
        st.subheader("4. Train Model")
        
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training..."):
                    # Placeholder for actual training
                    import time
                    time.sleep(2)
                    st.success("Model trained! (placeholder)")
        
        with tcol2:
            if st.button("📥 Download Test Predictions"):
                st.info("Generate predictions parquet file")
        
        st.markdown("---")
        st.subheader("5. Export")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("📄 Export Results (Parquet)"):
                st.info("Export to results.parquet")
        with exp_col2:
            if st.button("📊 Export Report (HTML)"):
                st.info("Generate HTML report")

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
if ORTHON_AVAILABLE:
    st.sidebar.caption(f"Ørthon v{orthon.__version__} | Systems lose coherence before they fail.")
else:
    st.sidebar.caption("Ørthon (module not loaded) | Systems lose coherence before they fail.")
