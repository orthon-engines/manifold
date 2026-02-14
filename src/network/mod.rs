use pyo3::prelude::*;

pub mod centrality;
pub mod structure;
pub mod paths;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // centrality
    m.add_function(wrap_pyfunction!(centrality::centrality_degree, m)?)?;
    m.add_function(wrap_pyfunction!(centrality::centrality_betweenness, m)?)?;
    m.add_function(wrap_pyfunction!(centrality::centrality_eigenvector, m)?)?;
    m.add_function(wrap_pyfunction!(centrality::centrality_closeness, m)?)?;

    // structure
    m.add_function(wrap_pyfunction!(structure::threshold_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(structure::network_density, m)?)?;
    m.add_function(wrap_pyfunction!(structure::clustering_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(structure::assortativity, m)?)?;

    // paths
    m.add_function(wrap_pyfunction!(paths::shortest_paths, m)?)?;
    m.add_function(wrap_pyfunction!(paths::average_path_length, m)?)?;
    m.add_function(wrap_pyfunction!(paths::diameter, m)?)?;

    Ok(())
}
