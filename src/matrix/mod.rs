use pyo3::prelude::*;

pub mod covariance;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(covariance::covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(covariance::correlation_matrix, m)?)?;

    Ok(())
}
