use pyo3::prelude::*;

pub mod delay;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delay::time_delay_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(delay::optimal_delay, m)?)?;
    m.add_function(wrap_pyfunction!(delay::optimal_dimension, m)?)?;

    Ok(())
}
