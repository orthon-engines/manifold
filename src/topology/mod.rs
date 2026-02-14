use pyo3::prelude::*;

pub mod persistence;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistence::persistence_diagram, m)?)?;
    m.add_function(wrap_pyfunction!(persistence::betti_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(persistence::persistence_entropy, m)?)?;

    Ok(())
}
