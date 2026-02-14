use pyo3::prelude::*;

pub mod transfer_entropy;
pub mod mutual_info;
pub mod divergence;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // mutual_info
    m.add_function(wrap_pyfunction!(mutual_info::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_info::conditional_mutual_information, m)?)?;

    // transfer_entropy
    m.add_function(wrap_pyfunction!(transfer_entropy::transfer_entropy, m)?)?;

    // divergence
    m.add_function(wrap_pyfunction!(divergence::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(divergence::kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(divergence::js_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(divergence::hellinger_distance, m)?)?;
    m.add_function(wrap_pyfunction!(divergence::total_variation_distance, m)?)?;

    Ok(())
}
