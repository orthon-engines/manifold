use pyo3::prelude::*;

pub mod lyapunov;
pub mod ftle;
pub mod rqa;
pub mod saddle;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // lyapunov
    m.add_function(wrap_pyfunction!(lyapunov::lyapunov_rosenstein, m)?)?;
    m.add_function(wrap_pyfunction!(lyapunov::lyapunov_kantz, m)?)?;
    m.add_function(wrap_pyfunction!(lyapunov::lyapunov_spectrum, m)?)?;

    // ftle
    m.add_function(wrap_pyfunction!(ftle::ftle_local_linearization, m)?)?;
    m.add_function(wrap_pyfunction!(ftle::ftle_direct_perturbation, m)?)?;
    m.add_function(wrap_pyfunction!(ftle::compute_cauchy_green_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(ftle::detect_lcs_ridges, m)?)?;

    // rqa
    m.add_function(wrap_pyfunction!(rqa::recurrence_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::recurrence_rate, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::determinism, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::laminarity, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::trapping_time, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::entropy_rqa, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::max_diagonal_line, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::divergence_rqa, m)?)?;
    m.add_function(wrap_pyfunction!(rqa::rqa_metrics, m)?)?;

    // saddle
    m.add_function(wrap_pyfunction!(saddle::estimate_jacobian_local, m)?)?;
    m.add_function(wrap_pyfunction!(saddle::detect_saddle_points, m)?)?;
    m.add_function(wrap_pyfunction!(saddle::compute_separatrix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(saddle::compute_basin_stability, m)?)?;

    Ok(())
}
