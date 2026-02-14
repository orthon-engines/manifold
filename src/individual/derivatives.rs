use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// First derivative via central differences.
/// Matches: manifold.primitives.individual.derivatives.first_derivative
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn first_derivative<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = values.as_slice()?;
    let n = y.len();
    if n < 2 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    let mut result = vec![0.0; n];

    // Forward difference at start
    result[0] = (y[1] - y[0]) / dt;

    // Central differences
    for i in 1..n - 1 {
        result[i] = (y[i + 1] - y[i - 1]) / (2.0 * dt);
    }

    // Backward difference at end
    result[n - 1] = (y[n - 1] - y[n - 2]) / dt;

    Ok(PyArray1::from_vec(py, result))
}

/// Second derivative via central differences.
/// Matches: manifold.primitives.individual.derivatives.second_derivative
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn second_derivative<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = values.as_slice()?;
    let n = y.len();
    if n < 3 {
        return Ok(PyArray1::from_vec(py, vec![0.0; n]));
    }

    let mut result = vec![0.0; n];
    let dt2 = dt * dt;

    for i in 1..n - 1 {
        result[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / dt2;
    }

    // Boundary: copy nearest interior value
    result[0] = result[1];
    result[n - 1] = result[n - 2];

    Ok(PyArray1::from_vec(py, result))
}

/// Gradient (same as numpy.gradient).
/// Matches: manifold.primitives.individual.derivatives.gradient
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // gradient is the same as first_derivative with central differences
    first_derivative(py, values, dt)
}

/// Finite difference of arbitrary order.
/// Matches: manifold.primitives.individual.derivatives.finite_difference
#[pyfunction]
#[pyo3(signature = (values, order=1, dt=1.0))]
pub fn finite_difference<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    order: usize,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = values.as_slice()?;
    let n = y.len();

    if order == 0 || n == 0 {
        return Ok(PyArray1::from_vec(py, y.to_vec()));
    }

    // Iterative forward difference
    let mut current = y.to_vec();
    for _ in 0..order {
        if current.len() < 2 {
            return Ok(PyArray1::from_vec(py, vec![0.0]));
        }
        let new_len = current.len() - 1;
        let mut next = vec![0.0; new_len];
        for i in 0..new_len {
            next[i] = (current[i + 1] - current[i]) / dt;
        }
        current = next;
    }

    Ok(PyArray1::from_vec(py, current))
}

/// Velocity (alias for first derivative).
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn velocity<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    first_derivative(py, values, dt)
}

/// Acceleration (alias for second derivative).
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn acceleration<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    second_derivative(py, values, dt)
}

/// Jerk (third derivative).
#[pyfunction]
#[pyo3(signature = (values, dt=1.0))]
pub fn jerk<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = values.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(PyArray1::from_vec(py, vec![0.0; n]));
    }

    // Compute second derivative first
    let dt2 = dt * dt;
    let mut d2 = vec![0.0; n];
    for i in 1..n - 1 {
        d2[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / dt2;
    }
    d2[0] = d2[1];
    d2[n - 1] = d2[n - 2];

    // Then first derivative of second derivative
    let mut result = vec![0.0; n];
    result[0] = (d2[1] - d2[0]) / dt;
    for i in 1..n - 1 {
        result[i] = (d2[i + 1] - d2[i - 1]) / (2.0 * dt);
    }
    result[n - 1] = (d2[n - 1] - d2[n - 2]) / dt;

    Ok(PyArray1::from_vec(py, result))
}
