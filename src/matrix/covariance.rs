use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Covariance matrix.
/// Matches: manifold.primitives.matrix.covariance.covariance_matrix
#[pyfunction]
#[pyo3(signature = (signals, ddof=1, rowvar=false))]
pub fn covariance_matrix<'py>(
    py: Python<'py>,
    signals: PyReadonlyArray2<f64>,
    ddof: usize,
    rowvar: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = signals.as_array();
    let (rows, cols) = (arr.nrows(), arr.ncols());

    // If rowvar, each row is a variable; else each column is a variable
    let (n_vars, n_obs) = if rowvar { (rows, cols) } else { (cols, rows) };

    if n_obs <= ddof {
        return Ok(PyArray2::from_vec2(py, &vec![vec![f64::NAN; n_vars]; n_vars])?);
    }

    // Compute means
    let mut means = vec![0.0f64; n_vars];
    for v in 0..n_vars {
        for o in 0..n_obs {
            let val = if rowvar { arr[[v, o]] } else { arr[[o, v]] };
            means[v] += val;
        }
        means[v] /= n_obs as f64;
    }

    // Compute covariance
    let mut cov = vec![vec![0.0f64; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in i..n_vars {
            let mut sum = 0.0;
            for o in 0..n_obs {
                let vi = if rowvar { arr[[i, o]] } else { arr[[o, i]] };
                let vj = if rowvar { arr[[j, o]] } else { arr[[o, j]] };
                sum += (vi - means[i]) * (vj - means[j]);
            }
            cov[i][j] = sum / (n_obs - ddof) as f64;
            cov[j][i] = cov[i][j];
        }
    }

    Ok(PyArray2::from_vec2(py, &cov)?)
}

/// Correlation matrix.
/// Matches: manifold.primitives.matrix.covariance.correlation_matrix
#[pyfunction]
#[pyo3(signature = (signals, rowvar=false))]
pub fn correlation_matrix<'py>(
    py: Python<'py>,
    signals: PyReadonlyArray2<f64>,
    rowvar: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = signals.as_array();
    let (rows, cols) = (arr.nrows(), arr.ncols());
    let (n_vars, n_obs) = if rowvar { (rows, cols) } else { (cols, rows) };

    if n_obs < 2 {
        return Ok(PyArray2::from_vec2(py, &vec![vec![f64::NAN; n_vars]; n_vars])?);
    }

    // Means
    let mut means = vec![0.0f64; n_vars];
    for v in 0..n_vars {
        for o in 0..n_obs {
            let val = if rowvar { arr[[v, o]] } else { arr[[o, v]] };
            means[v] += val;
        }
        means[v] /= n_obs as f64;
    }

    // Standard deviations
    let mut stds = vec![0.0f64; n_vars];
    for v in 0..n_vars {
        for o in 0..n_obs {
            let val = if rowvar { arr[[v, o]] } else { arr[[o, v]] };
            stds[v] += (val - means[v]).powi(2);
        }
        stds[v] = (stds[v] / n_obs as f64).sqrt();
    }

    // Correlation matrix
    let mut corr = vec![vec![0.0f64; n_vars]; n_vars];
    for i in 0..n_vars {
        corr[i][i] = 1.0;
        for j in i + 1..n_vars {
            if stds[i] < 1e-15 || stds[j] < 1e-15 {
                corr[i][j] = 0.0;
                corr[j][i] = 0.0;
                continue;
            }
            let mut sum = 0.0;
            for o in 0..n_obs {
                let vi = if rowvar { arr[[i, o]] } else { arr[[o, i]] };
                let vj = if rowvar { arr[[j, o]] } else { arr[[o, j]] };
                sum += (vi - means[i]) * (vj - means[j]);
            }
            let r = sum / (n_obs as f64 * stds[i] * stds[j]);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    Ok(PyArray2::from_vec2(py, &corr)?)
}
