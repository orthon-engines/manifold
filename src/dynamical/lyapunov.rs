use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Rosenstein's method for largest Lyapunov exponent.
/// Returns (max_lyapunov, divergence_curve, time_axis).
/// Matches: manifold.primitives.dynamical.lyapunov.lyapunov_rosenstein
#[pyfunction]
#[pyo3(signature = (signal, dimension=None, delay=None, min_tsep=None, max_iter=None))]
pub fn lyapunov_rosenstein<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: Option<usize>,
    delay: Option<usize>,
    min_tsep: Option<usize>,
    max_iter: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let dim = dimension.unwrap_or(3);
    let tau = delay.unwrap_or(1);
    let tsep = min_tsep.unwrap_or(dim * tau);

    // Build delay embedding
    let n_points = n.saturating_sub((dim - 1) * tau);
    if n_points < tsep + 2 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dim).map(|d| y[i + d * tau]).collect())
        .collect();

    let max_it = max_iter.unwrap_or(n_points / 4).min(n_points - 1);

    // For each point, find nearest neighbor (with temporal separation)
    let mut nn_idx = vec![0usize; n_points];
    for i in 0..n_points {
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for j in 0..n_points {
            if (i as isize - j as isize).unsigned_abs() < tsep {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_j = j;
            }
        }
        nn_idx[i] = best_j;
    }

    // Track divergence
    let mut divergence = vec![0.0f64; max_it];
    let mut counts = vec![0usize; max_it];

    for i in 0..n_points {
        let j = nn_idx[i];
        for k in 0..max_it {
            let ii = i + k;
            let jj = j + k;
            if ii >= n_points || jj >= n_points {
                break;
            }
            let dist: f64 = embedded[ii]
                .iter()
                .zip(embedded[jj].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist > 0.0 {
                divergence[k] += dist.ln();
                counts[k] += 1;
            }
        }
    }

    // Average divergence curve
    let mut div_curve = Vec::with_capacity(max_it);
    let mut time_axis = Vec::with_capacity(max_it);
    for k in 0..max_it {
        if counts[k] > 0 {
            div_curve.push(divergence[k] / counts[k] as f64);
            time_axis.push(k as f64);
        }
    }

    // Estimate slope via linear regression on first 10% of curve
    let fit_len = (div_curve.len() / 10).max(2).min(div_curve.len());
    let max_lyap = if fit_len >= 2 {
        let n_pts = fit_len as f64;
        let sum_x: f64 = time_axis[..fit_len].iter().sum();
        let sum_y: f64 = div_curve[..fit_len].iter().sum();
        let sum_xy: f64 = time_axis[..fit_len]
            .iter()
            .zip(div_curve[..fit_len].iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_xx: f64 = time_axis[..fit_len].iter().map(|x| x * x).sum();
        let denom = n_pts * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-15 {
            (n_pts * sum_xy - sum_x * sum_y) / denom
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    Ok((
        max_lyap,
        PyArray1::from_vec(py, div_curve),
        PyArray1::from_vec(py, time_axis),
    ))
}

/// Kantz's method for largest Lyapunov exponent.
/// Returns (max_lyapunov, divergence_curve).
/// Matches: manifold.primitives.dynamical.lyapunov.lyapunov_kantz
#[pyfunction]
#[pyo3(signature = (signal, dimension=None, delay=None, min_tsep=None, epsilon=None, max_iter=None))]
pub fn lyapunov_kantz<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: Option<usize>,
    delay: Option<usize>,
    min_tsep: Option<usize>,
    epsilon: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let dim = dimension.unwrap_or(3);
    let tau = delay.unwrap_or(1);
    let tsep = min_tsep.unwrap_or(dim * tau);

    let n_points = n.saturating_sub((dim - 1) * tau);
    if n_points < tsep + 2 {
        return Ok((f64::NAN, PyArray1::from_vec(py, vec![])));
    }

    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dim).map(|d| y[i + d * tau]).collect())
        .collect();

    // Auto epsilon: median nearest-neighbor distance
    let eps = epsilon.unwrap_or_else(|| {
        let std_dev = {
            let mean_val = y.iter().sum::<f64>() / n as f64;
            (y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        std_dev * 0.1
    });

    let max_it = max_iter.unwrap_or(n_points / 4).min(n_points - 1);

    // For each point, find all neighbors within epsilon
    let mut divergence = vec![0.0f64; max_it];
    let mut counts = vec![0usize; max_it];

    for i in 0..n_points {
        for j in 0..n_points {
            if (i as isize - j as isize).unsigned_abs() < tsep {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < eps && dist > 0.0 {
                // Track divergence of this pair
                for k in 0..max_it {
                    let ii = i + k;
                    let jj = j + k;
                    if ii >= n_points || jj >= n_points {
                        break;
                    }
                    let d: f64 = embedded[ii]
                        .iter()
                        .zip(embedded[jj].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if d > 0.0 {
                        divergence[k] += d.ln();
                        counts[k] += 1;
                    }
                }
            }
        }
    }

    let mut div_curve = Vec::with_capacity(max_it);
    for k in 0..max_it {
        if counts[k] > 0 {
            div_curve.push(divergence[k] / counts[k] as f64);
        }
    }

    // Slope of divergence curve
    let fit_len = (div_curve.len() / 10).max(2).min(div_curve.len());
    let max_lyap = if fit_len >= 2 {
        let n_pts = fit_len as f64;
        let sum_x: f64 = (0..fit_len).map(|i| i as f64).sum();
        let sum_y: f64 = div_curve[..fit_len].iter().sum();
        let sum_xy: f64 = div_curve[..fit_len]
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * y)
            .sum();
        let sum_xx: f64 = (0..fit_len).map(|i| (i as f64).powi(2)).sum();
        let denom = n_pts * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-15 {
            (n_pts * sum_xy - sum_x * sum_y) / denom
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    Ok((max_lyap, PyArray1::from_vec(py, div_curve)))
}

/// Full Lyapunov spectrum via Jacobian method.
/// Matches: manifold.primitives.dynamical.lyapunov.lyapunov_spectrum
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, n_exponents=None))]
pub fn lyapunov_spectrum<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    n_exponents: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = signal.as_slice()?;
    let n = y.len();
    let n_exp = n_exponents.unwrap_or(dimension);

    let n_points = n.saturating_sub((dimension - 1) * delay);
    if n_points < dimension + 2 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n_exp]));
    }

    // Build embedding
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y[i + d * delay]).collect())
        .collect();

    // Estimate local Jacobians and accumulate QR decomposition
    let mut exponents = vec![0.0f64; n_exp.min(dimension)];
    let mut count = 0;

    // Initialize Q as identity
    let mut q: Vec<Vec<f64>> = (0..dimension)
        .map(|i| {
            let mut row = vec![0.0; dimension];
            row[i] = 1.0;
            row
        })
        .collect();

    for i in 0..n_points.saturating_sub(1) {
        // Estimate Jacobian from successive points
        if i + 1 >= n_points {
            break;
        }

        // Simple finite-difference Jacobian estimate
        let mut jac = vec![vec![0.0; dimension]; dimension];
        for d in 0..dimension {
            if i + 1 < n_points {
                for d2 in 0..dimension {
                    // Map from embedded[i] to embedded[i+1]
                    jac[d][d2] = if d == d2 {
                        if embedded[i][d2].abs() > 1e-15 {
                            embedded[i + 1][d] / embedded[i][d2]
                        } else {
                            1.0
                        }
                    } else {
                        0.0
                    };
                }
            }
        }

        // Multiply Q by Jacobian: Q_new = J * Q
        let mut new_q = vec![vec![0.0; dimension]; dimension];
        for row in 0..dimension {
            for col in 0..dimension {
                for k in 0..dimension {
                    new_q[row][col] += jac[row][k] * q[k][col];
                }
            }
        }

        // Gram-Schmidt QR decomposition
        let mut r_diag = vec![0.0f64; dimension];
        for j in 0..dimension {
            // Orthogonalize against previous vectors
            for k in 0..j {
                let dot: f64 = (0..dimension).map(|i| new_q[i][j] * q[i][k]).sum();
                for i in 0..dimension {
                    new_q[i][j] -= dot * q[i][k];
                }
            }
            // Normalize
            let norm: f64 = (0..dimension).map(|i| new_q[i][j].powi(2)).sum::<f64>().sqrt();
            r_diag[j] = norm;
            if norm > 1e-15 {
                for i in 0..dimension {
                    new_q[i][j] /= norm;
                }
            }
        }

        q = new_q;

        for j in 0..n_exp.min(dimension) {
            if r_diag[j] > 0.0 {
                exponents[j] += r_diag[j].ln();
            }
        }
        count += 1;
    }

    if count > 0 {
        for e in exponents.iter_mut() {
            *e /= count as f64;
        }
    }

    Ok(PyArray1::from_vec(py, exponents))
}
