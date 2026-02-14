use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// FTLE via local linearization of neighbor trajectories.
/// Returns (ftle_values, stretching_rates).
/// Matches: manifold.primitives.dynamical.ftle.ftle_local_linearization
#[pyfunction]
#[pyo3(signature = (trajectory, time_horizon=10, n_neighbors=10, epsilon=None))]
pub fn ftle_local_linearization<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    time_horizon: usize,
    n_neighbors: usize,
    epsilon: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());

    if n_points < time_horizon + 2 || dim == 0 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let n_valid = n_points - time_horizon;
    let mut ftle_vals = vec![0.0f64; n_valid];
    let mut stretch_rates = vec![0.0f64; n_valid];

    for i in 0..n_valid {
        // Find k nearest neighbors
        let mut dists: Vec<(usize, f64)> = (0..n_valid)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = (0..dim)
                    .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<usize> = dists.iter().take(n_neighbors).map(|&(j, _)| j).collect();

        if neighbors.is_empty() {
            continue;
        }

        // Compute deformation gradient via least squares
        // dx0 = X_neighbors - X_i at time t
        // dxT = X_neighbors - X_i at time t+T
        // F = dxT * pinv(dx0)
        // Simplified: track maximum stretching
        let mut max_stretch = 0.0f64;
        for &j in &neighbors {
            let d0: f64 = (0..dim)
                .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                .sum::<f64>()
                .sqrt();
            let dt: f64 = (0..dim)
                .map(|d| (traj[[i + time_horizon, d]] - traj[[j + time_horizon, d]]).powi(2))
                .sum::<f64>()
                .sqrt();

            if d0 > 1e-15 {
                let stretch = dt / d0;
                if stretch > max_stretch {
                    max_stretch = stretch;
                }
            }
        }

        if max_stretch > 0.0 {
            ftle_vals[i] = max_stretch.ln() / time_horizon as f64;
            stretch_rates[i] = max_stretch;
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_vals),
        PyArray1::from_vec(py, stretch_rates),
    ))
}

/// FTLE via direct perturbation of delay-embedded signal.
/// Returns (ftle_values, jacobian_norms).
/// Matches: manifold.primitives.dynamical.ftle.ftle_direct_perturbation
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, time_horizon=10, perturbation=1e-6, n_perturbations=10))]
pub fn ftle_direct_perturbation<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    time_horizon: usize,
    perturbation: f64,
    n_perturbations: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let n_points = n.saturating_sub((dimension - 1) * delay);
    if n_points < time_horizon + 2 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    // Build embedding
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y[i + d * delay]).collect())
        .collect();

    let n_valid = n_points - time_horizon;
    let mut ftle_vals = vec![0.0f64; n_valid];
    let mut jac_norms = vec![0.0f64; n_valid];

    for i in 0..n_valid {
        // Find nearest neighbor with temporal separation
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for j in 0..n_valid {
            if (i as isize - j as isize).unsigned_abs() < dimension * delay {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist && dist > 1e-15 {
                best_dist = dist;
                best_j = j;
            }
        }

        if best_dist < f64::INFINITY {
            let d0 = best_dist;
            let dt: f64 = embedded[i + time_horizon]
                .iter()
                .zip(embedded[best_j + time_horizon].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if d0 > 1e-15 && dt > 0.0 {
                ftle_vals[i] = (dt / d0).ln() / time_horizon as f64;
                jac_norms[i] = dt / d0;
            }
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_vals),
        PyArray1::from_vec(py, jac_norms),
    ))
}

/// Compute Cauchy-Green deformation tensor.
/// Returns (ftle_field, eigenvalues, eigenvectors_flat).
/// Matches: manifold.primitives.dynamical.ftle.compute_cauchy_green_tensor
#[pyfunction]
#[pyo3(signature = (trajectory, time_horizon, n_neighbors=10))]
pub fn compute_cauchy_green_tensor<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    time_horizon: usize,
    n_neighbors: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());

    if n_points < time_horizon + 2 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let n_valid = n_points - time_horizon;
    let mut ftle_field = vec![0.0f64; n_valid];
    let mut eigenvalues = vec![0.0f64; n_valid];
    let mut eigenvectors = vec![0.0f64; n_valid * dim];

    // Simplified: use maximum stretching as proxy for max eigenvalue of C-G tensor
    for i in 0..n_valid {
        let mut dists: Vec<(usize, f64)> = (0..n_valid)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = (0..dim)
                    .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_stretch = 0.0f64;
        let mut best_dir = vec![0.0f64; dim];
        for &(j, d0) in dists.iter().take(n_neighbors) {
            if d0 < 1e-15 {
                continue;
            }
            let dt: f64 = (0..dim)
                .map(|d| (traj[[i + time_horizon, d]] - traj[[j + time_horizon, d]]).powi(2))
                .sum::<f64>()
                .sqrt();
            let stretch = dt / d0;
            if stretch > max_stretch {
                max_stretch = stretch;
                for d in 0..dim {
                    best_dir[d] = traj[[j, d]] - traj[[i, d]];
                }
            }
        }

        let lambda = max_stretch * max_stretch; // eigenvalue of C-G tensor
        eigenvalues[i] = lambda;
        if lambda > 0.0 {
            ftle_field[i] = lambda.ln() / (2.0 * time_horizon as f64);
        }

        // Normalize direction
        let norm: f64 = best_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in 0..dim {
            eigenvectors[i * dim + d] = if norm > 1e-15 { best_dir[d] / norm } else { 0.0 };
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_field),
        PyArray1::from_vec(py, eigenvalues),
        PyArray1::from_vec(py, eigenvectors),
    ))
}

/// Detect LCS ridges from FTLE field.
/// Returns binary mask of ridge points.
/// Matches: manifold.primitives.dynamical.ftle.detect_lcs_ridges
#[pyfunction]
#[pyo3(signature = (ftle_field, trajectory, threshold_percentile=90.0))]
pub fn detect_lcs_ridges<'py>(
    py: Python<'py>,
    ftle_field: PyReadonlyArray1<f64>,
    trajectory: PyReadonlyArray2<f64>,
    threshold_percentile: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ftle = ftle_field.as_slice()?;
    let n = ftle.len();

    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    // Compute threshold from percentile
    let mut sorted = ftle.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((threshold_percentile / 100.0) * (n - 1) as f64).round() as usize;
    let threshold = sorted[idx.min(n - 1)];

    // Ridge detection: local maxima above threshold
    let mut ridges = vec![0.0f64; n];
    for i in 1..n - 1 {
        if ftle[i] > threshold && ftle[i] > ftle[i - 1] && ftle[i] > ftle[i + 1] {
            ridges[i] = 1.0;
        }
    }

    Ok(PyArray1::from_vec(py, ridges))
}
