use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Estimate local Jacobian at a point in trajectory.
/// Matches: manifold.primitives.dynamical.saddle.estimate_jacobian_local
#[pyfunction]
#[pyo3(signature = (trajectory, point_idx, n_neighbors=None))]
pub fn estimate_jacobian_local<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    point_idx: usize,
    n_neighbors: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());
    let k = n_neighbors.unwrap_or(dim * 2 + 1).min(n_points - 1);

    if point_idx >= n_points - 1 || dim == 0 {
        return Ok(PyArray2::from_vec2(py, &vec![vec![0.0; dim]; dim])?);
    }

    // Find k nearest neighbors
    let mut dists: Vec<(usize, f64)> = (0..n_points)
        .filter(|&j| j != point_idx && j + 1 < n_points)
        .map(|j| {
            let d: f64 = (0..dim)
                .map(|d| (traj[[point_idx, d]] - traj[[j, d]]).powi(2))
                .sum::<f64>()
                .sqrt();
            (j, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let neighbors: Vec<usize> = dists.iter().take(k).map(|&(j, _)| j).collect();

    // Least squares estimate of Jacobian: dx_next = J * dx_current
    // Simplified: diagonal Jacobian estimate
    let mut jac = vec![vec![0.0f64; dim]; dim];
    for d in 0..dim {
        let mut sum_dx0_sq = 0.0;
        let mut sum_dx0_dx1 = 0.0;
        for &j in &neighbors {
            let dx0 = traj[[j, d]] - traj[[point_idx, d]];
            let dx1 = traj[[j + 1, d]] - traj[[point_idx + 1, d]];
            sum_dx0_sq += dx0 * dx0;
            sum_dx0_dx1 += dx0 * dx1;
        }
        jac[d][d] = if sum_dx0_sq > 1e-15 {
            sum_dx0_dx1 / sum_dx0_sq
        } else {
            1.0
        };
    }

    Ok(PyArray2::from_vec2(py, &jac)?)
}

/// Detect saddle-like points in trajectory.
/// Returns (indices, scores, metadata_list).
/// Matches: manifold.primitives.dynamical.saddle.detect_saddle_points
#[pyfunction]
#[pyo3(signature = (trajectory, velocity_threshold=0.1, n_neighbors=None))]
pub fn detect_saddle_points<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    velocity_threshold: f64,
    n_neighbors: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Py<PyAny>)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());

    if n_points < 3 || dim == 0 {
        let empty_list = pyo3::types::PyList::empty(py);
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
            empty_list.into(),
        ));
    }

    // Compute velocities
    let mut velocities = vec![0.0f64; n_points];
    for i in 1..n_points - 1 {
        let v: f64 = (0..dim)
            .map(|d| ((traj[[i + 1, d]] - traj[[i - 1, d]]) / 2.0).powi(2))
            .sum::<f64>()
            .sqrt();
        velocities[i] = v;
    }
    velocities[0] = velocities[1];
    velocities[n_points - 1] = velocities[n_points - 2];

    // Find low-velocity points (potential saddle candidates)
    let mut indices = Vec::new();
    let mut scores = Vec::new();

    for i in 1..n_points - 1 {
        if velocities[i] < velocity_threshold
            && velocities[i] <= velocities[i - 1]
            && velocities[i] <= velocities[i + 1]
        {
            indices.push(i as f64);
            scores.push(1.0 - velocities[i] / velocity_threshold);
        }
    }

    let metadata = pyo3::types::PyList::empty(py);

    Ok((
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, scores),
        metadata.into(),
    ))
}

/// Compute distance from trajectory to nearest separatrix.
/// Matches: manifold.primitives.dynamical.saddle.compute_separatrix_distance
#[pyfunction]
#[pyo3(signature = (trajectory, saddle_indices, stable_direction=None))]
pub fn compute_separatrix_distance<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    saddle_indices: PyReadonlyArray1<f64>,
    stable_direction: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());
    let saddle_idx = saddle_indices.as_slice()?;

    let mut distances = vec![f64::INFINITY; n_points];

    for i in 0..n_points {
        for &si in saddle_idx {
            let si = si as usize;
            if si >= n_points {
                continue;
            }
            let d: f64 = (0..dim)
                .map(|d| (traj[[i, d]] - traj[[si, d]]).powi(2))
                .sum::<f64>()
                .sqrt();
            if d < distances[i] {
                distances[i] = d;
            }
        }
    }

    Ok(PyArray1::from_vec(py, distances))
}

/// Compute basin stability around saddle points.
/// Matches: manifold.primitives.dynamical.saddle.compute_basin_stability
#[pyfunction]
#[pyo3(signature = (trajectory, saddle_score, window=50))]
pub fn compute_basin_stability<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    saddle_score: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let traj = trajectory.as_array();
    let n_points = traj.nrows();
    let scores = saddle_score.as_slice()?;

    let mut stability = vec![0.0f64; n_points];

    for i in 0..n_points {
        let start = if i >= window / 2 { i - window / 2 } else { 0 };
        let end = (i + window / 2).min(n_points);
        let window_scores = &scores[start..end.min(scores.len())];
        if !window_scores.is_empty() {
            let mean: f64 = window_scores.iter().sum::<f64>() / window_scores.len() as f64;
            stability[i] = 1.0 - mean; // high saddle score = low stability
        }
    }

    Ok(PyArray1::from_vec(py, stability))
}
