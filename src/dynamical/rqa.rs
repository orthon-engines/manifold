use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Build recurrence matrix.
/// Matches: manifold.primitives.dynamical.rqa.recurrence_matrix
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, threshold=None, threshold_percentile=10.0, metric="euclidean"))]
pub fn recurrence_matrix<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    threshold: Option<f64>,
    threshold_percentile: f64,
    metric: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = signal.as_slice()?;
    let n = y.len();
    let n_points = n.saturating_sub((dimension - 1) * delay);

    if n_points < 2 {
        return Ok(PyArray2::from_vec2(py, &vec![vec![0.0]])?);
    }

    // Delay embedding
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y[i + d * delay]).collect())
        .collect();

    // Compute distance matrix
    let mut distances = vec![vec![0.0f64; n_points]; n_points];
    for i in 0..n_points {
        for j in i + 1..n_points {
            let dist = match metric {
                "chebyshev" => embedded[i]
                    .iter()
                    .zip(embedded[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max),
                _ => embedded[i]
                    .iter()
                    .zip(embedded[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            };
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Determine threshold
    let eps = match threshold {
        Some(t) => t,
        None => {
            let mut all_dists: Vec<f64> = Vec::new();
            for i in 0..n_points {
                for j in i + 1..n_points {
                    all_dists.push(distances[i][j]);
                }
            }
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((threshold_percentile / 100.0) * (all_dists.len() - 1) as f64) as usize;
            all_dists[idx.min(all_dists.len() - 1)]
        }
    };

    // Build recurrence matrix
    let mut rm = vec![vec![0.0f64; n_points]; n_points];
    for i in 0..n_points {
        rm[i][i] = 1.0;
        for j in i + 1..n_points {
            if distances[i][j] <= eps {
                rm[i][j] = 1.0;
                rm[j][i] = 1.0;
            }
        }
    }

    Ok(PyArray2::from_vec2(py, &rm)?)
}

/// Recurrence rate.
#[pyfunction]
pub fn recurrence_rate(r: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let arr = r.as_array();
    let n = arr.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let sum: f64 = arr.iter().sum();
    Ok(sum / (n * n) as f64)
}

/// Determinism (ratio of recurrence points in diagonal lines).
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn determinism(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<f64> {
    let arr = r.as_array();
    let n = arr.nrows();
    if n < min_line {
        return Ok(0.0);
    }

    let mut total_recurrence = 0.0;
    let mut diagonal_recurrence = 0.0;

    // Count diagonal lines (excluding main diagonal)
    for offset in 1..n {
        let mut line_len = 0;
        for i in 0..n - offset {
            let j = i + offset;
            if arr[[i, j]] > 0.5 {
                line_len += 1;
                total_recurrence += 1.0;
            } else {
                if line_len >= min_line {
                    diagonal_recurrence += line_len as f64;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            diagonal_recurrence += line_len as f64;
        }
    }

    if total_recurrence <= 0.0 {
        return Ok(0.0);
    }
    Ok(diagonal_recurrence / total_recurrence)
}

/// Laminarity (ratio of recurrence points in vertical lines).
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn laminarity(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<f64> {
    let arr = r.as_array();
    let n = arr.nrows();
    if n < min_line {
        return Ok(0.0);
    }

    let mut total_recurrence = 0.0;
    let mut vertical_recurrence = 0.0;

    for j in 0..n {
        let mut line_len = 0;
        for i in 0..n {
            if arr[[i, j]] > 0.5 {
                line_len += 1;
                total_recurrence += 1.0;
            } else {
                if line_len >= min_line {
                    vertical_recurrence += line_len as f64;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            vertical_recurrence += line_len as f64;
        }
    }

    if total_recurrence <= 0.0 {
        return Ok(0.0);
    }
    Ok(vertical_recurrence / total_recurrence)
}

/// Trapping time (average length of vertical lines).
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn trapping_time(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<f64> {
    let arr = r.as_array();
    let n = arr.nrows();

    let mut total_len = 0;
    let mut count = 0;

    for j in 0..n {
        let mut line_len = 0;
        for i in 0..n {
            if arr[[i, j]] > 0.5 {
                line_len += 1;
            } else {
                if line_len >= min_line {
                    total_len += line_len;
                    count += 1;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            total_len += line_len;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(0.0);
    }
    Ok(total_len as f64 / count as f64)
}

/// Shannon entropy of diagonal line length distribution.
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn entropy_rqa(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<f64> {
    let arr = r.as_array();
    let n = arr.nrows();

    let mut line_counts: HashMap<usize, usize> = HashMap::new();

    for offset in 1..n {
        let mut line_len = 0;
        for i in 0..n - offset {
            let j = i + offset;
            if arr[[i, j]] > 0.5 {
                line_len += 1;
            } else {
                if line_len >= min_line {
                    *line_counts.entry(line_len).or_insert(0) += 1;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            *line_counts.entry(line_len).or_insert(0) += 1;
        }
    }

    let total: usize = line_counts.values().sum();
    if total == 0 {
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &count in line_counts.values() {
        let p = count as f64 / total as f64;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    Ok(entropy)
}

/// Maximum diagonal line length.
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn max_diagonal_line(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<usize> {
    let arr = r.as_array();
    let n = arr.nrows();

    let mut max_len = 0;

    for offset in 1..n {
        let mut line_len = 0;
        for i in 0..n - offset {
            let j = i + offset;
            if arr[[i, j]] > 0.5 {
                line_len += 1;
            } else {
                if line_len >= min_line && line_len > max_len {
                    max_len = line_len;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line && line_len > max_len {
            max_len = line_len;
        }
    }

    Ok(max_len)
}

/// Divergence (1 / max diagonal line).
#[pyfunction]
#[pyo3(signature = (r, min_line=2))]
pub fn divergence_rqa(r: PyReadonlyArray2<f64>, min_line: usize) -> PyResult<f64> {
    let max_l = max_diagonal_line(r, min_line)?;
    if max_l == 0 {
        return Ok(f64::INFINITY);
    }
    Ok(1.0 / max_l as f64)
}

/// Compute all RQA metrics at once.
/// Returns dict with rr, det, lam, tt, entr, l_max, div.
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, threshold=None, threshold_percentile=10.0, min_line=2, max_samples=20000))]
pub fn rqa_metrics<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    threshold: Option<f64>,
    threshold_percentile: f64,
    min_line: usize,
    max_samples: usize,
) -> PyResult<Py<PyAny>> {
    let y = signal.as_slice()?;

    // Subsample if needed
    let y_sub: Vec<f64> = if y.len() > max_samples {
        let step = y.len() / max_samples;
        y.iter().step_by(step).copied().take(max_samples).collect()
    } else {
        y.to_vec()
    };

    let n = y_sub.len();
    let n_points = n.saturating_sub((dimension - 1) * delay);

    if n_points < 2 {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("recurrence_rate", 0.0)?;
        dict.set_item("determinism", 0.0)?;
        dict.set_item("laminarity", 0.0)?;
        dict.set_item("trapping_time", 0.0)?;
        dict.set_item("entropy", 0.0)?;
        dict.set_item("max_diagonal_line", 0)?;
        dict.set_item("divergence", f64::INFINITY)?;
        return Ok(dict.into());
    }

    // Build embedding
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y_sub[i + d * delay]).collect())
        .collect();

    // Distance matrix
    let mut distances = vec![vec![0.0f64; n_points]; n_points];
    for i in 0..n_points {
        for j in i + 1..n_points {
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Threshold
    let eps = match threshold {
        Some(t) => t,
        None => {
            let mut all_dists: Vec<f64> = Vec::new();
            for i in 0..n_points {
                for j in i + 1..n_points {
                    all_dists.push(distances[i][j]);
                }
            }
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((threshold_percentile / 100.0) * (all_dists.len() - 1) as f64) as usize;
            all_dists[idx.min(all_dists.len() - 1)]
        }
    };

    // Build recurrence matrix as flat booleans for efficiency
    let mut rm = vec![false; n_points * n_points];
    let mut total_rec = 0.0f64;
    for i in 0..n_points {
        rm[i * n_points + i] = true;
        total_rec += 1.0;
        for j in i + 1..n_points {
            if distances[i][j] <= eps {
                rm[i * n_points + j] = true;
                rm[j * n_points + i] = true;
                total_rec += 2.0;
            }
        }
    }

    let rr = total_rec / (n_points * n_points) as f64;

    // Diagonal lines (excluding main diagonal)
    let mut diag_total_rec = 0.0;
    let mut diag_in_lines = 0.0;
    let mut diag_line_counts: HashMap<usize, usize> = HashMap::new();
    let mut max_diag = 0usize;

    for offset in 1..n_points {
        let mut line_len = 0;
        for i in 0..n_points - offset {
            let j = i + offset;
            if rm[i * n_points + j] {
                line_len += 1;
                diag_total_rec += 1.0;
            } else {
                if line_len >= min_line {
                    diag_in_lines += line_len as f64;
                    *diag_line_counts.entry(line_len).or_insert(0) += 1;
                    if line_len > max_diag {
                        max_diag = line_len;
                    }
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            diag_in_lines += line_len as f64;
            *diag_line_counts.entry(line_len).or_insert(0) += 1;
            if line_len > max_diag {
                max_diag = line_len;
            }
        }
    }

    let det = if diag_total_rec > 0.0 {
        diag_in_lines / diag_total_rec
    } else {
        0.0
    };

    // Entropy of diagonal line distribution
    let diag_total_lines: usize = diag_line_counts.values().sum();
    let entr = if diag_total_lines > 0 {
        let mut e = 0.0;
        for &c in diag_line_counts.values() {
            let p = c as f64 / diag_total_lines as f64;
            if p > 0.0 {
                e -= p * p.ln();
            }
        }
        e
    } else {
        0.0
    };

    let div = if max_diag > 0 {
        1.0 / max_diag as f64
    } else {
        f64::INFINITY
    };

    // Vertical lines
    let mut vert_total_rec = 0.0;
    let mut vert_in_lines = 0.0;
    let mut vert_total_len = 0usize;
    let mut vert_count = 0usize;

    for j in 0..n_points {
        let mut line_len = 0;
        for i in 0..n_points {
            if rm[i * n_points + j] {
                line_len += 1;
                vert_total_rec += 1.0;
            } else {
                if line_len >= min_line {
                    vert_in_lines += line_len as f64;
                    vert_total_len += line_len;
                    vert_count += 1;
                }
                line_len = 0;
            }
        }
        if line_len >= min_line {
            vert_in_lines += line_len as f64;
            vert_total_len += line_len;
            vert_count += 1;
        }
    }

    let lam = if vert_total_rec > 0.0 {
        vert_in_lines / vert_total_rec
    } else {
        0.0
    };

    let tt = if vert_count > 0 {
        vert_total_len as f64 / vert_count as f64
    } else {
        0.0
    };

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("recurrence_rate", rr)?;
    dict.set_item("determinism", det)?;
    dict.set_item("laminarity", lam)?;
    dict.set_item("trapping_time", tt)?;
    dict.set_item("entropy", entr)?;
    dict.set_item("max_diagonal_line", max_diag)?;
    dict.set_item("divergence", div)?;

    Ok(dict.into())
}
