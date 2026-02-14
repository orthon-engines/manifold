use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Threshold a matrix to create an adjacency matrix.
/// Matches: manifold.primitives.network.structure.threshold_matrix
#[pyfunction]
#[pyo3(signature = (matrix, threshold=None, percentile=None, keep="above", binary=true))]
pub fn threshold_matrix<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    threshold: Option<f64>,
    percentile: Option<f64>,
    keep: &str,
    binary: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mat = matrix.as_array();
    let (n, m) = (mat.nrows(), mat.ncols());

    // Determine threshold
    let thresh = if let Some(t) = threshold {
        t
    } else if let Some(p) = percentile {
        let mut values: Vec<f64> = mat.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (values.len() - 1) as f64) as usize;
        values[idx.min(values.len() - 1)]
    } else {
        0.0
    };

    let mut result = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            let val = mat[[i, j]];
            let keep_it = match keep {
                "below" => val < thresh,
                _ => val > thresh, // "above"
            };
            if keep_it {
                result[i][j] = if binary { 1.0 } else { val };
            }
        }
    }

    Ok(PyArray2::from_vec2(py, &result)?)
}

/// Network density.
/// Matches: manifold.primitives.network.structure.network_density
#[pyfunction]
#[pyo3(signature = (adjacency, directed=false))]
pub fn network_density(adjacency: PyReadonlyArray2<f64>, directed: bool) -> PyResult<f64> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 2 {
        return Ok(0.0);
    }

    let mut edges = 0;
    for i in 0..n {
        for j in 0..n {
            if i != j && adj[[i, j]].abs() > 1e-15 {
                edges += 1;
            }
        }
    }

    let max_edges = if directed {
        n * (n - 1)
    } else {
        edges /= 2; // counted each edge twice
        n * (n - 1) / 2
    };

    Ok(edges as f64 / max_edges as f64)
}

/// Clustering coefficient (average or per-node).
/// Matches: manifold.primitives.network.structure.clustering_coefficient
#[pyfunction]
#[pyo3(signature = (adjacency, node=None, weighted=false))]
pub fn clustering_coefficient(
    adjacency: PyReadonlyArray2<f64>,
    node: Option<usize>,
    weighted: bool,
) -> PyResult<f64> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 3 {
        return Ok(0.0);
    }

    let cc_for_node = |v: usize| -> f64 {
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| j != v && adj[[v, j]].abs() > 1e-15)
            .collect();
        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }

        let mut triangles = 0.0;
        for i in 0..k {
            for j in i + 1..k {
                if adj[[neighbors[i], neighbors[j]]].abs() > 1e-15 {
                    triangles += 1.0;
                }
            }
        }

        2.0 * triangles / (k * (k - 1)) as f64
    };

    match node {
        Some(v) => Ok(cc_for_node(v)),
        None => {
            let total: f64 = (0..n).map(|v| cc_for_node(v)).sum();
            Ok(total / n as f64)
        }
    }
}

/// Degree assortativity coefficient.
/// Matches: manifold.primitives.network.structure.assortativity
#[pyfunction]
#[pyo3(signature = (adjacency, attribute=None))]
pub fn assortativity(
    adjacency: PyReadonlyArray2<f64>,
    attribute: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 3 {
        return Ok(0.0);
    }

    // Compute degrees
    let degrees: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| adj[[i, j]].abs() > 1e-15)
                .count() as f64
        })
        .collect();

    let attr = match &attribute {
        Some(a) => a.as_slice()?.to_vec(),
        None => degrees.clone(),
    };

    // Pearson correlation of attribute values at edge endpoints
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut m = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            if adj[[i, j]].abs() > 1e-15 {
                sum_x += attr[i];
                sum_y += attr[j];
                sum_xy += attr[i] * attr[j];
                sum_x2 += attr[i] * attr[i];
                sum_y2 += attr[j] * attr[j];
                m += 1.0;
            }
        }
    }

    if m < 1.0 {
        return Ok(0.0);
    }

    let num = sum_xy / m - (sum_x / m) * (sum_y / m);
    let denom = ((sum_x2 / m - (sum_x / m).powi(2)) * (sum_y2 / m - (sum_y / m).powi(2))).sqrt();

    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok(num / denom)
}
