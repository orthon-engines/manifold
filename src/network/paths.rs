use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// All-pairs shortest paths (Dijkstra or BFS).
/// Matches: manifold.primitives.network.paths.shortest_paths
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false, source=None))]
pub fn shortest_paths<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    weighted: bool,
    source: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n == 0 {
        return Ok(PyArray2::from_vec2(py, &vec![vec![]])?);
    }

    let dijkstra = |s: usize| -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut visited = vec![false; n];

        for _ in 0..n {
            let mut min_d = f64::INFINITY;
            let mut u = 0;
            for j in 0..n {
                if !visited[j] && dist[j] < min_d {
                    min_d = dist[j];
                    u = j;
                }
            }
            if min_d.is_infinite() {
                break;
            }
            visited[u] = true;

            for v in 0..n {
                let edge = adj[[u, v]];
                if edge.abs() > 1e-15 {
                    let w = if weighted { 1.0 / edge.abs() } else { 1.0 };
                    let new_dist = dist[u] + w;
                    if new_dist < dist[v] {
                        dist[v] = new_dist;
                    }
                }
            }
        }
        dist
    };

    let result: Vec<Vec<f64>> = match source {
        Some(s) => vec![dijkstra(s)],
        None => (0..n).map(|s| dijkstra(s)).collect(),
    };

    Ok(PyArray2::from_vec2(py, &result)?)
}

/// Average path length.
/// Matches: manifold.primitives.network.paths.average_path_length
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false))]
pub fn average_path_length(adjacency: PyReadonlyArray2<f64>, weighted: bool) -> PyResult<f64> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 2 {
        return Ok(0.0);
    }

    let mut total = 0.0;
    let mut count = 0;

    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut visited = vec![false; n];

        for _ in 0..n {
            let mut min_d = f64::INFINITY;
            let mut u = 0;
            for j in 0..n {
                if !visited[j] && dist[j] < min_d {
                    min_d = dist[j];
                    u = j;
                }
            }
            if min_d.is_infinite() {
                break;
            }
            visited[u] = true;

            for v in 0..n {
                let edge = adj[[u, v]];
                if edge.abs() > 1e-15 {
                    let w = if weighted { 1.0 / edge.abs() } else { 1.0 };
                    let new_dist = dist[u] + w;
                    if new_dist < dist[v] {
                        dist[v] = new_dist;
                    }
                }
            }
        }

        for j in 0..n {
            if j != s && dist[j].is_finite() {
                total += dist[j];
                count += 1;
            }
        }
    }

    if count == 0 {
        return Ok(0.0);
    }

    Ok(total / count as f64)
}

/// Graph diameter (longest shortest path).
/// Matches: manifold.primitives.network.paths.diameter
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false))]
pub fn diameter(adjacency: PyReadonlyArray2<f64>, weighted: bool) -> PyResult<f64> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 2 {
        return Ok(0.0);
    }

    let mut max_dist = 0.0f64;

    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut visited = vec![false; n];

        for _ in 0..n {
            let mut min_d = f64::INFINITY;
            let mut u = 0;
            for j in 0..n {
                if !visited[j] && dist[j] < min_d {
                    min_d = dist[j];
                    u = j;
                }
            }
            if min_d.is_infinite() {
                break;
            }
            visited[u] = true;

            for v in 0..n {
                let edge = adj[[u, v]];
                if edge.abs() > 1e-15 {
                    let w = if weighted { 1.0 / edge.abs() } else { 1.0 };
                    let new_dist = dist[u] + w;
                    if new_dist < dist[v] {
                        dist[v] = new_dist;
                    }
                }
            }
        }

        for j in 0..n {
            if dist[j].is_finite() && dist[j] > max_dist {
                max_dist = dist[j];
            }
        }
    }

    Ok(max_dist)
}
