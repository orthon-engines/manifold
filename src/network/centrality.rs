use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Degree centrality.
/// Matches: manifold.primitives.network.centrality.centrality_degree
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false, normalized=true, direction="both"))]
pub fn centrality_degree<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    weighted: bool,
    normalized: bool,
    direction: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    let mut degrees = vec![0.0f64; n];

    for i in 0..n {
        match direction {
            "in" => {
                for j in 0..n {
                    if weighted {
                        degrees[i] += adj[[j, i]].abs();
                    } else if adj[[j, i]].abs() > 1e-15 {
                        degrees[i] += 1.0;
                    }
                }
            }
            "out" => {
                for j in 0..n {
                    if weighted {
                        degrees[i] += adj[[i, j]].abs();
                    } else if adj[[i, j]].abs() > 1e-15 {
                        degrees[i] += 1.0;
                    }
                }
            }
            _ => {
                // "both"
                for j in 0..n {
                    if weighted {
                        degrees[i] += adj[[i, j]].abs();
                    } else if adj[[i, j]].abs() > 1e-15 {
                        degrees[i] += 1.0;
                    }
                }
            }
        }
    }

    if normalized && n > 1 {
        let max_deg = (n - 1) as f64;
        for d in degrees.iter_mut() {
            *d /= max_deg;
        }
    }

    Ok(PyArray1::from_vec(py, degrees))
}

/// Betweenness centrality.
/// Matches: manifold.primitives.network.centrality.centrality_betweenness
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false, normalized=true))]
pub fn centrality_betweenness<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    weighted: bool,
    normalized: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 3 {
        return Ok(PyArray1::from_vec(py, vec![0.0; n]));
    }

    let mut betweenness = vec![0.0f64; n];

    // Brandes' algorithm
    for s in 0..n {
        let mut stack = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n]; // number of shortest paths
        sigma[s] = 1.0;
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut delta = vec![0.0f64; n];

        // BFS / Dijkstra
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in 0..n {
                let edge = adj[[v, w]];
                if edge.abs() < 1e-15 {
                    continue;
                }
                let new_dist = dist[v] + if weighted { 1.0 / edge.abs() } else { 1.0 };
                if new_dist < dist[w] {
                    dist[w] = new_dist;
                    sigma[w] = sigma[v];
                    predecessors[w] = vec![v];
                    queue.push_back(w);
                } else if (new_dist - dist[w]).abs() < 1e-10 {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                let ratio = sigma[v] / sigma[w];
                delta[v] += ratio * (1.0 + delta[w]);
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    if normalized && n > 2 {
        let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
        for b in betweenness.iter_mut() {
            *b *= scale;
        }
    }

    Ok(PyArray1::from_vec(py, betweenness))
}

/// Eigenvector centrality via power iteration.
/// Matches: manifold.primitives.network.centrality.centrality_eigenvector
#[pyfunction]
#[pyo3(signature = (adjacency, max_iter=100, tolerance=1e-6))]
pub fn centrality_eigenvector<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    max_iter: usize,
    tolerance: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    let mut x = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..max_iter {
        let mut new_x = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                new_x[i] += adj[[i, j]] * x[j];
            }
        }

        // Normalize
        let norm: f64 = new_x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return Ok(PyArray1::from_vec(py, vec![0.0; n]));
        }
        for v in new_x.iter_mut() {
            *v /= norm;
        }

        // Check convergence
        let diff: f64 = x.iter().zip(new_x.iter()).map(|(a, b)| (a - b).abs()).sum();
        x = new_x;
        if diff < tolerance {
            break;
        }
    }

    Ok(PyArray1::from_vec(py, x))
}

/// Closeness centrality.
/// Matches: manifold.primitives.network.centrality.centrality_closeness
#[pyfunction]
#[pyo3(signature = (adjacency, weighted=false, normalized=true))]
pub fn centrality_closeness<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    weighted: bool,
    normalized: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let adj = adjacency.as_array();
    let n = adj.nrows();

    if n < 2 {
        return Ok(PyArray1::from_vec(py, vec![0.0; n]));
    }

    let mut closeness = vec![0.0f64; n];

    for i in 0..n {
        // BFS/Dijkstra from node i
        let mut dist = vec![f64::INFINITY; n];
        dist[i] = 0.0;
        let mut visited = vec![false; n];

        for _ in 0..n {
            // Find nearest unvisited
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

        let total_dist: f64 = dist.iter().filter(|d| d.is_finite() && **d > 0.0).sum();
        let reachable = dist.iter().filter(|d| d.is_finite() && **d > 0.0).count();

        if total_dist > 0.0 && reachable > 0 {
            closeness[i] = reachable as f64 / total_dist;
            if normalized && n > 1 {
                closeness[i] *= reachable as f64 / (n - 1) as f64;
            }
        }
    }

    Ok(PyArray1::from_vec(py, closeness))
}
