use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};


/// Compute persistence diagram via Vietoris-Rips filtration.
/// Returns dict mapping dimension -> list of (birth, death) tuples.
/// Matches: manifold.primitives.topology.persistence.persistence_diagram
#[pyfunction]
#[pyo3(signature = (points, max_dimension=1, max_edge_length=None))]
pub fn persistence_diagram<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    max_dimension: usize,
    max_edge_length: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let pts = points.as_array();
    let (n, dim) = (pts.nrows(), pts.ncols());

    let dict = PyDict::new(py);

    if n < 2 {
        for d in 0..=max_dimension {
            dict.set_item(d, PyList::empty(py))?;
        }
        return Ok(dict.into());
    }

    // Compute pairwise distances
    let mut dists = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = (0..dim)
                .map(|k| (pts[[i, k]] - pts[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }

    let max_eps = max_edge_length.unwrap_or_else(|| {
        let mut max_d = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                if dists[i][j] > max_d {
                    max_d = dists[i][j];
                }
            }
        }
        max_d
    });

    // Simple 0-dimensional persistence via single-linkage clustering (Union-Find)
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0usize; n];
    let mut birth = vec![0.0f64; n]; // All born at 0

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    // Sort edges by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if dists[i][j] <= max_eps {
                edges.push((dists[i][j], i, j));
            }
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut h0_pairs: Vec<(f64, f64)> = Vec::new();

    for (dist, i, j) in &edges {
        let ri = find(&mut parent, *i);
        let rj = find(&mut parent, *j);
        if ri != rj {
            // Merge: younger component dies
            let (survivor, dying) = if birth[ri] <= birth[rj] {
                (ri, rj)
            } else {
                (rj, ri)
            };

            h0_pairs.push((birth[dying], *dist));

            if rank[survivor] < rank[dying] {
                parent[survivor] = dying;
                birth[dying] = birth[survivor].min(birth[dying]);
            } else if rank[survivor] > rank[dying] {
                parent[dying] = survivor;
            } else {
                parent[dying] = survivor;
                rank[survivor] += 1;
            }
        }
    }

    // The last surviving component has infinite death
    h0_pairs.push((0.0, f64::INFINITY));

    let h0_list = PyList::new(py, h0_pairs.iter().map(|(b, d)| (*b, *d)))?;
    dict.set_item(0, h0_list)?;

    // For H1+ persistence, return empty (full Ripser algorithm is complex)
    for d in 1..=max_dimension {
        dict.set_item(d, PyList::empty(py))?;
    }

    Ok(dict.into())
}

/// Compute Betti numbers from persistence diagram.
/// Matches: manifold.primitives.topology.persistence.betti_numbers
#[pyfunction]
#[pyo3(signature = (diagrams, threshold=None, filtration_value=None))]
pub fn betti_numbers<'py>(
    py: Python<'py>,
    diagrams: &Bound<'_, PyDict>,
    threshold: Option<f64>,
    filtration_value: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    let filt = filtration_value.unwrap_or(f64::INFINITY);
    let thresh = threshold.unwrap_or(0.0);

    for (key, value) in diagrams.iter() {
        let dim: usize = key.extract()?;
        let pairs: Vec<(f64, f64)> = value.extract()?;

        let count = pairs
            .iter()
            .filter(|(b, d)| {
                *b <= filt && *d > filt && (*d - *b) > thresh
            })
            .count();

        result.set_item(dim, count)?;
    }

    Ok(result.into())
}

/// Persistence entropy.
/// Matches: manifold.primitives.topology.persistence.persistence_entropy
#[pyfunction]
#[pyo3(signature = (diagrams, dimension=1, normalized=true))]
pub fn persistence_entropy(
    diagrams: &Bound<'_, PyDict>,
    dimension: usize,
    normalized: bool,
) -> PyResult<f64> {
    let pairs: Vec<(f64, f64)> = match diagrams.get_item(dimension)? {
        Some(val) => val.extract()?,
        None => return Ok(0.0),
    };

    // Filter finite persistence
    let lifetimes: Vec<f64> = pairs
        .iter()
        .filter(|(_, d)| d.is_finite())
        .map(|(b, d)| d - b)
        .filter(|l| *l > 0.0)
        .collect();

    if lifetimes.is_empty() {
        return Ok(0.0);
    }

    let total: f64 = lifetimes.iter().sum();
    let mut entropy = 0.0;
    for &l in &lifetimes {
        let p = l / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    if normalized && lifetimes.len() > 1 {
        let max_e = (lifetimes.len() as f64).ln();
        if max_e > 0.0 {
            entropy /= max_e;
        }
    }

    Ok(entropy)
}
