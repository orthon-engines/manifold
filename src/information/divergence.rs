use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Cross entropy H(P, Q).
/// Matches: manifold.primitives.information.divergence.cross_entropy
#[pyfunction]
#[pyo3(signature = (p, q, bins=None, base=2.0))]
pub fn cross_entropy(
    p: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    bins: Option<usize>,
    base: f64,
) -> PyResult<f64> {
    let pa = p.as_slice()?;
    let qa = q.as_slice()?;

    let (p_dist, q_dist) = to_distributions(pa, qa, bins);
    let log_base = base.ln();

    let mut ce = 0.0;
    for i in 0..p_dist.len() {
        if p_dist[i] > 0.0 && q_dist[i] > 0.0 {
            ce -= p_dist[i] * q_dist[i].ln() / log_base;
        }
    }

    Ok(ce)
}

/// KL divergence D_KL(P || Q).
/// Matches: manifold.primitives.information.divergence.kl_divergence
#[pyfunction]
#[pyo3(signature = (p, q, bins=None, base=2.0))]
pub fn kl_divergence(
    p: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    bins: Option<usize>,
    base: f64,
) -> PyResult<f64> {
    let pa = p.as_slice()?;
    let qa = q.as_slice()?;

    let (p_dist, q_dist) = to_distributions(pa, qa, bins);
    let log_base = base.ln();

    let mut kl = 0.0;
    for i in 0..p_dist.len() {
        if p_dist[i] > 0.0 && q_dist[i] > 0.0 {
            kl += p_dist[i] * (p_dist[i] / q_dist[i]).ln() / log_base;
        }
    }

    Ok(kl.max(0.0))
}

/// Jensen-Shannon divergence.
/// Matches: manifold.primitives.information.divergence.js_divergence
#[pyfunction]
#[pyo3(signature = (p, q, bins=None, base=2.0))]
pub fn js_divergence(
    p: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    bins: Option<usize>,
    base: f64,
) -> PyResult<f64> {
    let pa = p.as_slice()?;
    let qa = q.as_slice()?;

    let (p_dist, q_dist) = to_distributions(pa, qa, bins);
    let log_base = base.ln();

    // M = 0.5 * (P + Q)
    let m_dist: Vec<f64> = p_dist.iter().zip(q_dist.iter()).map(|(a, b)| 0.5 * (a + b)).collect();

    let mut kl_pm = 0.0;
    let mut kl_qm = 0.0;
    for i in 0..p_dist.len() {
        if p_dist[i] > 0.0 && m_dist[i] > 0.0 {
            kl_pm += p_dist[i] * (p_dist[i] / m_dist[i]).ln() / log_base;
        }
        if q_dist[i] > 0.0 && m_dist[i] > 0.0 {
            kl_qm += q_dist[i] * (q_dist[i] / m_dist[i]).ln() / log_base;
        }
    }

    Ok((0.5 * (kl_pm + kl_qm)).max(0.0))
}

/// Hellinger distance.
/// Matches: manifold.primitives.information.divergence.hellinger_distance
#[pyfunction]
#[pyo3(signature = (p, q, bins=None))]
pub fn hellinger_distance(
    p: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    bins: Option<usize>,
) -> PyResult<f64> {
    let pa = p.as_slice()?;
    let qa = q.as_slice()?;

    let (p_dist, q_dist) = to_distributions(pa, qa, bins);

    let sum: f64 = p_dist
        .iter()
        .zip(q_dist.iter())
        .map(|(p, q)| (p.sqrt() - q.sqrt()).powi(2))
        .sum();

    Ok((sum / 2.0).sqrt())
}

/// Total variation distance.
/// Matches: manifold.primitives.information.divergence.total_variation_distance
#[pyfunction]
#[pyo3(signature = (p, q, bins=None))]
pub fn total_variation_distance(
    p: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    bins: Option<usize>,
) -> PyResult<f64> {
    let pa = p.as_slice()?;
    let qa = q.as_slice()?;

    let (p_dist, q_dist) = to_distributions(pa, qa, bins);

    let sum: f64 = p_dist
        .iter()
        .zip(q_dist.iter())
        .map(|(p, q)| (p - q).abs())
        .sum();

    Ok(0.5 * sum)
}

/// Convert raw data to probability distributions via histograms.
fn to_distributions(a: &[f64], b: &[f64], bins: Option<usize>) -> (Vec<f64>, Vec<f64>) {
    let n_bins = bins.unwrap_or(
        ((a.len().max(b.len()) as f64).sqrt().ceil() as usize).max(4),
    );

    let min_val = a
        .iter()
        .chain(b.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = a
        .iter()
        .chain(b.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-15);

    let bin_idx = |v: f64| -> usize {
        ((v - min_val) / range * n_bins as f64).min((n_bins - 1) as f64) as usize
    };

    let mut hist_a = vec![0.0f64; n_bins];
    let mut hist_b = vec![0.0f64; n_bins];

    for &v in a {
        hist_a[bin_idx(v)] += 1.0;
    }
    for &v in b {
        hist_b[bin_idx(v)] += 1.0;
    }

    let sum_a: f64 = hist_a.iter().sum();
    let sum_b: f64 = hist_b.iter().sum();

    // Add small epsilon to avoid division by zero
    for v in hist_a.iter_mut() {
        *v = (*v + 1e-10) / (sum_a + n_bins as f64 * 1e-10);
    }
    for v in hist_b.iter_mut() {
        *v = (*v + 1e-10) / (sum_b + n_bins as f64 * 1e-10);
    }

    (hist_a, hist_b)
}
