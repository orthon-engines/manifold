use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Hurst exponent via R/S analysis.
/// Line-by-line port of manifold/primitives/individual/fractal.py::hurst_exponent(method='rs')
///
/// Config constants inlined from manifold.primitives.config:
///   rs_min_k = 10
///   rs_max_k_ratio = 0.5
///   rs_max_k_cap = 100
///   min_samples.hurst = 20 (but fractal.py checks rs_min_k=10)
#[pyfunction]
#[pyo3(signature = (signal, method="rs"))]
pub fn hurst_exponent(signal: PyReadonlyArray1<f64>, method: &str) -> PyResult<f64> {
    let raw = signal.as_slice()?;

    // flatten and strip NaN (matching np.asarray(signal).flatten(); signal[~np.isnan(signal)])
    let y: Vec<f64> = raw.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = y.len();

    // cfg.fractal.rs_min_k = 10
    const RS_MIN_K: usize = 10;

    if n < RS_MIN_K {
        return Ok(f64::NAN);
    }

    if method == "dfa" {
        // DFA not yet ported — return NaN so Python fallback handles it
        return Ok(f64::NAN);
    }

    // cfg.fractal.rs_max_k_ratio = 0.5, rs_max_k_cap = 100
    let max_k = ((n as f64 * 0.5) as usize).min(100);

    let mut k_values: Vec<f64> = Vec::new();
    let mut rs_values: Vec<f64> = Vec::new();

    for k in RS_MIN_K..max_k {
        let n_subseries = n / k;
        let mut rs_sum = 0.0f64;

        for i in 0..n_subseries {
            let subseries = &y[i * k..(i + 1) * k];

            // mean
            let mean: f64 = subseries.iter().sum::<f64>() / k as f64;

            // Cumulative deviation from mean
            let mut cum_dev = Vec::with_capacity(k);
            let mut running = 0.0;
            for &val in subseries {
                running += val - mean;
                cum_dev.push(running);
            }

            // Range
            let r = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            // Standard deviation with ddof=1 (matching np.std(subseries, ddof=1))
            let variance: f64 =
                subseries.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1) as f64;
            let s = variance.sqrt();

            if s > 0.0 {
                rs_sum += r / s;
            }
        }

        if n_subseries > 0 {
            let rs_avg = rs_sum / n_subseries as f64;
            if rs_avg > 0.0 {
                k_values.push((k as f64).ln());
                rs_values.push(rs_avg.ln());
            }
        }
    }

    // Need at least 3 points for a meaningful fit (matching len(k_values) < 3 check)
    if k_values.len() < 3 {
        return Ok(f64::NAN);
    }

    // Linear fit: log(R/S) = H * log(k) + c  (matching np.polyfit degree 1)
    let n_pts = k_values.len() as f64;
    let sum_x: f64 = k_values.iter().sum();
    let sum_y: f64 = rs_values.iter().sum();
    let sum_xy: f64 = k_values.iter().zip(rs_values.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = k_values.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return Ok(f64::NAN);
    }

    let h = (n_pts * sum_xy - sum_x * sum_y) / denom;

    // clip to [0, 1]  (matching np.clip(H, 0, 1))
    Ok(h.clamp(0.0, 1.0))
}
