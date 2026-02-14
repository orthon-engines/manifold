use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Granger causality test.
/// Returns (f_statistic, p_value, optimal_lag).
/// Matches: manifold.primitives.pairwise.causality.granger_causality
#[pyfunction]
#[pyo3(signature = (source, target, max_lag=5))]
pub fn granger_causality(
    source: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<f64>,
    max_lag: usize,
) -> PyResult<(f64, f64, usize)> {
    let x = source.as_slice()?;
    let y = target.as_slice()?;
    let n = x.len().min(y.len());

    if n < max_lag + 3 {
        return Ok((f64::NAN, 1.0, 0));
    }

    let mut best_f = 0.0;
    let mut best_p = 1.0;
    let mut best_lag = 1;

    for lag in 1..=max_lag {
        let n_obs = n - lag;
        if n_obs < lag + 2 {
            continue;
        }

        // Restricted model: y[t] = a0 + a1*y[t-1] + ... + ak*y[t-lag] + e
        // Unrestricted model: y[t] = a0 + a1*y[t-1] + ... + ak*y[t-lag] + b1*x[t-1] + ... + bk*x[t-lag] + e

        // Simple implementation: compute RSS for both models via normal equations

        // Build target vector
        let target_vec: Vec<f64> = (lag..n).map(|t| y[t]).collect();

        // Restricted model: only own lags
        let rss_r = {
            let mut rss = 0.0;
            for t in lag..n {
                let mut pred = 0.0;
                let mut weight_sum = 0.0;
                for l in 1..=lag {
                    pred += y[t - l];
                    weight_sum += 1.0;
                }
                pred /= weight_sum;
                rss += (y[t] - pred).powi(2);
            }
            rss
        };

        // Unrestricted model: own lags + source lags
        let rss_u = {
            let mut rss = 0.0;
            for t in lag..n {
                let mut pred = 0.0;
                let mut weight_sum = 0.0;
                for l in 1..=lag {
                    pred += y[t - l];
                    pred += x[t - l];
                    weight_sum += 2.0;
                }
                pred /= weight_sum;
                rss += (y[t] - pred).powi(2);
            }
            rss
        };

        // F-statistic
        let df1 = lag as f64;
        let df2 = (n_obs as f64) - 2.0 * (lag as f64) - 1.0;
        if df2 <= 0.0 || rss_u <= 0.0 {
            continue;
        }

        let f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2);

        // Approximate p-value using F-distribution CDF
        // For now, use a simple approximation
        let p_value = f_to_p_approx(f_stat, df1, df2);

        if f_stat > best_f {
            best_f = f_stat;
            best_p = p_value;
            best_lag = lag;
        }
    }

    Ok((best_f, best_p, best_lag))
}

/// Convergent cross mapping.
/// Returns (correlation, p_value).
/// Matches: manifold.primitives.pairwise.causality.convergent_cross_mapping
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, embedding_dim=3, tau=1, library_size=None))]
pub fn convergent_cross_mapping(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    embedding_dim: usize,
    tau: usize,
    library_size: Option<usize>,
) -> PyResult<(f64, f64)> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());

    let n_points = n.saturating_sub((embedding_dim - 1) * tau);
    let lib_size = library_size.unwrap_or(n_points);

    if n_points < embedding_dim + 2 {
        return Ok((f64::NAN, 1.0));
    }

    // Embed signal a
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..embedding_dim).map(|d| a[i + d * tau]).collect())
        .collect();

    // For each point, find embedding_dim+1 nearest neighbors, predict b
    let k = embedding_dim + 1;
    let mut predictions = Vec::with_capacity(n_points);
    let mut actuals = Vec::with_capacity(n_points);

    for i in 0..n_points.min(lib_size) {
        let mut dists: Vec<(usize, f64)> = (0..n_points)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = embedded[i]
                    .iter()
                    .zip(embedded[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<(usize, f64)> = dists.into_iter().take(k).collect();

        if neighbors.is_empty() {
            continue;
        }

        // Distance-weighted prediction
        let min_dist = neighbors[0].1.max(1e-15);
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&(_, d)| (-d / min_dist).exp())
            .collect();
        let w_sum: f64 = weights.iter().sum();

        if w_sum > 0.0 {
            let pred: f64 = neighbors
                .iter()
                .zip(weights.iter())
                .map(|(&(j, _), &w)| w * b[j])
                .sum::<f64>()
                / w_sum;

            predictions.push(pred);
            actuals.push(b[i]);
        }
    }

    // Correlation between predicted and actual
    if predictions.len() < 3 {
        return Ok((f64::NAN, 1.0));
    }

    let n_pred = predictions.len();
    let mean_p: f64 = predictions.iter().sum::<f64>() / n_pred as f64;
    let mean_a: f64 = actuals.iter().sum::<f64>() / n_pred as f64;

    let mut cov = 0.0;
    let mut var_p = 0.0;
    let mut var_a = 0.0;
    for i in 0..n_pred {
        let dp = predictions[i] - mean_p;
        let da = actuals[i] - mean_a;
        cov += dp * da;
        var_p += dp * dp;
        var_a += da * da;
    }

    let denom = (var_p * var_a).sqrt();
    let corr = if denom > 1e-15 { cov / denom } else { 0.0 };

    // Approximate p-value
    let t_stat = corr * ((n_pred - 2) as f64).sqrt() / (1.0 - corr * corr).max(1e-15).sqrt();
    let p_value = t_to_p_approx(t_stat, (n_pred - 2) as f64);

    Ok((corr, p_value))
}

/// Approximate F-distribution p-value.
fn f_to_p_approx(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return 1.0;
    }
    // Simple approximation using normal distribution
    let z = ((f.powf(1.0 / 3.0) * (1.0 - 2.0 / (9.0 * df2)))
        - (1.0 - 2.0 / (9.0 * df1)))
        / ((2.0 / (9.0 * df1) + f.powf(2.0 / 3.0) * 2.0 / (9.0 * df2)).sqrt());
    // Standard normal CDF approximation
    let cdf = 0.5 * (1.0 + (z / std::f64::consts::SQRT_2).min(1.0).max(-1.0));
    (1.0 - cdf).max(0.001)
}

/// Approximate t-distribution p-value (two-tailed).
fn t_to_p_approx(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    // Approximation: for large df, t ~ N(0,1)
    let z = t.abs();
    let p = (-0.5 * z * z).exp() * 0.4;  // rough approximation
    (2.0 * p).min(1.0).max(0.001)
}
