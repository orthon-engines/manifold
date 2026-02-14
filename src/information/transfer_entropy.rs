use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Transfer entropy from X to Y.
/// Matches: manifold.primitives.information.transfer_entropy
#[pyfunction]
#[pyo3(signature = (x, y, lag=1, bins=None, base=2.0))]
pub fn transfer_entropy(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    lag: usize,
    bins: Option<usize>,
    base: f64,
) -> PyResult<f64> {
    let a = x.as_slice()?;
    let b = y.as_slice()?;
    let n = a.len().min(b.len());

    if n < lag + 2 {
        return Ok(f64::NAN);
    }

    let n_bins = bins.unwrap_or(((n as f64).cbrt().ceil() as usize).max(3));

    let min_a = a.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_a = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = b.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_b = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range_a = (max_a - min_a).max(1e-15);
    let range_b = (max_b - min_b).max(1e-15);

    let bin_a = |v: f64| -> usize { ((v - min_a) / range_a * n_bins as f64).min((n_bins - 1) as f64) as usize };
    let bin_b = |v: f64| -> usize { ((v - min_b) / range_b * n_bins as f64).min((n_bins - 1) as f64) as usize };

    // TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    // = sum p(y_f, y_p, x_p) * log( p(y_f | y_p, x_p) / p(y_f | y_p) )
    let n_obs = n - lag;

    // 3D histogram: (y_future, y_past, x_past)
    let mut joint_3 = vec![vec![vec![0usize; n_bins]; n_bins]; n_bins];
    let mut joint_yfy = vec![vec![0usize; n_bins]; n_bins];
    let mut joint_yx = vec![vec![0usize; n_bins]; n_bins];
    let mut margin_y = vec![0usize; n_bins];

    for t in lag..n {
        let yf = bin_b(b[t]);
        let yp = bin_b(b[t - lag]);
        let xp = bin_a(a[t - lag]);
        joint_3[yf][yp][xp] += 1;
        joint_yfy[yf][yp] += 1;
        joint_yx[yp][xp] += 1;
        margin_y[yp] += 1;
    }

    let total = n_obs as f64;
    let log_base = base.ln();

    let mut te = 0.0;
    for yf in 0..n_bins {
        for yp in 0..n_bins {
            for xp in 0..n_bins {
                if joint_3[yf][yp][xp] > 0 && joint_yx[yp][xp] > 0 && margin_y[yp] > 0 {
                    let p_yfypxp = joint_3[yf][yp][xp] as f64 / total;
                    let p_yf_given_ypxp =
                        joint_3[yf][yp][xp] as f64 / joint_yx[yp][xp] as f64;
                    let p_yf_given_yp =
                        joint_yfy[yf][yp] as f64 / margin_y[yp] as f64;

                    if p_yf_given_yp > 0.0 {
                        te += p_yfypxp * (p_yf_given_ypxp / p_yf_given_yp).ln() / log_base;
                    }
                }
            }
        }
    }

    Ok(te.max(0.0))
}
