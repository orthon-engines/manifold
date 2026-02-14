use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Mutual information between two signals.
/// Matches: manifold.primitives.information.mutual.mutual_information
#[pyfunction]
#[pyo3(signature = (x, y, bins=None, normalized=false, base=2.0))]
pub fn mutual_information(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    bins: Option<usize>,
    normalized: bool,
    base: f64,
) -> PyResult<f64> {
    let a = x.as_slice()?;
    let b = y.as_slice()?;
    let n = a.len().min(b.len());

    if n < 2 {
        return Ok(f64::NAN);
    }

    let n_bins = bins.unwrap_or(((n as f64).sqrt().ceil() as usize).max(4));

    let (min_a, max_a) = min_max(a);
    let (min_b, max_b) = min_max(b);
    let range_a = (max_a - min_a).max(1e-15);
    let range_b = (max_b - min_b).max(1e-15);

    let bin_a = |v: f64| -> usize { ((v - min_a) / range_a * n_bins as f64).min((n_bins - 1) as f64) as usize };
    let bin_b = |v: f64| -> usize { ((v - min_b) / range_b * n_bins as f64).min((n_bins - 1) as f64) as usize };

    let mut joint = vec![vec![0usize; n_bins]; n_bins];
    let mut margin_a = vec![0usize; n_bins];
    let mut margin_b = vec![0usize; n_bins];

    for i in 0..n {
        let ba = bin_a(a[i]);
        let bb = bin_b(b[i]);
        joint[ba][bb] += 1;
        margin_a[ba] += 1;
        margin_b[bb] += 1;
    }

    let total = n as f64;
    let log_base = base.ln();

    let mut mi = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            if joint[i][j] > 0 {
                let pij = joint[i][j] as f64 / total;
                let pi = margin_a[i] as f64 / total;
                let pj = margin_b[j] as f64 / total;
                mi += pij * (pij / (pi * pj)).ln() / log_base;
            }
        }
    }

    if normalized {
        // Normalized MI: MI / sqrt(H(X) * H(Y))
        let mut h_a = 0.0;
        let mut h_b = 0.0;
        for i in 0..n_bins {
            if margin_a[i] > 0 {
                let p = margin_a[i] as f64 / total;
                h_a -= p * p.ln() / log_base;
            }
            if margin_b[i] > 0 {
                let p = margin_b[i] as f64 / total;
                h_b -= p * p.ln() / log_base;
            }
        }
        let denom = (h_a * h_b).sqrt();
        if denom > 1e-15 {
            mi /= denom;
        }
    }

    Ok(mi.max(0.0))
}

/// Conditional mutual information I(X;Y|Z).
/// Matches: manifold.primitives.information.mutual.conditional_mutual_information
#[pyfunction]
#[pyo3(signature = (x, y, z, bins=None, base=2.0))]
pub fn conditional_mutual_information(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    bins: Option<usize>,
    base: f64,
) -> PyResult<f64> {
    let a = x.as_slice()?;
    let b = y.as_slice()?;
    let c = z.as_slice()?;
    let n = a.len().min(b.len()).min(c.len());

    if n < 4 {
        return Ok(f64::NAN);
    }

    let n_bins = bins.unwrap_or(((n as f64).cbrt().ceil() as usize).max(3));

    let (min_a, max_a) = min_max(a);
    let (min_b, max_b) = min_max(b);
    let (min_c, max_c) = min_max(c);
    let range_a = (max_a - min_a).max(1e-15);
    let range_b = (max_b - min_b).max(1e-15);
    let range_c = (max_c - min_c).max(1e-15);

    let bin_fn = |v: f64, min: f64, range: f64| -> usize {
        ((v - min) / range * n_bins as f64).min((n_bins - 1) as f64) as usize
    };

    // 3D histogram
    let mut xyz = vec![vec![vec![0usize; n_bins]; n_bins]; n_bins];
    let mut xz = vec![vec![0usize; n_bins]; n_bins];
    let mut yz = vec![vec![0usize; n_bins]; n_bins];
    let mut z_margin = vec![0usize; n_bins];

    for i in 0..n {
        let ba = bin_fn(a[i], min_a, range_a);
        let bb = bin_fn(b[i], min_b, range_b);
        let bc = bin_fn(c[i], min_c, range_c);
        xyz[ba][bb][bc] += 1;
        xz[ba][bc] += 1;
        yz[bb][bc] += 1;
        z_margin[bc] += 1;
    }

    let total = n as f64;
    let log_base = base.ln();

    // CMI = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    let mut h_xz = 0.0;
    let mut h_yz = 0.0;
    let mut h_xyz = 0.0;
    let mut h_z = 0.0;

    for i in 0..n_bins {
        for j in 0..n_bins {
            if xz[i][j] > 0 {
                let p = xz[i][j] as f64 / total;
                h_xz -= p * p.ln() / log_base;
            }
            if yz[i][j] > 0 {
                let p = yz[i][j] as f64 / total;
                h_yz -= p * p.ln() / log_base;
            }
            for k in 0..n_bins {
                if xyz[i][j][k] > 0 {
                    let p = xyz[i][j][k] as f64 / total;
                    h_xyz -= p * p.ln() / log_base;
                }
            }
        }
        if z_margin[i] > 0 {
            let p = z_margin[i] as f64 / total;
            h_z -= p * p.ln() / log_base;
        }
    }

    Ok((h_xz + h_yz - h_xyz - h_z).max(0.0))
}

fn min_max(data: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in data {
        if v < min { min = v; }
        if v > max { max = v; }
    }
    (min, max)
}
