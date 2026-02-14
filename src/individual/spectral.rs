use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

/// FFT of a real signal. Returns magnitude spectrum.
/// Matches: manifold.primitives.individual.spectral.fft
#[pyfunction]
pub fn fft<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = signal.as_slice()?;
    let n = y.len();

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    // Return magnitude spectrum (first half + 1)
    let half = n / 2 + 1;
    let magnitudes: Vec<f64> = buffer[..half].iter().map(|c| c.norm()).collect();
    Ok(PyArray1::from_vec(py, magnitudes))
}

/// Power spectral density via Welch's method.
/// Matches: manifold.primitives.individual.spectral.psd
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, nperseg=None, method="welch"))]
pub fn psd<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    fs: f64,
    nperseg: Option<usize>,
    method: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let seg_len = nperseg.unwrap_or_else(|| (n as f64).min(256.0) as usize);
    let seg_len = seg_len.min(n);

    // Simple periodogram if method != welch or segment == signal length
    let overlap = seg_len / 2;
    let n_freqs = seg_len / 2 + 1;

    let mut psd_accum = vec![0.0; n_freqs];
    let mut n_segments = 0;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    // Hann window
    let window: Vec<f64> = (0..seg_len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (seg_len - 1) as f64).cos()))
        .collect();
    let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>();

    let step = if overlap >= seg_len { 1 } else { seg_len - overlap };
    let mut start = 0;
    while start + seg_len <= n {
        let segment = &y[start..start + seg_len];

        // Apply window
        let mut buffer: Vec<Complex<f64>> = segment
            .iter()
            .zip(window.iter())
            .map(|(&v, &w)| Complex::new(v * w, 0.0))
            .collect();

        fft_algo.process(&mut buffer);

        for i in 0..n_freqs {
            let power = buffer[i].norm_sqr();
            psd_accum[i] += power;
        }

        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Signal too short for PSD computation",
        ));
    }

    // Normalize
    let scale = 1.0 / (fs * window_power * n_segments as f64);
    for p in psd_accum.iter_mut() {
        *p *= scale;
    }
    // Double non-DC, non-Nyquist bins for one-sided spectrum
    for p in psd_accum[1..n_freqs - 1].iter_mut() {
        *p *= 2.0;
    }

    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * fs / seg_len as f64).collect();

    Ok((PyArray1::from_vec(py, freqs), PyArray1::from_vec(py, psd_accum)))
}

/// Dominant frequency (peak of PSD).
/// Matches: manifold.primitives.individual.spectral.dominant_frequency
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0))]
pub fn dominant_frequency(signal: PyReadonlyArray1<f64>, fs: f64) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(0.0);
    }

    let seg_len = n.min(256);
    let n_freqs = seg_len / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    let mut buffer: Vec<Complex<f64>> = y[..seg_len].iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    // Find peak (skip DC)
    let mut max_power = 0.0;
    let mut max_idx = 1;
    for i in 1..n_freqs {
        let power = buffer[i].norm_sqr();
        if power > max_power {
            max_power = power;
            max_idx = i;
        }
    }

    Ok(max_idx as f64 * fs / seg_len as f64)
}

/// Spectral centroid.
/// Matches: manifold.primitives.individual.spectral.spectral_centroid
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0))]
pub fn spectral_centroid(signal: PyReadonlyArray1<f64>, fs: f64) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(0.0);
    }

    let seg_len = n.min(256);
    let n_freqs = seg_len / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    let mut buffer: Vec<Complex<f64>> = y[..seg_len].iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    let magnitudes: Vec<f64> = buffer[..n_freqs].iter().map(|c| c.norm()).collect();
    let total: f64 = magnitudes.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    let centroid: f64 = magnitudes
        .iter()
        .enumerate()
        .map(|(i, &m)| i as f64 * fs / seg_len as f64 * m)
        .sum::<f64>()
        / total;

    Ok(centroid)
}

/// Spectral bandwidth.
/// Matches: manifold.primitives.individual.spectral.spectral_bandwidth
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, p=2))]
pub fn spectral_bandwidth(signal: PyReadonlyArray1<f64>, fs: f64, p: i32) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(0.0);
    }

    let seg_len = n.min(256);
    let n_freqs = seg_len / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    let mut buffer: Vec<Complex<f64>> = y[..seg_len].iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    let magnitudes: Vec<f64> = buffer[..n_freqs].iter().map(|c| c.norm()).collect();
    let total: f64 = magnitudes.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    // Compute centroid first
    let centroid: f64 = magnitudes
        .iter()
        .enumerate()
        .map(|(i, &m)| i as f64 * fs / seg_len as f64 * m)
        .sum::<f64>()
        / total;

    // Bandwidth
    let bw: f64 = magnitudes
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let freq = i as f64 * fs / seg_len as f64;
            m * (freq - centroid).abs().powi(p)
        })
        .sum::<f64>()
        / total;

    Ok(bw.powf(1.0 / p as f64))
}

/// Spectral entropy.
/// Matches: manifold.primitives.individual.spectral.spectral_entropy
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, normalize=true))]
pub fn spectral_entropy(signal: PyReadonlyArray1<f64>, fs: f64, normalize: bool) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(0.0);
    }

    let seg_len = n.min(256);
    let n_freqs = seg_len / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    let mut buffer: Vec<Complex<f64>> = y[..seg_len].iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    // Power spectrum
    let power: Vec<f64> = buffer[..n_freqs].iter().map(|c| c.norm_sqr()).collect();
    let total: f64 = power.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    // Normalize to probability distribution
    let mut entropy = 0.0;
    for &p in &power {
        let prob = p / total;
        if prob > 0.0 {
            entropy -= prob * prob.ln();
        }
    }

    if normalize {
        let max_entropy = (n_freqs as f64).ln();
        if max_entropy > 0.0 {
            entropy /= max_entropy;
        }
    }

    Ok(entropy)
}
