use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Arithmetic mean.
#[pyfunction]
pub fn mean(signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(y.iter().sum::<f64>() / y.len() as f64)
}

/// Standard deviation.
#[pyfunction]
#[pyo3(signature = (signal, ddof=0))]
pub fn std(signal: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n <= ddof {
        return Ok(f64::NAN);
    }
    let mean_val = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / (n - ddof) as f64;
    Ok(var.sqrt())
}

/// Variance.
#[pyfunction]
#[pyo3(signature = (signal, ddof=0))]
pub fn variance(signal: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n <= ddof {
        return Ok(f64::NAN);
    }
    let mean_val = y.iter().sum::<f64>() / n as f64;
    Ok(y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / (n - ddof) as f64)
}

/// Skewness (Fisher).
#[pyfunction]
pub fn skewness(signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 3 {
        return Ok(f64::NAN);
    }
    let mean_val = y.iter().sum::<f64>() / n as f64;
    let m2: f64 = y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64;
    let m3: f64 = y.iter().map(|x| (x - mean_val).powi(3)).sum::<f64>() / n as f64;

    if m2 <= 0.0 {
        return Ok(0.0);
    }

    Ok(m3 / m2.powf(1.5))
}

/// Kurtosis (Fisher by default: excess kurtosis = kurtosis - 3).
#[pyfunction]
#[pyo3(signature = (signal, fisher=true))]
pub fn kurtosis(signal: PyReadonlyArray1<f64>, fisher: bool) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 4 {
        return Ok(f64::NAN);
    }
    let mean_val = y.iter().sum::<f64>() / n as f64;
    let m2: f64 = y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64;
    let m4: f64 = y.iter().map(|x| (x - mean_val).powi(4)).sum::<f64>() / n as f64;

    if m2 <= 0.0 {
        return Ok(0.0);
    }

    let kurt = m4 / (m2 * m2);
    if fisher {
        Ok(kurt - 3.0)
    } else {
        Ok(kurt)
    }
}

/// Root mean square.
#[pyfunction]
pub fn rms(signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.is_empty() {
        return Ok(f64::NAN);
    }
    let ms: f64 = y.iter().map(|x| x * x).sum::<f64>() / y.len() as f64;
    Ok(ms.sqrt())
}

/// Peak to peak (max - min).
#[pyfunction]
pub fn peak_to_peak(signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.is_empty() {
        return Ok(f64::NAN);
    }
    let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    Ok(max - min)
}

/// Crest factor (peak / RMS).
#[pyfunction]
pub fn crest_factor(signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.is_empty() {
        return Ok(f64::NAN);
    }
    let peak = y.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let ms: f64 = y.iter().map(|x| x * x).sum::<f64>() / y.len() as f64;
    let rms_val = ms.sqrt();
    if rms_val <= 0.0 {
        return Ok(f64::NAN);
    }
    Ok(peak / rms_val)
}

/// Count of zero crossings.
#[pyfunction]
pub fn zero_crossings(signal: PyReadonlyArray1<f64>) -> PyResult<usize> {
    let y = signal.as_slice()?;
    if y.len() < 2 {
        return Ok(0);
    }
    let mut count = 0;
    for i in 1..y.len() {
        if (y[i - 1] >= 0.0 && y[i] < 0.0) || (y[i - 1] < 0.0 && y[i] >= 0.0) {
            count += 1;
        }
    }
    Ok(count)
}

/// Count of mean crossings.
#[pyfunction]
pub fn mean_crossings(signal: PyReadonlyArray1<f64>) -> PyResult<usize> {
    let y = signal.as_slice()?;
    let n = y.len();
    if n < 2 {
        return Ok(0);
    }
    let mean_val = y.iter().sum::<f64>() / n as f64;
    let mut count = 0;
    for i in 1..n {
        let prev = y[i - 1] - mean_val;
        let curr = y[i] - mean_val;
        if (prev >= 0.0 && curr < 0.0) || (prev < 0.0 && curr >= 0.0) {
            count += 1;
        }
    }
    Ok(count)
}
