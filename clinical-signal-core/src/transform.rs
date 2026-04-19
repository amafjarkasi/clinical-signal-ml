use crate::types::SignalError;

pub fn interpolate_nan_gaps(data: &[f64], max_gap: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if max_gap == 0 {
        return Ok(data.to_vec());
    }
    if data.iter().any(|v| v.is_infinite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i < n {
        if !out[i].is_nan() {
            i += 1;
            continue;
        }

        let start = i;
        while i < n && out[i].is_nan() {
            i += 1;
        }
        let end = i; // exclusive
        let gap_len = end - start;

        if gap_len > max_gap || start == 0 || end >= n {
            continue;
        }

        let left = out[start - 1];
        let right = out[end];
        if left.is_nan() || right.is_nan() {
            continue;
        }

        let span = (gap_len + 1) as f64;
        for j in 0..gap_len {
            let t = (j + 1) as f64 / span;
            out[start + j] = left + (right - left) * t;
        }
    }

    Ok(out)
}

pub fn derivative(data: &[f64]) -> Result<Vec<f64>, SignalError> {
    if data.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut out = vec![0.0; data.len()];
    out[0] = data[1] - data[0];
    out[data.len() - 1] = data[data.len() - 1] - data[data.len() - 2];
    for i in 1..(data.len() - 1) {
        out[i] = (data[i + 1] - data[i - 1]) / 2.0;
    }
    Ok(out)
}

pub fn second_derivative(data: &[f64]) -> Result<Vec<f64>, SignalError> {
    if data.len() < 3 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut out = vec![0.0; data.len()];
    for i in 1..(data.len() - 1) {
        out[i] = data[i + 1] - 2.0 * data[i] + data[i - 1];
    }
    out[0] = out[1];
    out[data.len() - 1] = out[data.len() - 2];
    Ok(out)
}

pub fn resample_linear(data: &[f64], n: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if n < 2 {
        return Err(SignalError::InvalidLength);
    }

    let src_len = data.len();
    if src_len == 1 {
        return Ok(vec![data[0]; n]);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * (src_len - 1) as f64 / (n - 1) as f64;
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(src_len - 1);
        let frac = t - lo as f64;
        out.push(data[lo] + frac * (data[hi] - data[lo]));
    }
    Ok(out)
}

pub fn rate_of_change(data: &[f64]) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut out = vec![0.0; data.len()];
    for i in 1..data.len() {
        if data[i - 1].abs() < f64::EPSILON {
            out[i] = 0.0;
        } else {
            out[i] = (data[i] - data[i - 1]) / data[i - 1].abs() * 100.0;
        }
    }
    Ok(out)
}

pub fn signal_energy(data: &[f64]) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    Ok(data.iter().map(|v| v * v).sum())
}

pub fn rms_power(data: &[f64]) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    let sum_sq: f64 = data.iter().map(|v| v * v).sum();
    Ok((sum_sq / data.len() as f64).sqrt())
}

pub fn signal_to_noise_ratio(data: &[f64]) -> Result<f64, SignalError> {
    if data.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|v| { let d = v - mean; d * d }).sum::<f64>() / n;
    if var < f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    let std = var.sqrt();
    Ok(20.0 * (mean.abs() / std).log10())
}

pub fn vital_sign_zscore(data: &[f64], pop_mean: f64, pop_std: f64) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if pop_std <= 0.0 || !pop_std.is_finite() || !pop_mean.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }

    Ok(data.iter().map(|v| (v - pop_mean) / pop_std).collect())
}
