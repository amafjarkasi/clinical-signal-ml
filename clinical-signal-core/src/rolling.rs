use crate::types::SignalError;

pub fn trend_slope(values: &[f64]) -> Result<f64, SignalError> {
    if values.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, y) in values.iter().enumerate() {
        let x = i as f64;
        let dx = x - x_mean;
        numerator += dx * (y - y_mean);
        denominator += dx * dx;
    }

    if denominator == 0.0 {
        return Ok(0.0);
    }
    Ok(numerator / denominator)
}

pub fn ewma(data: &[f64], alpha: f64) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(SignalError::InvalidAlpha);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(alpha * data[i] + (1.0 - alpha) * out[i - 1]);
    }
    Ok(out)
}

pub fn rolling_mean(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    let mut sum: f64 = data[..window].iter().sum();
    out[window - 1] = sum / window as f64;
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        out[i] = sum / window as f64;
    }
    Ok(out)
}

pub fn rolling_variance(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / window as f64;
        out[i] = var;
    }
    Ok(out)
}

pub fn rolling_median(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let mut slice: Vec<f64> = data[(i + 1 - window)..=i].to_vec();
        slice.sort_by(|a, b| a.total_cmp(b));
        let m = window / 2;
        out[i] = if window % 2 == 0 {
            (slice[m - 1] + slice[m]) / 2.0
        } else {
            slice[m]
        };
    }
    Ok(out)
}

pub fn rolling_min(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        out[i] = slice.iter().copied().fold(f64::INFINITY, f64::min);
    }
    Ok(out)
}

pub fn rolling_max(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        out[i] = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    Ok(out)
}

pub fn rolling_range(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    let mins = rolling_min(data, window)?;
    let maxs = rolling_max(data, window)?;
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !mins[i].is_nan() && !maxs[i].is_nan() {
            out[i] = maxs[i] - mins[i];
        }
    }
    Ok(out)
}

pub fn rolling_slope(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        out[i] = trend_slope(&data[(i + 1 - window)..=i])?;
    }
    Ok(out)
}

pub fn trend_change_points(data: &[f64], window: usize) -> Result<Vec<usize>, SignalError> {
    let slopes = rolling_slope(data, window)?;
    let mut out = Vec::new();
    let mut last_sign = 0i8;
    let mut last_idx = 0usize;
    for (i, slope) in slopes.iter().enumerate() {
        if !slope.is_finite() {
            continue;
        }
        let sign = if *slope > 0.0 {
            1
        } else if *slope < 0.0 {
            -1
        } else {
            0
        };
        if sign == 0 {
            continue;
        }
        if last_sign != 0 && sign != last_sign {
            out.push((last_idx + i) / 2);
        }
        last_sign = sign;
        last_idx = i;
    }
    Ok(out)
}

pub fn rolling_cv(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        if mean.abs() < f64::EPSILON {
            continue;
        }
        let var = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / window as f64;
        out[i] = var.sqrt() / mean.abs();
    }
    Ok(out)
}

pub fn rolling_skewness(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 3 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let n = window as f64;
        let mean = slice.iter().sum::<f64>() / n;
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        for &v in slice {
            let d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n;
        m3 /= n;
        if m2 < f64::EPSILON {
            continue;
        }
        out[i] = m3 / (m2 * m2.sqrt());
    }
    Ok(out)
}

pub fn rolling_kurtosis(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 4 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let n = window as f64;
        let mean = slice.iter().sum::<f64>() / n;
        let mut m2 = 0.0f64;
        let mut m4 = 0.0f64;
        for &v in slice {
            let d = v - mean;
            m2 += d * d;
            m4 += d * d * d * d;
        }
        m2 /= n;
        m4 /= n;
        if m2 < f64::EPSILON {
            continue;
        }
        // excess kurtosis (Fisher): kurt - 3
        out[i] = m4 / (m2 * m2) - 3.0;
    }
    Ok(out)
}

pub fn rolling_entropy(data: &[f64], window: usize, bins: usize) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    if bins == 0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let min_val = slice.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        if range < f64::EPSILON {
            out[i] = 0.0;
            continue;
        }
        let bin_width = range / bins as f64;
        let mut counts = vec![0usize; bins];
        for &v in slice {
            let idx = ((v - min_val) / bin_width).floor() as usize;
            let idx = if idx >= bins { bins - 1 } else { idx };
            counts[idx] += 1;
        }
        let n = window as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        out[i] = entropy;
    }
    Ok(out)
}

pub fn moving_percentile(data: &[f64], window: usize, percentile: f64) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    if percentile < 0.0 || percentile > 100.0 || !percentile.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![f64::NAN; data.len()];
    for i in (window - 1)..data.len() {
        let mut slice: Vec<f64> = data[(i + 1 - window)..=i].to_vec();
        slice.sort_by(|a, b| a.total_cmp(b));
        let rank = percentile / 100.0 * (window - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = (lo + 1).min(window - 1);
        let frac = rank - lo as f64;
        out[i] = slice[lo] + frac * (slice[hi] - slice[lo]);
    }
    Ok(out)
}
