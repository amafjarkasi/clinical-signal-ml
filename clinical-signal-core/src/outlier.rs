use crate::types::SignalError;

pub fn detect_spikes_zscore(
    data: &[f64],
    window: usize,
    z_threshold: f64,
) -> Result<Vec<usize>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window >= data.len() {
        return Err(SignalError::InvalidWindow);
    }
    if !z_threshold.is_finite() || z_threshold <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut spikes = Vec::new();
    for i in window..data.len() {
        let slice = &data[(i - window)..i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / window as f64;
        let std = var.sqrt();
        if std == 0.0 {
            continue;
        }
        let z = ((data[i] - mean) / std).abs();
        if z >= z_threshold {
            spikes.push(i);
        }
    }
    Ok(spikes)
}

pub fn mad_outlier_flags(
    data: &[f64],
    window: usize,
    threshold: f64,
) -> Result<Vec<bool>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![false; data.len()];
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let mut sorted = slice.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let m = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[m - 1] + sorted[m]) / 2.0
        } else {
            sorted[m]
        };

        let mut deviations: Vec<f64> = slice.iter().map(|v| (v - median).abs()).collect();
        deviations.sort_by(|a, b| a.total_cmp(b));
        let dm = deviations.len() / 2;
        let mad = if deviations.len() % 2 == 0 {
            (deviations[dm - 1] + deviations[dm]) / 2.0
        } else {
            deviations[dm]
        };
        if mad == 0.0 {
            out[i] = (data[i] - median).abs() > 0.0;
            continue;
        }

        // 0.6745 scales MAD to be comparable with z-score under normality.
        let modified_z = 0.6745 * (data[i] - median).abs() / mad;
        if modified_z >= threshold {
            out[i] = true;
        }
    }
    Ok(out)
}

pub fn cusum_flags(data: &[f64], drift: f64, threshold: f64) -> Result<Vec<bool>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !drift.is_finite() || drift < 0.0 {
        return Err(SignalError::InvalidThreshold);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![false; data.len()];
    let mut mean = data[0];
    let mut pos = 0.0f64;
    let mut neg = 0.0f64;
    for i in 1..data.len() {
        let x = data[i];
        pos = (pos + x - mean - drift).max(0.0);
        neg = (neg + mean - x - drift).max(0.0);
        if pos > threshold || neg > threshold {
            out[i] = true;
            pos = 0.0;
            neg = 0.0;
        }
        mean += (x - mean) / (i as f64 + 1.0);
    }
    Ok(out)
}

pub fn page_hinkley_flags(
    data: &[f64],
    delta: f64,
    threshold: f64,
) -> Result<Vec<bool>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !delta.is_finite() || delta < 0.0 {
        return Err(SignalError::InvalidThreshold);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![false; data.len()];
    let mut mean = data[0];
    let mut cumulative = 0.0f64;
    let mut min_cumulative = 0.0f64;
    for i in 1..data.len() {
        let x = data[i];
        mean += (x - mean) / (i as f64 + 1.0);
        cumulative += x - mean - delta;
        min_cumulative = min_cumulative.min(cumulative);
        if cumulative - min_cumulative > threshold {
            out[i] = true;
            cumulative = 0.0;
            min_cumulative = 0.0;
        }
    }
    Ok(out)
}

pub fn ewma_residual_flags(
    data: &[f64],
    alpha: f64,
    threshold: f64,
) -> Result<Vec<bool>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(SignalError::InvalidAlpha);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut out = vec![false; data.len()];
    let mut baseline = data[0];
    let mut residual_sum_sq = 0.0f64;
    for i in 1..data.len() {
        let residual = data[i] - baseline;
        if i > 1 {
            let std = (residual_sum_sq / (i as f64 - 1.0)).sqrt();
            if std <= f64::EPSILON {
                out[i] = residual.abs() > threshold;
            } else {
                out[i] = residual.abs() > threshold * std;
            }
        }
        residual_sum_sq += residual * residual;
        baseline = alpha * data[i] + (1.0 - alpha) * baseline;
    }
    Ok(out)
}

pub fn artifact_ratio(data: &[f64], threshold_abs_diff: f64) -> Result<f64, SignalError> {
    if data.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !threshold_abs_diff.is_finite() || threshold_abs_diff <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut artifacts = 0usize;
    for i in 1..data.len() {
        if (data[i] - data[i - 1]).abs() > threshold_abs_diff {
            artifacts += 1;
        }
    }
    Ok(artifacts as f64 / (data.len() - 1) as f64)
}

pub fn baseline_drift_score(data: &[f64], window: usize) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window == 0 || window * 2 > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let head = &data[..window];
    let tail = &data[(data.len() - window)..];
    let head_mean = head.iter().sum::<f64>() / window as f64;
    let tail_mean = tail.iter().sum::<f64>() / window as f64;

    let all_mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data
        .iter()
        .map(|v| {
            let d = *v - all_mean;
            d * d
        })
        .sum::<f64>()
        / data.len() as f64;
    let std = var.sqrt().max(1e-12);
    Ok((tail_mean - head_mean).abs() / std)
}
