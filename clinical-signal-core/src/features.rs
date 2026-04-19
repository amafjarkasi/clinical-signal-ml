use crate::types::{SignalError, MACross};
use crate::rolling::rolling_mean;
use crate::rolling::rolling_slope;

pub fn crossings_above_threshold(data: &[f64], threshold: f64) -> Result<usize, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !threshold.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }
    let mut count = 0usize;
    for i in 1..data.len() {
        if data[i - 1] <= threshold && data[i] > threshold {
            count += 1;
        }
    }
    Ok(count)
}

pub fn time_above_threshold_ratio(data: &[f64], threshold: f64) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !threshold.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }
    let above = data.iter().filter(|v| **v > threshold).count();
    Ok(above as f64 / data.len() as f64)
}

pub fn threshold_dwell_time(data: &[f64], threshold: f64) -> Result<usize, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !threshold.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }

    let mut best = 0usize;
    let mut current = 0usize;
    for &v in data {
        if v > threshold {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 0;
        }
    }
    Ok(best)
}

pub fn threshold_burden(data: &[f64], threshold: f64) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if !threshold.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }

    let area: f64 = data.iter().map(|&v| (v - threshold).max(0.0)).sum();
    Ok(area)
}

pub fn recovery_half_time(data: &[f64]) -> Result<Option<usize>, SignalError> {
    if data.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut peak_idx = 0usize;
    let mut peak_val = data[0];
    for (i, &v) in data.iter().enumerate() {
        if v > peak_val {
            peak_val = v;
            peak_idx = i;
        }
    }

    let half_peak = peak_val / 2.0;
    for i in (peak_idx + 1)..data.len() {
        if data[i] <= half_peak {
            return Ok(Some(i - peak_idx));
        }
    }
    Ok(None)
}

pub fn trend_stability_index(data: &[f64], window: usize) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if window < 2 || window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let slopes = rolling_slope(data, window)?;
    let mut sum_abs_slope = 0.0f64;
    let mut count = 0usize;
    let mut sum_abs_diff = 0.0f64;

    for i in 1..data.len() {
        let diff = (data[i] - data[i - 1]).abs();
        sum_abs_diff += diff;
    }

    for &s in &slopes {
        if !s.is_nan() {
            sum_abs_slope += s.abs();
            count += 1;
        }
    }

    if count == 0 || sum_abs_diff < f64::EPSILON {
        return Ok(0.0);
    }

    let avg_abs_slope = sum_abs_slope / count as f64;
    let avg_abs_diff = sum_abs_diff / (data.len() - 1) as f64;
    Ok((avg_abs_slope / avg_abs_diff).clamp(0.0, 1.0))
}

pub fn peak_to_peak_interval(data: &[f64]) -> Result<Vec<usize>, SignalError> {
    if data.len() < 3 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..(data.len() - 1) {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    let mut intervals = Vec::new();
    for w in peaks.windows(2) {
        intervals.push(w[1] - w[0]);
    }
    Ok(intervals)
}

pub fn spectral_flatness(data: &[f64], window: usize) -> Result<Vec<f64>, SignalError> {
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
        let min_val = slice.iter().copied().fold(f64::INFINITY, f64::min);
        // Shift all values to be strictly positive
        let shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };
        let shifted: Vec<f64> = slice.iter().map(|v| *v + shift).collect();
        let arith_mean = shifted.iter().sum::<f64>() / window as f64;
        if arith_mean <= 0.0 {
            continue;
        }
        let log_sum: f64 = shifted.iter().map(|v| v.ln()).sum();
        let geo_mean = (log_sum / window as f64).exp();
        out[i] = (geo_mean / arith_mean).clamp(0.0, 1.0);
    }
    Ok(out)
}

pub fn zero_crossing_rate(data: &[f64]) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if data.len() < 2 {
        return Ok(0.0);
    }

    let mut crossings = 0usize;
    for i in 1..data.len() {
        let prev_sign = data[i - 1].signum();
        let curr_sign = data[i].signum();
        if prev_sign != curr_sign && prev_sign != 0.0 && curr_sign != 0.0 {
            crossings += 1;
        }
    }
    Ok(crossings as f64 / (data.len() - 1) as f64)
}

pub fn fractal_dimension(data: &[f64], k_max: usize) -> Result<f64, SignalError> {
    if data.len() < 3 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if k_max < 2 || k_max >= data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let n = data.len();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for k in 2..=k_max {
        let m = (n - 1) / k;
        if m == 0 {
            continue;
        }
        let mut lk = 0.0f64;
        for j in 0..m {
            let start = j * k;
            let end = (start + k).min(n - 1);
            lk += (data[end] - data[start]).abs();
        }
        let norm_factor = (n - 1) as f64 / (m * k) as f64;
        lk *= norm_factor / k as f64;
        if lk > 0.0 {
            xs.push((k as f64).ln());
            ys.push(lk.ln());
        }
    }

    if xs.len() < 2 {
        return Ok(1.0);
    }

    let xn = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / xn;
    let y_mean = ys.iter().sum::<f64>() / xn;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = *x - x_mean;
        num += dx * (*y - y_mean);
        den += dx * dx;
    }
    if den.abs() < f64::EPSILON {
        return Ok(1.0);
    }
    Ok((num / den).abs())
}

pub fn lagged_autocorrelation(data: &[f64], lag: usize) -> Result<f64, SignalError> {
    if data.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if lag >= data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut var = 0.0f64;
    for &v in data {
        let d = v - mean;
        var += d * d;
    }
    if var < f64::EPSILON {
        return Ok(1.0);
    }

    let mut acf = 0.0f64;
    for i in 0..(n - lag) {
        acf += (data[i] - mean) * (data[i + lag] - mean);
    }
    Ok(acf / var)
}

pub fn dominant_frequency(data: &[f64]) -> Result<f64, SignalError> {
    if data.len() < 4 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let demeaned: Vec<f64> = data.iter().map(|v| v - mean).collect();
    let n = demeaned.len();
    let max_lag = n / 2;

    let mut var = 0.0f64;
    for &v in &demeaned {
        var += v * v;
    }
    if var < f64::EPSILON {
        return Ok(0.0);
    }

    let mut best_lag = 1usize;
    let mut best_acf = f64::NEG_INFINITY;
    for lag in 1..=max_lag {
        let mut acf = 0.0f64;
        for i in 0..(n - lag) {
            acf += demeaned[i] * demeaned[i + lag];
        }
        acf /= var;
        if acf > best_acf {
            best_acf = acf;
            best_lag = lag;
        }
    }

    Ok(1.0 / best_lag as f64)
}

pub fn moving_average_crossover(data: &[f64], fast_window: usize, slow_window: usize) -> Result<Vec<MACross>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if fast_window < 2 || slow_window < 2 || fast_window >= slow_window {
        return Err(SignalError::InvalidWindow);
    }
    if slow_window > data.len() {
        return Err(SignalError::InvalidWindow);
    }

    let fast_ma = rolling_mean(data, fast_window)?;
    let slow_ma = rolling_mean(data, slow_window)?;

    let mut signals = Vec::new();
    for i in 1..data.len() {
        let prev_fast = fast_ma[i - 1];
        let prev_slow = slow_ma[i - 1];
        let curr_fast = fast_ma[i];
        let curr_slow = slow_ma[i];

        if prev_fast.is_nan() || prev_slow.is_nan() || curr_fast.is_nan() || curr_slow.is_nan() {
            continue;
        }

        if prev_fast <= prev_slow && curr_fast > curr_slow {
            signals.push(MACross::Bullish);
        } else if prev_fast >= prev_slow && curr_fast < curr_slow {
            signals.push(MACross::Bearish);
        }
    }
    Ok(signals)
}

pub fn band_energy(data: &[f64]) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let demeaned: Vec<f64> = data.iter().map(|v| v - mean).collect();

    // Estimate frequency via zero-crossings, then compute energy
    let n = demeaned.len();
    let mut crossings = 0usize;
    for i in 1..n {
        let prev_sign = demeaned[i - 1].signum();
        let curr_sign = demeaned[i].signum();
        if prev_sign != curr_sign && prev_sign != 0.0 && curr_sign != 0.0 {
            crossings += 1;
        }
    }

    let energy: f64 = demeaned.iter().map(|v| v * v).sum();
    let freq_factor = crossings as f64 / n as f64;
    Ok(energy * freq_factor)
}

pub fn waveform_symmetry(data: &[f64]) -> Result<f64, SignalError> {
    if data.len() < 3 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let n = data.len();
    let mid = n / 2;
    let half_len = mid.min(n - mid - 1);
    if half_len == 0 {
        return Ok(1.0);
    }

    let mut left_sum = 0.0f64;
    let mut diff_sum = 0.0f64;
    for i in 0..half_len {
        let left = data[mid - 1 - i];
        let right = data[mid + 1 + i];
        left_sum += left.abs() + right.abs();
        diff_sum += (left - right).abs();
    }

    if left_sum < f64::EPSILON {
        return Ok(1.0);
    }
    Ok(1.0 - diff_sum / left_sum)
}

pub fn jitter_score(peak_indices: &[usize]) -> Result<f64, SignalError> {
    if peak_indices.len() < 2 {
        return Err(SignalError::InvalidLength);
    }

    let mut intervals: Vec<f64> = Vec::new();
    for w in peak_indices.windows(2) {
        intervals.push((w[1] - w[0]) as f64);
    }

    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean < f64::EPSILON {
        return Ok(0.0);
    }

    let var = intervals.iter().map(|v| { let d = v - mean; d * d }).sum::<f64>() / intervals.len() as f64;
    Ok(var.sqrt() / mean)
}

pub fn peak_prominence(data: &[f64]) -> Result<Vec<(usize, f64)>, SignalError> {
    if data.len() < 3 {
        return Err(SignalError::InvalidLength);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..(data.len() - 1) {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    let mut result = Vec::new();
    for &pk in &peaks {
        // Find minimum to the left
        let left_min = data[..pk].iter().copied().fold(f64::INFINITY, f64::min);
        // Find minimum to the right
        let right_min = data[(pk + 1)..].iter().copied().fold(f64::INFINITY, f64::min);
        let prominence = data[pk] - left_min.min(right_min);
        result.push((pk, prominence));
    }
    Ok(result)
}

pub fn signal_saturation_ratio(data: &[f64], lo: f64, hi: f64) -> Result<f64, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if lo >= hi || !lo.is_finite() || !hi.is_finite() {
        return Err(SignalError::InvalidRange);
    }

    let saturated = data.iter().filter(|&&v| v <= lo || v >= hi).count();
    Ok(saturated as f64 / data.len() as f64)
}
