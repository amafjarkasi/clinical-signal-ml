use crate::types::SignalError;

#[derive(Debug, Clone, PartialEq)]
pub struct KalmanState {
    pub estimate: f64,
    pub error_covariance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanConfig {
    pub process_noise: f64,
    pub measurement_noise: f64,
    pub initial_estimate: f64,
    pub initial_error: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 1e-5,
            measurement_noise: 1e-2,
            initial_estimate: 0.0,
            initial_error: 1.0,
        }
    }
}

pub fn kalman_filter(data: &[f64], config: &KalmanConfig) -> Result<Vec<KalmanState>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if config.process_noise < 0.0 || config.measurement_noise <= 0.0 {
        return Err(SignalError::InvalidThreshold);
    }

    let mut states = Vec::with_capacity(data.len());
    let mut estimate = config.initial_estimate;
    let mut error_cov = config.initial_error;

    states.push(KalmanState { estimate, error_covariance: error_cov });

    for &measurement in &data[1..] {
        // Predict
        error_cov += config.process_noise;

        // Update
        let kalman_gain = error_cov / (error_cov + config.measurement_noise);
        estimate = estimate + kalman_gain * (measurement - estimate);
        error_cov = (1.0 - kalman_gain) * error_cov;

        states.push(KalmanState { estimate, error_covariance: error_cov });
    }

    Ok(states)
}

pub fn kalman_baseline(data: &[f64], process_noise: f64, measurement_noise: f64) -> Result<Vec<f64>, SignalError> {
    let config = KalmanConfig {
        process_noise,
        measurement_noise,
        initial_estimate: data.first().copied().unwrap_or(0.0),
        initial_error: 1.0,
    };
    let states = kalman_filter(data, &config)?;
    Ok(states.iter().map(|s| s.estimate).collect())
}

pub fn kalman_residuals(data: &[f64], config: &KalmanConfig) -> Result<Vec<f64>, SignalError> {
    let states = kalman_filter(data, config)?;
    let residuals: Vec<f64> = data.iter().zip(states.iter())
        .map(|(&v, s)| v - s.estimate)
        .collect();
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_filter_smooths_noisy_signal() {
        let signal: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.01).sin() * 2.0).collect();
        let config = KalmanConfig { initial_estimate: signal[0], ..Default::default() };
        let states = kalman_filter(&signal, &config).unwrap();
        assert_eq!(states.len(), 50);
        for s in &states {
            assert!(s.estimate > 5.0 && s.estimate < 15.0);
        }
    }

    #[test]
    fn kalman_filter_reduces_variance() {
        let noisy = [10.0, 12.0, 8.0, 11.0, 9.0, 10.0, 13.0, 7.0, 10.0, 10.0];
        let config = KalmanConfig {
            initial_estimate: noisy[0],
            measurement_noise: 4.0,
            process_noise: 1e-6,
            initial_error: 1.0,
        };
        let states = kalman_filter(&noisy, &config).unwrap();
        let estimates: Vec<f64> = states.iter().map(|s| s.estimate).collect();
        let est_mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let est_var = estimates.iter().map(|v| { let d = v - est_mean; d * d }).sum::<f64>() / estimates.len() as f64;
        let sig_mean = noisy.iter().sum::<f64>() / noisy.len() as f64;
        let sig_var = noisy.iter().map(|v| { let d = v - sig_mean; d * d }).sum::<f64>() / noisy.len() as f64;
        assert!(est_var < sig_var, "Kalman estimates should have lower variance: est_var={est_var}, sig_var={sig_var}");
    }

    #[test]
    fn kalman_filter_rejects_empty() {
        let config = KalmanConfig::default();
        assert_eq!(kalman_filter(&[], &config).unwrap_err(), SignalError::EmptyInput);
    }

    #[test]
    fn kalman_filter_rejects_bad_noise() {
        let config = KalmanConfig { measurement_noise: 0.0, ..Default::default() };
        assert_eq!(kalman_filter(&[1.0], &config).unwrap_err(), SignalError::InvalidThreshold);
    }

    #[test]
    fn kalman_baseline_returns_estimates() {
        let data = [5.0, 6.0, 4.0, 7.0, 5.0];
        let baseline = kalman_baseline(&data, 1e-3, 1.0).unwrap();
        assert_eq!(baseline.len(), 5);
        assert!((baseline[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn kalman_residuals_detect_outliers() {
        let data = [10.0, 10.0, 10.0, 50.0, 10.0, 10.0];
        let config = KalmanConfig { measurement_noise: 2.0, ..Default::default() };
        let residuals = kalman_residuals(&data, &config).unwrap();
        assert!(residuals[3].abs() > residuals[1].abs(), "Spike residual should be larger than normal");
    }
}
