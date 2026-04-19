use crate::types::SignalError;

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingEwma {
    pub value: f64,
    pub alpha: f64,
    pub initialized: bool,
}

impl StreamingEwma {
    pub fn new(alpha: f64) -> Result<Self, SignalError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(SignalError::InvalidAlpha);
        }
        Ok(Self { value: 0.0, alpha, initialized: false })
    }

    pub fn update(&mut self, sample: f64) -> Result<f64, SignalError> {
        if !sample.is_finite() {
            return Err(SignalError::NonFiniteValue);
        }
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        Ok(self.value)
    }

    pub fn process_batch(&mut self, data: &[f64]) -> Result<Vec<f64>, SignalError> {
        if data.is_empty() {
            return Err(SignalError::EmptyInput);
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(SignalError::NonFiniteValue);
        }
        let mut out = Vec::with_capacity(data.len());
        for &v in data {
            out.push(self.update(v)?);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingMean {
    pub mean: f64,
    pub count: usize,
}

impl StreamingMean {
    pub fn new() -> Self {
        Self { mean: 0.0, count: 0 }
    }

    pub fn update(&mut self, sample: f64) -> Result<f64, SignalError> {
        if !sample.is_finite() {
            return Err(SignalError::NonFiniteValue);
        }
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        Ok(self.mean)
    }

    pub fn process_batch(&mut self, data: &[f64]) -> Result<Vec<f64>, SignalError> {
        if data.is_empty() {
            return Err(SignalError::EmptyInput);
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(SignalError::NonFiniteValue);
        }
        let mut out = Vec::with_capacity(data.len());
        for &v in data {
            out.push(self.update(v)?);
        }
        Ok(out)
    }
}

impl Default for StreamingMean {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingVariance {
    pub mean: f64,
    pub m2: f64,
    pub count: usize,
}

impl StreamingVariance {
    pub fn new() -> Self {
        Self { mean: 0.0, m2: 0.0, count: 0 }
    }

    pub fn update(&mut self, sample: f64) -> Result<(f64, f64), SignalError> {
        if !sample.is_finite() {
            return Err(SignalError::NonFiniteValue);
        }
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
        let variance = if self.count < 2 { 0.0 } else { self.m2 / self.count as f64 };
        Ok((self.mean, variance))
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / self.count as f64 }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for StreamingVariance {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingCusum {
    pub pos_sum: f64,
    pub neg_sum: f64,
    pub mean: f64,
    pub count: usize,
    pub drift: f64,
    pub threshold: f64,
    pub flags: Vec<usize>,
}

impl StreamingCusum {
    pub fn new(drift: f64, threshold: f64) -> Result<Self, SignalError> {
        if !drift.is_finite() || drift < 0.0 {
            return Err(SignalError::InvalidThreshold);
        }
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(SignalError::InvalidThreshold);
        }
        Ok(Self {
            pos_sum: 0.0,
            neg_sum: 0.0,
            mean: 0.0,
            count: 0,
            drift,
            threshold,
            flags: Vec::new(),
        })
    }

    pub fn update(&mut self, sample: f64) -> Result<bool, SignalError> {
        if !sample.is_finite() {
            return Err(SignalError::NonFiniteValue);
        }
        self.count += 1;
        if self.count == 1 {
            self.mean = sample;
            return Ok(false);
        }
        self.mean += (sample - self.mean) / self.count as f64;
        self.pos_sum = (self.pos_sum + sample - self.mean - self.drift).max(0.0);
        self.neg_sum = (self.neg_sum + self.mean - sample - self.drift).max(0.0);
        let flagged = self.pos_sum > self.threshold || self.neg_sum > self.threshold;
        if flagged {
            self.flags.push(self.count - 1);
            self.pos_sum = 0.0;
            self.neg_sum = 0.0;
        }
        Ok(flagged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_ewma_processes_samples() {
        let mut ewma = StreamingEwma::new(0.5).unwrap();
        let v1 = ewma.update(10.0).unwrap();
        assert!((v1 - 10.0).abs() < 1e-12);
        let v2 = ewma.update(20.0).unwrap();
        assert!((v2 - 15.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_ewma_batch_matches_individual() {
        let mut e1 = StreamingEwma::new(0.3).unwrap();
        let mut e2 = StreamingEwma::new(0.3).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let batch = e1.process_batch(&data).unwrap();
        for &v in &data {
            e2.update(v).unwrap();
        }
        assert!((e1.value - e2.value).abs() < 1e-12);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn streaming_ewma_rejects_bad_alpha() {
        assert_eq!(StreamingEwma::new(-0.1).unwrap_err(), SignalError::InvalidAlpha);
    }

    #[test]
    fn streaming_mean_converges() {
        let mut sm = StreamingMean::new();
        for v in [10.0, 20.0, 30.0] {
            sm.update(v).unwrap();
        }
        assert!((sm.mean - 20.0).abs() < 1e-12);
        assert_eq!(sm.count, 3);
    }

    #[test]
    fn streaming_variance_welford() {
        let mut sv = StreamingVariance::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            sv.update(v).unwrap();
        }
        assert!((sv.mean - 5.0).abs() < 1e-12);
        assert!((sv.variance() - 4.0).abs() < 1e-12);
        assert!((sv.std_dev() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_cusum_detects_shift() {
        let mut cusum = StreamingCusum::new(0.05, 1.5).unwrap();
        let mut detected = false;
        for &v in &[0.0, 0.0, 0.0, 2.0, 2.2, 2.1, 2.0] {
            if cusum.update(v).unwrap() {
                detected = true;
            }
        }
        assert!(detected, "Streaming CUSUM should detect mean shift");
        assert!(!cusum.flags.is_empty());
    }

    #[test]
    fn streaming_cusum_rejects_bad_threshold() {
        assert_eq!(StreamingCusum::new(0.1, 0.0).unwrap_err(), SignalError::InvalidThreshold);
    }

    #[test]
    fn streaming_mean_batch() {
        let mut sm = StreamingMean::new();
        let out = sm.process_batch(&[10.0, 20.0, 30.0]).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-12);
        assert!((out[1] - 15.0).abs() < 1e-12);
        assert!((out[2] - 20.0).abs() < 1e-12);
    }
}
