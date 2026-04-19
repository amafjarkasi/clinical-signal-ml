<p align="center">
  <img src="docs/logo-simple.svg" alt="Clinical Signal ML" width="860" />
</p>

<h1 align="center">Clinical Signal ML</h1>

<p align="center">
  <strong>Clean-room healthcare signal analytics for edge, bedside, and remote monitoring systems.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.70%2B-orange" alt="Rust 1.70+" />
  <img src="https://img.shields.io/badge/tests-127%20passing-brightgreen" alt="127 tests passing" />
  <img src="https://img.shields.io/badge/TDD-red%E2%86%92green-blue" alt="TDD red-green" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
</p>

<p align="center">
  <code>Rust core</code> &middot; <code>TDD-first</code> &middot; <code>Deterministic</code> &middot; <code>Zero ML deps</code> &middot; <code>Healthcare-native</code>
</p>

---

## Why This Exists

Building clinical monitoring software shouldn't require pulling in a heavyweight ML framework just to compute a rolling heart-rate trend or flag a deteriorating patient. **Clinical Signal ML** is a focused, deterministic Rust crate that gives you:

- **Interpretable features** — every function has a clear clinical or signal-processing meaning, no black boxes
- **Strict input validation** — non-finite values, impossible SpO2 ranges, mismatched array lengths all return explicit errors
- **Predictable performance** — no hidden allocations in hot loops, no garbage collection pauses, suitable for embedded and edge targets
- **Auditable outputs** — pure functions with deterministic results, reproducible across runs

## API Reference

Every public function takes plain `&[f64]` slices or primitive values and returns `Result<_, SignalError>`. Rolling-window functions output `Vec<f64>` with `NaN` for positions where the window hasn't filled yet.

---

### Rolling Window Primitives

Core building blocks for any time-series pipeline. These are the foundation that higher-level features build on.

#### `ewma(data: &[f64], alpha: f64) -> Vec<f64>`

**Exponentially Weighted Moving Average.** Produces a smoothed version of the input where each output sample is a weighted blend of the current input and the previous output. The `alpha` parameter (0-1) controls responsiveness: high alpha tracks the input closely, low alpha produces heavy smoothing. Time constant is approximately `1/alpha` samples. Commonly used as a fast, state-friendly alternative to a simple moving average when you need O(1) updates per sample.

- **`data`** -- input signal samples
- **`alpha`** -- smoothing factor in `[0, 1]`. 1.0 = no smoothing, 0.0 = frozen at first value
- **Returns** -- smoothed signal of same length

#### `rolling_mean(data: &[f64], window: usize) -> Vec<f64>`

**Simple Moving Average (SMA).** Computes the arithmetic mean of the last `window` samples at each position. Slides a fixed-width window across the signal. Each output sample represents the local average, filtering out high-frequency noise. The output is `NaN` for the first `window - 1` positions. Internally uses a running sum for O(n) total computation instead of recomputing the sum at each step.

- **`window`** -- number of samples to average (must be >= 1 and <= data length)
- **Returns** -- smoothed signal with `NaN` prefix

#### `rolling_variance(data: &[f64], window: usize) -> Vec<f64>`

**Population Variance per Window.** Measures how much the signal deviates from its local mean within each window. High variance indicates a noisy or erratic segment; low variance indicates a stable segment. Uses population variance (divides by `n`, not `n-1`) because a rolling window is a complete set, not a sample from a larger population. Useful for detecting periods of instability in vital signs.

- **Returns** -- variance values in the same units as the input squared

#### `rolling_median(data: &[f64], window: usize) -> Vec<f64>`

**Rolling Median.** Computes the middle value of the sorted window contents at each position. Unlike the mean, the median is **robust to outliers** -- a single spike or artifact won't shift the median much. Ideal for baseline estimation in noisy physiological signals where you want to reject transient artifacts without distorting the underlying trend.

- **Returns** -- median values, robust against spikes

#### `rolling_min / rolling_max(data: &[f64], window: usize) -> Vec<f64>`

**Rolling Minimum / Maximum.** Tracks the lowest or highest value within each window. Useful for envelope detection (finding the bounds of a waveform), detecting sensor saturation, or establishing dynamic thresholds for alert generation.

- **Returns** -- per-window extreme values

#### `rolling_range(data: &[f64], window: usize) -> Vec<f64>`

**Rolling Range (max - min).** Measures the peak-to-peak spread within each window -- how much the signal varies locally. A rising range can indicate increasing instability in a vital sign (e.g., blood pressure becoming more variable). Computed by combining `rolling_min` and `rolling_max`.

#### `rolling_slope(data: &[f64], window: usize) -> Vec<f64>`

**OLS Linear Trend Slope per Window.** Fits a straight line through the samples in each window using ordinary least squares and returns the slope. Positive slope = upward trend, negative = downward, magnitude = rate of change per sample. Essential for detecting deteriorating trends: a persistently positive heart-rate slope over clinical windows is an early warning sign.

- **`window`** -- must be >= 2 (need at least 2 points to define a slope)
- **Returns** -- slope in units-per-sample at each position

#### `rolling_cv(data: &[f64], window: usize) -> Vec<f64>`

**Coefficient of Variation per Window.** Computes `std / |mean|` for each window. This is a normalized measure of dispersion that allows comparing variability across signals with different scales. A CV of 0.1 means the standard deviation is 10% of the mean. Useful for detecting when a signal becomes proportionally more variable regardless of its absolute level.

- **Returns** -- dimensionless variability ratio (NaN where mean is near zero)

#### `rolling_entropy(data: &[f64], window: usize, bins: usize) -> Vec<f64>`

**Shannon Entropy over Discretized Bins.** Divides the value range within each window into `bins` equal-width buckets, counts how many samples fall into each bucket, and computes the Shannon entropy of the resulting distribution. High entropy = values are spread uniformly across the range (complex/unpredictable signal). Low entropy = values are concentrated in few bins (simple/periodic signal). Useful for distinguishing regular rhythms from irregular ones.

- **`bins`** -- number of histogram buckets (more bins = finer granularity, but needs more samples per window)
- **Returns** -- entropy in bits (0 = constant signal, log2(bins) = maximally spread)

#### `rolling_skewness(data: &[f64], window: usize) -> Vec<f64>`

**Third Standardized Moment (Asymmetry).** Measures whether the distribution within each window leans to one side. Positive skew = right tail (occasional high spikes), negative skew = left tail (occasional dips), zero = symmetric. In clinical contexts, skewed vital-sign distributions often indicate intermittent events (e.g., occasional blood pressure spikes) superimposed on a stable baseline.

- **`window`** -- must be >= 3
- **Returns** -- dimensionless skewness value

#### `rolling_kurtosis(data: &[f64], window: usize) -> Vec<f64>`

**Excess Kurtosis (Fisher Definition).** Measures the "tailedness" of the distribution within each window. Kurtosis > 0 (excess) means heavier tails than a normal distribution -- outliers are more frequent. Kurtosis < 0 means lighter tails -- values are concentrated near the mean. High kurtosis in a vital sign can indicate intermittent extreme deviations (e.g., arrhythmia bursts within otherwise normal heart rate).

- **`window`** -- must be >= 4
- **Returns** -- excess kurtosis (0 = same tails as Gaussian, positive = heavier tails)

#### `trend_change_points(data: &[f64], window: usize) -> Vec<usize>`

**Trend Reversal Detector.** Runs `rolling_slope` and identifies indices where the slope sign flips (positive to negative or vice versa). Each index represents a point where the signal's direction changed. Useful for finding peaks, troughs, and inflection points in physiological waveforms without requiring the signal to cross a fixed threshold.

- **Returns** -- indices where the trend reversed

#### `trend_slope(values: &[f64]) -> f64`

**Global OLS Slope.** Fits a single straight line through the entire input using ordinary least squares. Returns the slope in units-per-sample. Useful as a quick summary of whether a vital sign is trending up, down, or flat over the entire observation window.

---

### Outlier and Artifact Detection

These functions identify corrupt samples, sensor disconnects, and transient artifacts before they contaminate downstream analytics. Each uses a different statistical model, so they can be combined for robust multi-method artifact rejection.

#### `detect_spikes_zscore(data: &[f64], window: usize, z_threshold: f64) -> Vec<usize>`

**Z-Score Spike Detector.** For each sample, computes the mean and standard deviation of the preceding `window` samples, then calculates how many standard deviations the current sample is from that local mean. If the absolute z-score exceeds `z_threshold`, the sample index is flagged. Assumes the "normal" signal is locally stationary and any large deviation is suspicious. Best for detecting sudden, isolated transients (motion artifacts, sensor glitches).

- **`window`** -- trailing reference window (must be >= 2 and < data length)
- **`z_threshold`** -- typically 2.5-4.0; higher = fewer detections, more conservative
- **Returns** -- indices of detected spikes

#### `mad_outlier_flags(data: &[f64], window: usize, threshold: f64) -> Vec<bool>`

**Median Absolute Deviation (MAD) Outlier Detector.** A robust alternative to z-score that uses the median and MAD instead of mean and standard deviation. The MAD is the median of absolute deviations from the median -- it measures typical spread without being influenced by the outliers themselves. Each sample gets a modified z-score: `0.6745 * |value - median| / MAD`. The 0.6745 constant scales MAD to be comparable with standard deviation under normality. Far more reliable than z-score when the signal itself contains many outliers.

- **`window`** -- must be >= 2
- **`threshold`** -- modified z-score cutoff, typically 3.0-3.5
- **Returns** -- boolean flag per sample (true = outlier)

#### `artifact_ratio(data: &[f64], threshold_abs_diff: f64) -> f64`

**Jump Fraction Metric.** Counts what fraction of consecutive sample-to-sample differences exceed an absolute threshold, then divides by the total number of intervals. A ratio of 0.0 means no artifacts; 1.0 means every sample transition is anomalous. Simple but effective for detecting sensor disconnects or motion artifacts that produce sudden, large jumps. The threshold should be set based on the physiological maximum rate of change for the signal being monitored.

- **`threshold_abs_diff`** -- maximum expected sample-to-sample change in the signal's units
- **Returns** -- fraction in [0, 1]

#### `signal_quality_score(data: &[f64]) -> SignalQualityScore`

**Composite Signal Quality Score.** Produces a single 0-1 quality metric combining three independent indicators:

| Component | What it measures | Weight |
|---|---|---|
| **Completeness** | Fraction of non-NaN samples | Direct multiplier |
| **Sampling stability** | Standard deviation of inter-sample intervals (low = regular sampling) | 0.85x penalty if irregular |
| **Outlier ratio** | Fraction flagged by MAD outlier detection | Reduces effective completeness |

The final score = `completeness * (1 - outlier_ratio) * stability_factor`. A score above 0.9 indicates a clean, reliable signal. Below 0.5 suggests significant data quality issues.

- **Returns** -- `SignalQualityScore { score, completeness, stable_sampling, outlier_ratio }`

#### `baseline_drift_score(data: &[f64], window: usize) -> f64`

**Head-vs-Tail Drift Metric.** Compares the mean of the first `window` samples to the mean of the last `window` samples, normalizes by the overall standard deviation. A score of 0 means no drift; higher values indicate progressive upward or downward shift. Useful for detecting slow sensor calibration drift or physiological trends that develop over the recording period. Requires at least `2 * window` samples.

- **`window`** -- must be <= half the data length
- **Returns** -- dimensionless drift magnitude (units of standard deviation)

#### `cusum_flags(data: &[f64], drift: f64, threshold: f64) -> Vec<bool>`

**Cumulative Sum Control Chart.** A classic statistical process control method that accumulates deviations from a running mean in both positive and negative directions. When either cumulative sum exceeds the threshold, a change point is flagged and the accumulators reset. The `drift` parameter acts as a tolerance band (similar to a dead zone) that prevents small fluctuations from accumulating. Particularly effective at detecting slow, sustained shifts that z-score methods might miss.

- **`drift`** -- minimum per-sample deviation that contributes to the cumulative sum (>= 0)
- **`threshold`** -- cumulative sum value that triggers a flag (> 0)
- **Returns** -- boolean flag per sample

#### `page_hinkley_flags(data: &[f64], delta: f64, threshold: f64) -> Vec<bool>`

**Page-Hinkley Change Detector.** Similar to CUSUM but tracks the running minimum of the cumulative sum and flags when the current value exceeds the minimum by more than `threshold`. This makes it more sensitive to one-directional shifts (monotonic changes) while being less prone to false alarms from oscillations. The `delta` parameter is the expected per-sample drift under normal conditions. Widely used in industrial process monitoring and adapts well to physiological signal monitoring.

- **`delta`** -- expected per-sample drift under null hypothesis (>= 0)
- **`threshold`** -- detection threshold (> 0)
- **Returns** -- boolean flag per sample

#### `ewma_residual_flags(data: &[f64], alpha: f64, threshold: f64) -> Vec<bool>`

**EWMA Residual Deviation Detector.** Maintains an exponentially weighted baseline estimate, computes the residual (difference between actual and baseline), and flags samples where the residual exceeds `threshold` times the running standard deviation of residuals. The `alpha` controls how quickly the baseline adapts to new data. This method naturally follows gradual trends (they get absorbed into the baseline) while flagging sudden departures. Useful for detecting acute events against a slowly varying baseline.

- **`alpha`** -- EWMA smoothing factor [0, 1]
- **`threshold`** -- number of residual standard deviations for flagging (> 0)
- **Returns** -- boolean flag per sample

---

### Signal Characterization

Higher-order descriptors of signal morphology. These go beyond basic statistics to capture shape, complexity, and dynamics -- useful for waveform classification, rhythm analysis, and comparing signal quality across patients or sensors.

#### `crossings_above_threshold(data: &[f64], threshold: f64) -> usize`

**Rising-Edge Threshold Crossing Counter.** Counts the number of times the signal crosses above the threshold from below. Each crossing represents a transition from "normal" to "elevated" territory. Useful for counting episodes: how many times did heart rate exceed 120? How many fever spikes above 38.5C? Only counts upward transitions (not downward), so it measures the number of distinct episodes, not the total time spent above threshold.

- **Returns** -- count of upward crossings

#### `time_above_threshold_ratio(data: &[f64], threshold: f64) -> f64`

**Fraction of Time Above Threshold.** Simply counts what fraction of samples exceed the threshold and divides by total length. Unlike `crossings_above_threshold` which counts episodes, this measures burden -- how much of the observation window was spent in the elevated zone. A ratio of 0.33 means the signal was above threshold for one-third of the time.

- **Returns** -- fraction in [0, 1]

#### `threshold_dwell_time(data: &[f64], threshold: f64) -> usize`

**Longest Contiguous Run Above Threshold.** Finds the maximum number of consecutive samples that all exceed the threshold. This captures episode duration: not just how many episodes or what fraction of time, but the longest single sustained event. A dwell time of 120 samples at 1 Hz means the signal stayed elevated for 2 consecutive minutes without dropping back below the threshold.

- **Returns** -- maximum run length in samples

#### `threshold_burden(data: &[f64], threshold: f64) -> f64`

**Cumulative Area Above Threshold.** Sums up `(value - threshold)` for every sample that exceeds the threshold. This combines both how often and how far the signal exceeds the boundary. A signal that barely crosses the threshold for many samples has a lower burden than one that shoots far above it for fewer samples. Analogous to the medical concept of "dose" -- not just whether a drug was taken, but how much and for how long.

- **Returns** -- total area (sum of excess values, in the signal's units)

#### `recovery_half_time(data: &[f64]) -> Option<usize>`

**Peak-to-Half-Recovery Distance.** Finds the global maximum in the signal, computes half that peak value, then scans forward to find the first sample at or below the half-peak level. Returns the distance in samples, or `None` if the signal never recovers to half-peak. Measures how quickly a physiological response returns to baseline after peaking -- a slow recovery can indicate impaired homeostatic regulation.

- **Returns** -- `Some(sample_count)` or `None`

#### `trend_stability_index(data: &[f64], window: usize) -> f64`

**Directional Persistence vs Noise Ratio.** Compares the average absolute OLS slope (from `rolling_slope`) to the average absolute sample-to-sample difference. If slopes are consistently large relative to noise, the signal has strong directional persistence (stability index near 1.0). If slopes are small relative to noise, the signal is oscillatory/random (near 0.0). A dropping trend stability index in a vital sign can signal loss of regulatory control.

- **Returns** -- value clamped to [0, 1]

#### `peak_to_peak_interval(data: &[f64]) -> Vec<usize>`

**Inter-Peak Distances.** Finds all local maxima (samples higher than both neighbors) and returns the sample distances between consecutive peaks. Essential for heart rate analysis: if peaks represent R-waves in an ECG, the intervals are RR intervals. Irregular intervals suggest arrhythmia; consistent intervals suggest normal sinus rhythm.

- **Returns** -- distances in samples between each pair of adjacent peaks

#### `spectral_flatness(data: &[f64], window: usize) -> Vec<f64>`

**Wiener Spectral Flatness per Window.** Computes the ratio of the geometric mean to the arithmetic mean of the (shifted-to-positive) samples within each window. A flatness near 1.0 means the signal resembles white noise (all frequency components present equally). A flatness near 0.0 means the signal is dominated by a few large values (tonal/spiky). In clinical use: flat, noisy sensor signals have high flatness; clean, periodic waveforms have low flatness.

- **Returns** -- flatness ratio per position, clamped to [0, 1]

#### `zero_crossing_rate(data: &[f64]) -> f64`

**Sign-Change Density.** Counts how often the signal crosses zero (changes sign between consecutive samples) divided by the total number of intervals. Directly related to the dominant frequency of an oscillating signal: a signal that crosses zero twice per cycle has a zero-crossing rate proportional to its frequency. Useful as a lightweight frequency estimator that works on very short signal segments where FFT would be unreliable.

- **Returns** -- rate in [0, 1] (1.0 = alternates sign every sample)

#### `fractal_dimension(data: &[f64], k_max: usize) -> f64`

**Higuchi Fractal Dimension.** Estimates the fractal dimension of the signal using the Higuchi algorithm: the signal is reconstructed at multiple sub-sampling scales (`k = 2` to `k_max`), the average curve length is computed at each scale, and the fractal dimension is estimated as the negative slope of `log(length)` vs `log(k)`. A dimension near 1.0 means a smooth, predictable curve; higher values (approaching 2.0) mean rough, complex, unpredictable behavior. HRV signals with higher fractal dimensions are associated with certain cardiac pathologies.

- **`k_max`** -- maximum sub-sampling scale (must be >= 2 and < data length)
- **Returns** -- fractal dimension estimate (typically 1.0 to 2.0)

#### `lagged_autocorrelation(data: &[f64], lag: usize) -> f64`

**Autocorrelation at Configurable Lag.** Computes the normalized cross-correlation of the signal with a time-shifted copy of itself at the specified lag. At lag 0 the autocorrelation is always 1.0 (a signal is perfectly correlated with itself). At the signal's dominant period, autocorrelation peaks near 1.0 again. Useful for detecting periodicity: scan multiple lags and find where autocorrelation peaks to estimate the period.

- **`lag`** -- number of samples to shift (0 = self-correlation = 1.0)
- **Returns** -- correlation coefficient in [-1, 1]

#### `dominant_frequency(data: &[f64]) -> f64`

**Zero-Crossing-Based Frequency Estimate.** Demeans the signal, computes autocorrelation at all lags up to half the signal length, and returns `1.0 / best_lag` where `best_lag` is the lag with highest autocorrelation. This gives the dominant periodic frequency in cycles per sample. For a signal sampled at 100 Hz with a dominant frequency of 0.25 cycles/sample, the physical frequency is 25 Hz.

- **Returns** -- frequency in cycles/sample (e.g., 0.25 = one cycle every 4 samples)

#### `signal_to_noise_ratio(data: &[f64]) -> f64`

**SNR in Decibels.** Estimates signal-to-noise ratio as `20 * log10(|mean| / std)`. The mean represents the "signal" level and the standard deviation represents the "noise." A constant signal has infinite SNR; a zero-mean noisy signal has negative SNR. Useful for comparing signal quality across different sensors or monitoring conditions.

- **Returns** -- SNR in dB (higher = cleaner signal)

#### `signal_energy(data: &[f64]) -> f64`

**Total Signal Energy.** Computes the sum of squared sample values: `sum(x_i^2)`. Energy captures both the amplitude and duration of a signal's activity -- a sustained moderate signal can have the same energy as a brief intense one. Used as a baseline for comparing different signal segments or as a feature in classification pipelines.

- **Returns** -- energy in squared units of the input

#### `rms_power(data: &[f64]) -> f64`

**Root Mean Square Power.** Computes `sqrt(mean(x_i^2))` -- the square root of signal energy divided by sample count. RMS represents the effective amplitude of the signal, accounting for both positive and negative excursions. Widely used in ECG analysis to quantify overall signal magnitude independent of waveform shape.

- **Returns** -- RMS value in the same units as input

#### `moving_average_crossover(data: &[f64], fast_window: usize, slow_window: usize) -> Vec<MACross>`

**Fast/Slow MA Crossover Events.** Computes two moving averages with different window sizes and detects where they cross. When the fast MA crosses above the slow MA, that's a `Bullish` signal (uptrend starting). When the fast MA crosses below, that's `Bearish` (downtrend starting). The crossover lag means this is a lagging indicator -- it confirms trends rather than predicting them. Commonly used in vital-sign trend monitoring to detect sustained directional changes while filtering out transient noise.

- **`fast_window`** -- must be >= 2 and < slow_window
- **`slow_window`** -- must be >= 2 and <= data length
- **Returns** -- sequence of `MACross::Bullish` or `MACross::Bearish` events

#### `band_energy(data: &[f64]) -> f64`

**Frequency-Weighted Signal Energy.** Computes total signal energy (sum of squared values after mean removal) weighted by the zero-crossing rate. High-frequency oscillating signals get higher scores than flat or slowly-varying signals with the same energy. Useful for distinguishing active signal content (oscillations, heartbeats) from baseline drift or DC offset.

- **Returns** -- weighted energy value

#### `waveform_symmetry(data: &[f64]) -> f64`

**Waveform Symmetry Ratio.** Compares the left and right halves of the signal around the central sample. Computes the ratio of absolute differences to absolute values between mirrored samples, subtracted from 1.0. A perfectly symmetric waveform (like a sine wave or a symmetric triangle) scores near 1.0; a heavily asymmetric waveform scores near 0.0. In clinical use, many normal physiological waveforms have characteristic symmetry profiles -- deviations can indicate pathology.

- **Returns** -- symmetry ratio in [0, 1]

#### `jitter_score(peak_indices: &[usize]) -> f64`

**Peak Timing Jitter.** Given a sequence of peak indices (e.g., from R-wave detection), computes the coefficient of variation of the inter-peak intervals: `std(intervals) / mean(intervals)`. Low jitter (< 0.05) indicates a regular rhythm; high jitter (> 0.1) indicates timing irregularity. In cardiology, high RR-interval jitter is associated with arrhythmias and autonomic dysfunction.

- **Returns** -- dimensionless jitter ratio (lower = more regular)

#### `peak_prominence(data: &[f64]) -> Vec<(usize, f64)>`

**Peak Height Above Surrounding Minima.** For each local maximum, finds the lowest point to the left and the lowest point to the right, then computes how much the peak rises above the higher of those two minima. Prominence measures how "stand-out" a peak is regardless of absolute amplitude -- a small peak in a flat region can have high prominence, while a large peak between even larger peaks can have low prominence. Returns both the peak index and its prominence value.

- **Returns** -- list of (peak_index, prominence) pairs

#### `signal_saturation_ratio(data: &[f64], lo: f64, hi: f64) -> f64`

**Clipping / Saturation Fraction.** Counts what fraction of samples fall at or beyond the signal's valid range bounds (`lo` and `hi`). A high saturation ratio indicates sensor clipping, ADC saturation, or signal loss -- the sensor is hitting its measurement limits and can't represent the true physiological value. Even a few percent saturation can seriously bias downstream analytics.

- **`lo`** -- lower bound of valid range (must be < hi)
- **`hi`** -- upper bound of valid range
- **Returns** -- fraction in [0, 1]

---

### Heart Rate Variability

Time-domain HRV metrics computed from RR interval sequences (in milliseconds). These are standard measures used in cardiology to assess autonomic nervous system function.

#### `hrv_rmssd(rr_intervals_ms: &[f64]) -> f64`

**Root Mean Square of Successive Differences.** The most commonly used short-term HRV metric. Computes `sqrt(mean((RR_i - RR_{i-1})^2))`. RMSSD reflects parasympathetic (vagal) activity -- higher values indicate stronger vagal tone and better autonomic flexibility. Low RMSSD is associated with stress, fatigue, and cardiac risk. Values are in milliseconds; typical healthy range is 20-60 ms.

- **Input** -- RR intervals in milliseconds (must be positive, at least 2)
- **Returns** -- RMSSD in ms

#### `hrv_sdnn(rr_intervals_ms: &[f64]) -> f64`

**Standard Deviation of NN Intervals.** The broadest time-domain HRV measure, reflecting both sympathetic and parasympathetic influences. Computes the population standard deviation of all RR intervals. Longer recordings give more meaningful SDNN values. In clinical practice, SDNN < 50 ms suggests reduced autonomic modulation; > 100 ms suggests healthy variability.

- **Input** -- RR intervals in milliseconds (must be positive, at least 2)
- **Returns** -- SDNN in ms

#### `rr_interval_variability(rr_intervals_ms: &[f64], cv_threshold: f64) -> RRVariability`

**Regular vs Irregular Rhythm Classification.** Computes the coefficient of variation (CV = std/mean) of RR intervals and classifies the rhythm as `Regular` (CV <= threshold) or `Irregular` (CV > threshold). A simple binary classifier for rhythm regularity. The threshold should be tuned for the clinical context; 0.10 (10%) is a common starting point.

- **`cv_threshold`** -- CV cutoff for classification (must be > 0)
- **Returns** -- `RRVariability::Regular` or `RRVariability::Irregular`

---

### Gap Repair + Signal Transforms

#### `interpolate_nan_gaps(data: &[f64], max_gap: usize) -> Vec<f64>`

**Bounded-Gap Linear Interpolation.** Scans for runs of NaN values and fills them with linear interpolation between the surrounding non-NaN values, but only if the gap length is <= `max_gap`. Longer gaps are left as NaN to avoid fabricating data. This preserves the integrity of the signal while repairing short dropouts from temporary sensor disconnection or transmission errors.

- **`max_gap`** -- maximum number of consecutive NaNs to fill (0 = don't fill any)
- **Returns** -- repaired signal (same length)

#### `derivative(data: &[f64]) -> Vec<f64>`

**First Derivative (Finite Differences).** Estimates the rate of change at each point using central differences for interior points (more accurate) and forward/backward differences at the endpoints. The derivative reveals where the signal is changing fastest -- peaks in the derivative correspond to the steepest rising/falling edges of the original waveform. Useful for detecting onset times and measuring response latencies.

- **Input** -- must be >= 2 samples
- **Returns** -- rate of change per sample (in input units per sample)

#### `second_derivative(data: &[f64]) -> Vec<f64>`

**Second Derivative (Curvature/Acceleration).** Estimates the rate of change of the rate of change using the discrete Laplacian: `x[i+1] - 2*x[i] + x[i-1]`. Positive second derivative = concave up (accelerating upward), negative = concave down (decelerating). Zero crossings of the second derivative correspond to inflection points where the signal changes curvature. Useful for finding exact peak locations (second derivative crosses zero at the peak).

- **Input** -- must be >= 3 samples
- **Returns** -- curvature values

#### `resample_linear(data: &[f64], n: usize) -> Vec<f64>`

**Linear Interpolation Resampling.** Resamples the input to a target length `n` using linear interpolation. The first and last output values always match the first and last input values. Useful for normalizing signals to a common length before feature extraction or comparison, regardless of their original sample count.

- **`n`** -- target output length (must be >= 2)
- **Returns** -- resampled signal of exactly `n` samples

#### `rate_of_change(data: &[f64]) -> Vec<f64>`

**Percentage Rate of Change.** Computes the percentage change between consecutive samples: `((x[i] - x[i-1]) / |x[i-1]|) * 100`. The first sample always gets 0.0 (no previous value to compare against). Useful for tracking how rapidly a vital sign is changing in percentage terms, independent of its absolute level.

- **Returns** -- percentage changes (0.0 for first sample)

#### `moving_percentile(data: &[f64], window: usize, percentile: f64) -> Vec<f64>`

**Arbitrary Rolling Percentile.** Computes the value at a given percentile rank within each rolling window using linear interpolation between sorted values. Percentile 50 = median, 25 = first quartile, 75 = third quartile, 95 = 95th percentile. More flexible than `rolling_median` which is fixed at the 50th percentile. Useful for establishing dynamic confidence bands (e.g., P5-P95 range) around a vital sign.

- **`percentile`** -- target percentile in [0, 100]
- **Returns** -- percentile values per position

---

### Population Comparison

#### `vital_sign_zscore(data: &[f64], pop_mean: f64, pop_std: f64) -> Vec<f64>`

**Patient-Relative Z-Scores.** Translates each sample into the number of standard deviations it deviates from a reference population mean. A z-score of +2.0 means the value is 2 standard deviations above the population average. This enables comparing a patient's vitals against established norms rather than absolute thresholds. Requires a known population mean and standard deviation (from clinical literature or a reference dataset).

- **`pop_mean`** -- reference population mean
- **`pop_std`** -- reference population standard deviation (must be > 0)
- **Returns** -- z-scores (positive = above average, negative = below)

---

### Clinical Risk Scoring

Deterioration detection inspired by the UK's National Early Warning Score (NEWS2) and early warning score systems. These functions transform raw vital-sign measurements into actionable clinical risk indicators.

#### `deterioration_flags(hr, sbp, spo2, rr) -> DeteriorationFlags`

**Snapshot Deterioration Detector.** Evaluates four vital signs against clinical thresholds and produces binary flags plus a risk count:

| Flag | Threshold |
|---|---|
| `tachycardia` | Heart rate > 100 bpm |
| `hypotension` | Systolic BP < 90 mmHg |
| `hypoxemia` | SpO2 < 92% |
| `tachypnea` | Respiratory rate > 22 breaths/min |

The `risk_score` (0-4) counts how many flags are active. `high_risk` is true when 2+ flags are active simultaneously. This is a simplified model based on the "qSOFA" rapid sepsis assessment criteria.

#### `deterioration_trend_flags(hr, sbp, spo2, rr, window) -> TrendDeteriorationFlags`

**Trend-Based Deterioration Detector.** Takes time-series of all four vital signs and analyzes whether each is trending in a clinically concerning direction over the last `window` samples using OLS linear regression:

| Flag | Concerning direction |
|---|---|
| `rising_heart_rate` | Positive HR slope |
| `falling_systolic_bp` | Negative SBP slope |
| `falling_spo2` | Negative SpO2 slope |
| `rising_respiratory_rate` | Positive RR slope |

`high_risk` when 3+ trends are concerning simultaneously. This catches gradual deterioration that snapshot methods miss.

#### `news2_lite_score(hr, rr, spo2, sbp, temp, consciousness) -> News2LiteScore`

**NEWS2-Lite Composite Score.** A simplified version of the UK Royal College of Physicians' NEWS2 scoring system. Each vital sign is scored 0-3 based on how far it deviates from the normal range, then summed into a total. The scoring table:

| Parameter | Score 0 (normal) | Score 3 (critical) |
|---|---|---|
| Heart rate | 51-90 | <= 40 or >= 131 |
| Respiratory rate | 12-20 | <= 8 or >= 25 |
| SpO2 | >= 96 | <= 91 |
| Systolic BP | 111-219 | <= 90 or >= 220 |
| Temperature | 36.1-38.0 | <= 35.0 |
| Consciousness | Alert | New confusion |

`high_risk` is flagged when: total >= 7, any single parameter scores 3, or altered consciousness is present.

#### `risk_summary(hr, sbp, spo2, rr, temp, consciousness, trend_window) -> RiskSummary`

**Combined Risk Aggregator.** Runs all three scoring methods (snapshot deterioration, trend deterioration, NEWS2-lite) in one call and combines their results:

- `snapshot_score` -- from `deterioration_flags` (0-4)
- `trend_score` -- from `deterioration_trend_flags` (0-4)
- `news2_score` -- from `news2_lite_score` (0-18)
- `total_score` -- sum of all three
- `high_risk` -- true if any individual method flags high risk, or total >= 8

#### `early_warning_window(high_risk_windows: &[bool], min_consecutive: usize) -> EarlyWarningWindow`

**Sustained Alert Detector.** Takes a sequence of boolean risk flags (one per observation window) and checks whether `min_consecutive` or more high-risk windows occurred back-to-back. This filters out brief transient alerts and only triggers when risk is sustained -- reducing alarm fatigue while maintaining sensitivity to genuine deterioration.

- **`min_consecutive`** -- number of consecutive high-risk windows required for alert (must be >= 1)
- **Returns** -- `should_alert` (bool) + `sustained_windows` (longest consecutive run)

#### `patient_state_transition(previous_risk, current_risk, delta_threshold) -> PatientState`

**Risk Trajectory Classifier.** Compares two risk scores (e.g., from consecutive time windows) and classifies the patient's trajectory:

| State | Condition |
|---|---|
| `Improving` | Risk dropped by >= delta_threshold |
| `Worsening` | Risk rose by >= delta_threshold |
| `Stable` | Change is within +/- delta_threshold |

The delta threshold lets you tune sensitivity: a threshold of 1 means any single-point change triggers a state change; a threshold of 3 requires more dramatic shifts.

### Error Handling

Every public function returns `Result<_, SignalError>` with one of:

| Variant | Meaning |
|---|---|
| `EmptyInput` | Empty slice provided |
| `InvalidAlpha` | EWMA alpha outside [0, 1] |
| `InvalidWindow` | Window size 0, too large, or too small |
| `InvalidThreshold` | Non-finite or non-positive threshold |
| `InvalidLength` | Not enough samples for the operation |
| `InvalidRange` | Clinically impossible value (e.g. SpO2 > 100%) |
| `NonFiniteValue` | NaN or Inf in input data |

## Quick Start

```bash
# Prerequisites: Rust toolchain (rustup.rs)
# Build everything
cargo build --workspace

# Run the full test suite (113 tests, 0 external dependencies)
cargo test --workspace

# Or via npm
npm test
```

The `npm test` script maps to `cargo test --workspace`. No Node runtime is needed for the Rust crate itself — the `package.json` exists for CI convenience and future WASM bindings.

## Example (Rust)

```rust
use clinical_signal_core::{
    crossings_above_threshold,
    derivative,
    dominant_frequency,
    fractal_dimension,
    news2_lite_score,
    peak_to_peak_interval,
    resample_linear,
    risk_summary,
    rolling_cv,
    rolling_entropy,
    signal_saturation_ratio,
    spectral_flatness,
    zero_crossing_rate,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Patient vitals time series ---
    let hr  = [82.0, 88.0, 95.0, 102.0];
    let sbp = [128.0, 120.0, 112.0, 104.0];
    let spo2 = [98.0, 96.0, 94.0, 92.0];
    let rr  = [16.0, 18.0, 20.0, 23.0];

    // --- Composite risk scoring ---
    let summary = risk_summary(&hr, &sbp, &spo2, &rr, 38.1, false, 4)?;
    println!("Risk: snapshot={} trend={} news2={} total={} high_risk={}",
        summary.snapshot_score, summary.trend_score,
        summary.news2_score, summary.total_score, summary.high_risk);

    // --- Signal characterization ---
    let waveform = [1.0, 2.0, 5.0, 3.0, 1.0, 4.0, 7.0, 4.0, 1.0];
    let intervals = peak_to_peak_interval(&waveform)?;
    let fd = fractal_dimension(&waveform, 3)?;
    let zcr = zero_crossing_rate(&[-1.0, 1.0, -1.0, 1.0])?;
    let sf = spectral_flatness(&[10.0, 11.0, 10.0, 11.0], 3)?;
    let ent = rolling_entropy(&[1.0, 2.0, 3.0, 4.0], 4, 4)?;
    let d = derivative(&waveform)?;
    let resampled = resample_linear(&waveform, 20)?;
    let freq = dominant_frequency(&[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])?;
    let sat = signal_saturation_ratio(&[0.0, 0.0, 5.0, 10.0, 10.0], 0.0, 10.0)?;

    println!("Peak intervals: {:?}", intervals);
    println!("Fractal dim:    {:.3}", fd);
    println!("Zero-cross rate: {:.2}", zcr);
    println!("Spectral flat:   {:?}", sf);
    println!("Entropy (last):  {:.3}", ent.last().unwrap_or(&f64::NAN));
    println!("Derivative:      {:?}", d);
    println!("Resampled len:   {}", resampled.len());
    println!("Dominant freq:   {:.3}", freq);
    println!("Saturation:      {:.2}", sat);

    Ok(())
}
```

## Engineering Principles

1. **Clean-room implementations** — every module is written from spec, not copied from external sources
2. **TDD-first** — tests are written before implementation; `cargo test --workspace` must pass before any batch lands
3. **Minimal API surface** — flat functions, no trait hierarchies, no generic soup
4. **Explicit failure modes** — invalid clinical ranges and non-finite inputs always return `Err`, never panic
5. **Deterministic** — same input always produces the same output; suitable for auditable clinical workflows
6. **Zero ML dependencies** — no PyTorch, no ONNX, no tensor crates — just Rust stdlib math

## Architecture

```
clinical-signal-core/src/
  lib.rs          Module declarations + re-exports + 113 integration tests
  types.rs        SignalError, DeteriorationFlags, News2LiteScore, RiskSummary, PatientState, etc.
  rolling.rs      EWMA, rolling mean/variance/median/min/max/range/slope/cv/entropy/skewness/kurtosis/percentile
  outlier.rs      Z-score spikes, MAD outlier flags, CUSUM, Page-Hinkley, EWMA residuals, artifact ratio, baseline drift
  features.rs     Threshold analysis, peak detection, spectral flatness, zero-crossing rate, fractal dimension,
                  autocorrelation, dominant frequency, MA crossover, band energy, waveform symmetry, jitter, prominence
  clinical.rs     HRV (RMSSD, SDNN), RR variability, deterioration flags, NEWS2-lite, risk summary, early warning,
                  patient state transitions, signal quality scoring
  transform.rs    Derivatives, resampling, interpolation, rate of change, signal energy/RMS/SNR, z-scores
  kalman.rs       1D Kalman filter (configurable), Kalman baseline estimator, Kalman residual detector
  streaming.rs    Stateful streaming processors: EWMA, running mean, Welford variance, CUSUM
clinical-signal-core/benches/
  benchmarks.rs   Criterion benchmarks for all hot paths (EWMA, rolling, spikes, MAD, HRV, Kalman, streaming)
```

### Design Decisions

**Why flat functions instead of a builder or trait system?** Every function takes `&[f64]` or primitive values and returns `Result<_, SignalError>`. This keeps the API surface trivially FFI-friendly (for WASM or C bindings) and avoids the trait resolution complexity that slows compilation and confuses newcomers.

**Why `Result` everywhere instead of panics?** Clinical software must never crash on bad input. A corrupted sensor reading or a firmware bug producing `NaN` should produce a recoverable error, not a runtime panic. Every function validates inputs before processing.

**Why population variance instead of sample variance?** Rolling windows typically cover a fixed, known number of samples (not a random sample from an infinite population). Population variance (`/n`) is the correct choice for signal processing contexts.

**Why no FFT?** FFT requires either a heavy external dependency or a complex hand-rolled implementation. The frequency-domain features here (dominant frequency, spectral flatness, band energy) use zero-crossing and geometric/arithmetic mean approximations that are good enough for real-time clinical screening and work on arbitrarily short signals.

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Core library | Rust (edition 2021) | All signal processing and clinical logic |
| Build system | Cargo | Workspace with single crate |
| Testing | `#[test]` + `cargo test` | 113 unit/integration tests, no test framework dependency |
| CI script | `package.json` | `npm test` maps to `cargo test --workspace` |
| Future | `wasm-pack` | WASM bindings for browser/Edge deployment |

### Zero External Dependencies

The crate compiles with zero crates from crates.io. Everything is implemented using Rust's `std` library alone:
- `f64` arithmetic for all signal processing
- `Vec` for output buffers
- `Result` for error propagation
- No unsafe code, no FFI, no platform-specific code

## Test Coverage

```
running 113 tests ... ok. 113 passed; 0 failed; 0 ignored; 0 measured
```

Every public function has at minimum:
- A **happy-path** test with known expected values
- An **edge-case** test (empty input, boundary values)
- An **error-path** test verifying the correct `SignalError` variant

Tests are organized chronologically by batch number, making it easy to trace which tests cover which feature wave.

## Roadmap

| Phase | Theme | Status |
|---|---|---|
| 1 | Rolling window primitives + input validation | Done |
| 2 | Outlier/artifact detection + gap repair | Done |
| 3 | Threshold analysis + recovery features | Done |
| 4 | Signal complexity (fractal, entropy, spectral) | Done |
| 5 | Autocorrelation, frequency, SNR, HRV classification, z-scores, moments | Done |
| 6 | Derivatives, resampling, percentiles, band energy, symmetry, jitter, prominence | Done |
| 7 | Kalman-filter baseline tracking | Done |
| 8 | Streaming / chunked API for real-time bedside use | Done |
| 10 | Benchmark harness for reproducible perf regression tracking | Done |

**Phase 9 (WASM bindings + browser demo)** is the only remaining planned phase and requires `wasm-pack` tooling setup.

## Status

Active development. Shipped in **batched TDD passes** (5-10 features per batch, tests written red then implemented green). Currently at **63 public functions** across **9 source modules** with **127 passing tests** and a **criterion benchmark harness** covering hot paths.
