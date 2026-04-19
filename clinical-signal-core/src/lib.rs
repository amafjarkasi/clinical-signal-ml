pub mod types;
pub mod rolling;
pub mod outlier;
pub mod features;
pub mod clinical;
pub mod transform;

// Re-export all public types
pub use types::*;
pub use rolling::*;
pub use outlier::*;
pub use features::*;
pub use clinical::*;
pub use transform::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_smooths_data() {
        let values = [10.0, 20.0, 10.0];
        let out = ewma(&values, 0.5).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 10.0).abs() < 1e-12);
        assert!((out[1] - 15.0).abs() < 1e-12);
        assert!((out[2] - 12.5).abs() < 1e-12);
    }

    #[test]
    fn rolling_mean_uses_full_windows() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let out = rolling_mean(&values, 3).unwrap();
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!((out[2] - 2.0).abs() < 1e-12);
        assert!((out[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn zscore_detects_single_spike() {
        let values = [100.0, 101.0, 100.5, 100.8, 140.0, 100.9];
        let spikes = detect_spikes_zscore(&values, 4, 3.0).unwrap();
        assert_eq!(spikes, vec![4]);
    }

    #[test]
    fn rolling_variance_matches_expected_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let out = rolling_variance(&values, 2).unwrap();
        assert!(out[0].is_nan());
        assert!((out[1] - 0.25).abs() < 1e-12);
        assert!((out[2] - 0.25).abs() < 1e-12);
        assert!((out[3] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(ewma(&[], 0.3).unwrap_err(), SignalError::EmptyInput);
        assert_eq!(ewma(&[1.0], -0.1).unwrap_err(), SignalError::InvalidAlpha);
        assert_eq!(rolling_mean(&[1.0, 2.0], 0).unwrap_err(), SignalError::InvalidWindow);
        assert_eq!(
            detect_spikes_zscore(&[1.0, 2.0, 3.0], 2, 0.0).unwrap_err(),
            SignalError::InvalidThreshold
        );
    }

    #[test]
    fn hrv_rmssd_matches_expected_value() {
        let rr = [800.0, 810.0, 790.0, 805.0];
        let rmssd = hrv_rmssd(&rr).unwrap();
        assert!((rmssd - 15.5456).abs() < 1e-3);
    }

    #[test]
    fn hrv_sdnn_matches_expected_value() {
        let rr = [800.0, 820.0, 780.0, 810.0, 790.0];
        let sdnn = hrv_sdnn(&rr).unwrap();
        assert!((sdnn - 14.1421).abs() < 1e-3);
    }

    #[test]
    fn trend_slope_detects_upward_signal() {
        let values = [97.8, 98.0, 98.2, 98.4, 98.6];
        let slope = trend_slope(&values).unwrap();
        assert!(slope > 0.0);
        assert!((slope - 0.2).abs() < 1e-9);
    }

    #[test]
    fn deterioration_flags_identify_high_risk_state() {
        let flags = deterioration_flags(115.0, 85.0, 90.0, 24.0).unwrap();
        assert!(flags.tachycardia);
        assert!(flags.hypotension);
        assert!(flags.hypoxemia);
        assert!(flags.tachypnea);
        assert_eq!(flags.risk_score, 4);
        assert!(flags.high_risk);
    }

    #[test]
    fn deterioration_flags_rejects_impossible_spo2() {
        let err = deterioration_flags(80.0, 120.0, 105.0, 16.0).unwrap_err();
        assert_eq!(err, SignalError::InvalidRange);
    }

    #[test]
    fn news2_lite_scores_high_risk_case() {
        let result = news2_lite_score(130.0, 26.0, 88.0, 85.0, 39.4, true).unwrap();
        assert_eq!(result.score, 16);
        assert!(result.high_risk);
    }

    #[test]
    fn news2_lite_scores_low_risk_case() {
        let result = news2_lite_score(72.0, 16.0, 98.0, 122.0, 36.8, false).unwrap();
        assert_eq!(result.score, 0);
        assert!(!result.high_risk);
    }

    #[test]
    fn news2_lite_rejects_non_finite_input() {
        let err = news2_lite_score(f64::NAN, 16.0, 98.0, 122.0, 36.8, false).unwrap_err();
        assert_eq!(err, SignalError::NonFiniteValue);
    }

    #[test]
    fn rolling_median_handles_spike_robustly() {
        let values = [1.0, 1.0, 1.0, 50.0, 1.0];
        let med = rolling_median(&values, 3).unwrap();
        assert!(med[0].is_nan());
        assert!(med[1].is_nan());
        assert!((med[2] - 1.0).abs() < 1e-12);
        assert!((med[3] - 1.0).abs() < 1e-12);
        assert!((med[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mad_outlier_flags_marks_single_anomaly() {
        let values = [10.0, 10.0, 10.0, 30.0, 10.0, 10.0];
        let flags = mad_outlier_flags(&values, 4, 3.5).unwrap();
        assert_eq!(flags.len(), values.len());
        assert!(!flags[0] && !flags[1] && !flags[2]);
        assert!(flags[3]);
        assert!(!flags[4] && !flags[5]);
    }

    #[test]
    fn deterioration_trend_flags_detect_worsening_pattern() {
        let hr = [82.0, 88.0, 95.0, 102.0];
        let sbp = [128.0, 120.0, 112.0, 104.0];
        let spo2 = [98.0, 96.0, 94.0, 92.0];
        let rr = [16.0, 18.0, 20.0, 23.0];

        let flags = deterioration_trend_flags(&hr, &sbp, &spo2, &rr, 4).unwrap();
        assert!(flags.rising_heart_rate);
        assert!(flags.falling_systolic_bp);
        assert!(flags.falling_spo2);
        assert!(flags.rising_respiratory_rate);
        assert_eq!(flags.risk_score, 4);
        assert!(flags.high_risk);
    }

    #[test]
    fn deterioration_trend_flags_rejects_mismatched_lengths() {
        let hr = [82.0, 88.0, 95.0];
        let sbp = [128.0, 120.0, 112.0, 104.0];
        let spo2 = [98.0, 96.0, 94.0, 92.0];
        let rr = [16.0, 18.0, 20.0, 23.0];
        let err = deterioration_trend_flags(&hr, &sbp, &spo2, &rr, 3).unwrap_err();
        assert_eq!(err, SignalError::InvalidLength);
    }

    #[test]
    fn deterioration_trend_flags_rejects_invalid_spo2_range() {
        let hr = [82.0, 88.0, 95.0, 102.0];
        let sbp = [128.0, 120.0, 112.0, 104.0];
        let spo2 = [98.0, 96.0, 101.0, 92.0];
        let rr = [16.0, 18.0, 20.0, 23.0];
        let err = deterioration_trend_flags(&hr, &sbp, &spo2, &rr, 4).unwrap_err();
        assert_eq!(err, SignalError::InvalidRange);
    }

    #[test]
    fn news2_lite_rejects_impossible_spo2() {
        let err = news2_lite_score(80.0, 16.0, 101.0, 122.0, 36.8, false).unwrap_err();
        assert_eq!(err, SignalError::InvalidRange);
    }

    #[test]
    fn interpolate_nan_gaps_fills_short_gap() {
        let values = [1.0, f64::NAN, 3.0];
        let out = interpolate_nan_gaps(&values, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn interpolate_nan_gaps_leaves_long_gap() {
        let values = [1.0, f64::NAN, f64::NAN, 4.0];
        let out = interpolate_nan_gaps(&values, 1).unwrap();
        assert!(out[1].is_nan());
        assert!(out[2].is_nan());
    }

    #[test]
    fn interpolate_nan_gaps_rejects_infinite_values() {
        let values = [1.0, f64::INFINITY, 3.0];
        let err = interpolate_nan_gaps(&values, 1).unwrap_err();
        assert_eq!(err, SignalError::NonFiniteValue);
    }

    #[test]
    fn signal_quality_score_reports_expected_ranges() {
        let values = [1.0, 1.0, 1.0, 20.0, 1.0, f64::NAN];
        let q = signal_quality_score(&values).unwrap();
        assert!((q.completeness - (5.0 / 6.0)).abs() < 1e-12);
        assert!(q.outlier_ratio > 0.0);
        assert!(q.score >= 0.0 && q.score <= 1.0);
    }

    #[test]
    fn risk_summary_combines_snapshot_trend_news2() {
        let hr = [82.0, 88.0, 95.0, 102.0];
        let sbp = [128.0, 120.0, 112.0, 104.0];
        let spo2 = [98.0, 96.0, 94.0, 92.0];
        let rr = [16.0, 18.0, 20.0, 23.0];
        let summary = risk_summary(&hr, &sbp, &spo2, &rr, 38.1, false, 4).unwrap();
        assert_eq!(summary.snapshot_score, 2);
        assert_eq!(summary.trend_score, 4);
        assert_eq!(summary.news2_score, 7);
        assert_eq!(summary.total_score, 13);
        assert!(summary.high_risk);
    }

    #[test]
    fn early_warning_window_requires_sustained_high_risk() {
        let risks = [false, true, true, true];
        let out = early_warning_window(&risks, 3).unwrap();
        assert!(out.should_alert);
        assert_eq!(out.sustained_windows, 3);
    }

    #[test]
    fn baseline_drift_score_detects_shift() {
        let values = [1.0, 1.1, 1.0, 1.0, 2.0, 2.1, 2.0, 2.0];
        let score = baseline_drift_score(&values, 4).unwrap();
        assert!(score > 1.9);
    }

    #[test]
    fn artifact_ratio_counts_jumps() {
        let values = [10.0, 10.2, 35.0, 35.1, 10.0];
        let ratio = artifact_ratio(&values, 5.0).unwrap();
        assert!((ratio - 0.5).abs() < 1e-12);
    }

    #[test]
    fn patient_state_transition_classifies_direction() {
        let up = patient_state_transition(3, 7, 2).unwrap();
        let down = patient_state_transition(7, 3, 2).unwrap();
        let flat = patient_state_transition(7, 8, 2).unwrap();
        assert_eq!(up, PatientState::Worsening);
        assert_eq!(down, PatientState::Improving);
        assert_eq!(flat, PatientState::Stable);
    }

    #[test]
    fn rolling_min_and_max_track_bounds() {
        let values = [3.0, 1.0, 4.0, 2.0];
        let min_out = rolling_min(&values, 2).unwrap();
        let max_out = rolling_max(&values, 2).unwrap();
        assert!(min_out[0].is_nan() && max_out[0].is_nan());
        assert!((min_out[1] - 1.0).abs() < 1e-12);
        assert!((max_out[1] - 3.0).abs() < 1e-12);
        assert!((min_out[3] - 2.0).abs() < 1e-12);
        assert!((max_out[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_range_computes_spread() {
        let values = [3.0, 1.0, 4.0, 2.0];
        let out = rolling_range(&values, 2).unwrap();
        assert!(out[0].is_nan());
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_slope_detects_local_trend() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let slopes = rolling_slope(&values, 3).unwrap();
        assert!(slopes[0].is_nan() && slopes[1].is_nan());
        assert!((slopes[2] - 1.0).abs() < 1e-12);
        assert!((slopes[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trend_change_points_detects_sign_flip() {
        let values = [1.0, 2.0, 3.0, 2.0, 1.0];
        let cps = trend_change_points(&values, 3).unwrap();
        assert_eq!(cps, vec![3]);
    }

    #[test]
    fn crossings_above_threshold_counts_rising_edges() {
        let values = [95.0, 96.0, 101.0, 102.0, 99.0, 103.0];
        let count = crossings_above_threshold(&values, 100.0).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn time_above_threshold_ratio_matches_fraction() {
        let values = [95.0, 101.0, 102.0, 99.0, 103.0, 97.0];
        let ratio = time_above_threshold_ratio(&values, 100.0).unwrap();
        assert!((ratio - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cusum_flags_detect_mean_shift() {
        let values = [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 2.0, 2.2, 2.1, 2.0];
        let flags = cusum_flags(&values, 0.05, 1.5).unwrap();
        assert!(flags.iter().any(|v| *v));
    }

    #[test]
    fn page_hinkley_flags_detect_mean_shift() {
        let values = [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 2.0, 2.2, 2.1, 2.0];
        let flags = page_hinkley_flags(&values, 0.05, 1.5).unwrap();
        assert!(flags.iter().any(|v| *v));
    }

    #[test]
    fn ewma_residual_flags_detect_spike() {
        let values = [1.0, 1.0, 1.0, 10.0, 1.0];
        let flags = ewma_residual_flags(&values, 0.3, 2.0).unwrap();
        assert!(!flags[2]);
        assert!(flags[3]);
    }

    // ── Batch 3: threshold / recovery / CV / trend stability ──

    #[test]
    fn threshold_dwell_time_measures_longest_above_run() {
        // above 100: positions 2-5 (4 samples)
        let values = [98.0, 99.0, 101.0, 102.0, 103.0, 104.0, 99.0];
        let dwell = threshold_dwell_time(&values, 100.0).unwrap();
        assert_eq!(dwell, 4);
    }

    #[test]
    fn threshold_dwell_time_returns_zero_when_never_above() {
        let values = [90.0, 91.0, 92.0];
        let dwell = threshold_dwell_time(&values, 100.0).unwrap();
        assert_eq!(dwell, 0);
    }

    #[test]
    fn threshold_dwell_time_rejects_empty() {
        assert_eq!(
            threshold_dwell_time(&[], 100.0).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn threshold_burden_computes_area_above() {
        // values above 10: 12 (+2), 15 (+5), 11 (+1) => total area = 8
        let values = [8.0, 12.0, 15.0, 11.0, 9.0];
        let burden = threshold_burden(&values, 10.0).unwrap();
        assert!((burden - 8.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_burden_zero_when_all_below() {
        let values = [5.0, 6.0, 7.0];
        let burden = threshold_burden(&values, 10.0).unwrap();
        assert!((burden - 0.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_burden_rejects_empty() {
        assert_eq!(
            threshold_burden(&[], 10.0).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn recovery_half_time_measures_post_peak_decay() {
        // peak at index 2 (value 20.0), half-peak = 10.0
        // recovery to <= 10.0 at index 6
        let values = [5.0, 10.0, 20.0, 16.0, 13.0, 11.0, 10.0, 8.0];
        let half_time = recovery_half_time(&values).unwrap();
        // distance from peak (idx 2) to recovery (idx 6) = 4
        assert_eq!(half_time, Some(4));
    }

    #[test]
    fn recovery_half_time_returns_none_if_never_recovers() {
        // peak at index 2 (20.0), half-peak = 10.0, never goes back down
        let values = [5.0, 10.0, 20.0, 15.0, 14.0, 13.0, 12.0];
        let half_time = recovery_half_time(&values).unwrap();
        assert_eq!(half_time, None);
    }

    #[test]
    fn recovery_half_time_rejects_short_input() {
        assert_eq!(
            recovery_half_time(&[1.0]).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn rolling_cv_computes_coefficient_of_variation() {
        // window=3, last window [2,4,6]: mean=4, var=8/3, std=sqrt(8/3), cv=sqrt(8/3)/4
        let values = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        let cv = rolling_cv(&values, 3).unwrap();
        assert!(cv[0].is_nan() && cv[1].is_nan());
        let last = cv[5];
        let expected_cv = (8.0_f64 / 3.0).sqrt() / 4.0;
        assert!((last - expected_cv).abs() < 1e-12);
    }

    #[test]
    fn rolling_cv_rejects_zero_window() {
        assert_eq!(
            rolling_cv(&[1.0, 2.0], 0).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn trend_stability_index_low_for_noisy_signal() {
        let noisy = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let idx = trend_stability_index(&noisy, 3).unwrap();
        assert!(idx < 0.3);
    }

    #[test]
    fn trend_stability_index_high_for_trend_signal() {
        let trending = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx = trend_stability_index(&trending, 3).unwrap();
        assert!(idx > 0.7);
    }

    #[test]
    fn trend_stability_index_rejects_short_input() {
        assert_eq!(
            trend_stability_index(&[1.0, 2.0], 3).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    // ── Batch 4: peak-to-peak / spectral flatness / zero-crossing / entropy / fractal ──

    #[test]
    fn peak_to_peak_interval_finds_distances() {
        // peaks at indices 2 (val 5) and 6 (val 7)
        let values = [1.0, 2.0, 5.0, 3.0, 2.0, 4.0, 7.0, 4.0, 1.0];
        let intervals = peak_to_peak_interval(&values).unwrap();
        assert_eq!(intervals, vec![4]);
    }

    #[test]
    fn peak_to_peak_interval_empty_when_single_peak() {
        let values = [1.0, 2.0, 3.0, 2.0, 1.0];
        let intervals = peak_to_peak_interval(&values).unwrap();
        assert!(intervals.is_empty());
    }

    #[test]
    fn peak_to_peak_interval_rejects_short_input() {
        assert_eq!(
            peak_to_peak_interval(&[1.0, 2.0]).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn spectral_flatness_near_one_for_white_noise() {
        // shifted-to-positive uniform-ish values => high flatness
        let values = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0];
        let sf = spectral_flatness(&values, 4).unwrap();
        let last = sf[5];
        assert!(last > 0.7, "spectral flatness was {last}");
    }

    #[test]
    fn spectral_flatness_near_zero_for_spike() {
        // mostly flat with a spike => geometric mean << arithmetic mean
        let values = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0];
        let sf = spectral_flatness(&values, 4).unwrap();
        let last = sf[5];
        assert!(last < 0.5);
    }

    #[test]
    fn spectral_flatness_rejects_bad_window() {
        assert_eq!(
            spectral_flatness(&[1.0, 2.0], 0).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn zero_crossing_rate_counts_sign_changes() {
        // sign changes at indices 1,2,3,4,5 => 5 crossings in 5 intervals => rate = 5/5 = 1.0
        let values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let rate = zero_crossing_rate(&values).unwrap();
        assert!((rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_crossing_rate_zero_for_monotone() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rate = zero_crossing_rate(&values).unwrap();
        assert!((rate - 0.0).abs() < 1e-12);
    }

    #[test]
    fn zero_crossing_rate_rejects_empty() {
        assert_eq!(
            zero_crossing_rate(&[]).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn rolling_entropy_high_for_uniform_distribution() {
        // 4 unique values each appearing once in a window of 4 => max entropy = log2(4) = 2.0
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ent = rolling_entropy(&values, 4, 4).unwrap();
        let last = ent[7];
        assert!(last > 1.9);
    }

    #[test]
    fn rolling_entropy_low_for_constant() {
        let values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let ent = rolling_entropy(&values, 4, 4).unwrap();
        let last = ent[5];
        assert!((last - 0.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_entropy_rejects_bad_window() {
        assert_eq!(
            rolling_entropy(&[1.0, 2.0], 0, 4).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn rolling_entropy_rejects_zero_bins() {
        assert_eq!(
            rolling_entropy(&[1.0, 2.0, 3.0], 3, 0).unwrap_err(),
            SignalError::InvalidThreshold
        );
    }

    #[test]
    fn fractal_dimension_higher_for_noisy_signal() {
        let noisy: Vec<f64> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let smooth: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
        let fd_noisy = fractal_dimension(&noisy, 6).unwrap();
        let fd_smooth = fractal_dimension(&smooth, 6).unwrap();
        assert!(fd_noisy > fd_smooth, "noisy={fd_noisy}, smooth={fd_smooth}");
    }

    #[test]
    fn fractal_dimension_rejects_short_input() {
        assert_eq!(
            fractal_dimension(&[1.0], 2).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn fractal_dimension_rejects_bad_k() {
        assert_eq!(
            fractal_dimension(&[1.0, 2.0, 3.0, 4.0], 1).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    // ── Batch 5: autocorrelation / frequency / SNR / RR / z-score / moments / energy / RMS / MA-x ──

    #[test]
    fn lagged_autocorrelation_peak_at_zero_lag() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let acf = lagged_autocorrelation(&data, 0).unwrap();
        assert!((acf - 1.0).abs() < 1e-12, "lag-0 autocorrelation should be 1.0, got {acf}");
    }

    #[test]
    fn lagged_autocorrelation_periodic_signal() {
        // sine-like period of 4, longer sequence for stronger autocorrelation
        let data: Vec<f64> = (0..16).map(|i| {
            let phase = (i as f64) * std::f64::consts::FRAC_PI_2;
            phase.sin()
        }).collect();
        let acf4 = lagged_autocorrelation(&data, 4).unwrap();
        assert!(acf4 > 0.7, "lag-4 should be high for period-4 signal, got {acf4}");
    }

    #[test]
    fn lagged_autocorrelation_rejects_short() {
        assert_eq!(
            lagged_autocorrelation(&[1.0], 0).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn dominant_frequency_detects_period() {
        // period of 4 samples => freq = 0.25 cycles/sample
        let data = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let freq = dominant_frequency(&data).unwrap();
        assert!((freq - 0.25).abs() < 0.05, "expected ~0.25, got {freq}");
    }

    #[test]
    fn dominant_frequency_rejects_short() {
        assert_eq!(
            dominant_frequency(&[1.0, 2.0]).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn signal_to_noise_ratio_clean_signal_high() {
        let clean = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let snr = signal_to_noise_ratio(&clean).unwrap();
        assert!(snr > 40.0, "constant signal should have very high SNR, got {snr}");
    }

    #[test]
    fn signal_to_noise_ratio_noisy_signal_lower() {
        let noisy = [10.0, 12.0, 8.0, 11.0, 9.0, 10.0];
        let snr = signal_to_noise_ratio(&noisy).unwrap();
        assert!(snr < 30.0, "noisy signal should have lower SNR, got {snr}");
    }

    #[test]
    fn signal_to_noise_ratio_rejects_short() {
        assert_eq!(
            signal_to_noise_ratio(&[1.0]).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn rr_interval_variability_classifies_regular() {
        let rr = [800.0, 805.0, 795.0, 800.0, 805.0, 795.0];
        let var = rr_interval_variability(&rr, 0.1).unwrap();
        assert_eq!(var, RRVariability::Regular);
    }

    #[test]
    fn rr_interval_variability_classifies_irregular() {
        let rr = [800.0, 600.0, 1000.0, 500.0, 900.0, 700.0];
        let var = rr_interval_variability(&rr, 0.1).unwrap();
        assert_eq!(var, RRVariability::Irregular);
    }

    #[test]
    fn rr_interval_variability_rejects_short() {
        assert_eq!(
            rr_interval_variability(&[800.0], 0.1).unwrap_err(),
            SignalError::InvalidLength
        );
    }

    #[test]
    fn vital_sign_zscore_computes_deviation() {
        let values = [98.0, 99.0, 100.0, 101.0, 102.0];
        let pop_mean = 100.0;
        let pop_std = 2.0;
        let z = vital_sign_zscore(&values, pop_mean, pop_std).unwrap();
        assert!((z[0] - (-1.0)).abs() < 1e-12);
        assert!((z[2] - 0.0).abs() < 1e-12);
        assert!((z[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vital_sign_zscore_rejects_zero_std() {
        assert_eq!(
            vital_sign_zscore(&[1.0, 2.0], 1.0, 0.0).unwrap_err(),
            SignalError::InvalidThreshold
        );
    }

    #[test]
    fn rolling_skewness_detects_asymmetry() {
        // right-skewed: many small values, one large
        let values = [1.0, 1.0, 1.0, 1.0, 10.0];
        let skew = rolling_skewness(&values, 5).unwrap();
        assert!(skew[4] > 0.5, "right-skewed data should have positive skewness, got {}", skew[4]);
    }

    #[test]
    fn rolling_skewness_rejects_bad_window() {
        assert_eq!(
            rolling_skewness(&[1.0, 2.0], 0).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn rolling_kurtosis_detects_heavy_tails() {
        // heavy-tailed: many values near mean, one extreme outlier
        let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0];
        let kurt = rolling_kurtosis(&values, 10).unwrap();
        assert!(kurt[9] > 3.0, "heavy-tailed data should have kurtosis > 3, got {}", kurt[9]);
    }

    #[test]
    fn rolling_kurtosis_rejects_bad_window() {
        assert_eq!(
            rolling_kurtosis(&[1.0, 2.0], 0).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn signal_energy_computes_sum_of_squares() {
        let values = [3.0, 4.0];
        let energy = signal_energy(&values).unwrap();
        assert!((energy - 25.0).abs() < 1e-12);
    }

    #[test]
    fn signal_energy_rejects_empty() {
        assert_eq!(
            signal_energy(&[]).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn rms_power_computes_root_mean_square() {
        let values = [3.0, 4.0];
        let rms = rms_power(&values).unwrap();
        assert!((rms - 3.5355339).abs() < 1e-5, "expected ~3.5355, got {rms}");
    }

    #[test]
    fn rms_power_rejects_empty() {
        assert_eq!(
            rms_power(&[]).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn moving_average_crossover_detects_crossings() {
        // downtrend then uptrend: fast MA should cross above slow MA at the turn
        let values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let signals = moving_average_crossover(&values, 2, 4).unwrap();
        assert!(signals.iter().any(|s| *s == MACross::Bullish), "expected at least one bullish crossing, got {:?}", signals);
    }

    #[test]
    fn moving_average_crossover_rejects_bad_windows() {
        assert_eq!(
            moving_average_crossover(&[1.0, 2.0, 3.0], 0, 2).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    // ── Batch 6: derivative / resample / percentile / band_energy / symmetry / rate / jitter / prominence / saturation ──

    #[test]
    fn derivative_computes_central_differences() {
        let values = [1.0, 4.0, 9.0, 16.0, 25.0];
        let d = derivative(&values).unwrap();
        assert_eq!(d.len(), 5);
        // central difference at index 2: (16-4)/2 = 6
        assert!((d[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn derivative_rejects_short() {
        assert_eq!(derivative(&[1.0]).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn second_derivative_computes_acceleration() {
        // y = x^2 => first deriv = 2x, second deriv = 2
        let values = [0.0, 1.0, 4.0, 9.0, 16.0];
        let sd = second_derivative(&values).unwrap();
        assert!((sd[2] - 2.0).abs() < 0.5, "second deriv near center should be ~2, got {}", sd[2]);
    }

    #[test]
    fn second_derivative_rejects_short() {
        assert_eq!(second_derivative(&[1.0, 2.0]).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn resample_linear_changes_length() {
        let values = [0.0, 10.0, 20.0, 30.0];
        let out = resample_linear(&values, 7).unwrap();
        assert_eq!(out.len(), 7);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[6] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn resample_linear_rejects_bad_n() {
        assert_eq!(resample_linear(&[1.0, 2.0], 0).unwrap_err(), SignalError::InvalidLength);
        assert_eq!(resample_linear(&[1.0, 2.0], 1).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn moving_percentile_computes_p50() {
        let values = [5.0, 3.0, 1.0, 4.0, 2.0, 6.0];
        let p50 = moving_percentile(&values, 4, 50.0).unwrap();
        // window [5,3,1,4] sorted = [1,3,4,5], p50 = 3.5
        assert!((p50[3] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn moving_percentile_rejects_bad_window() {
        assert_eq!(
            moving_percentile(&[1.0, 2.0], 0, 50.0).unwrap_err(),
            SignalError::InvalidWindow
        );
    }

    #[test]
    fn band_energy_approximates_from_zero_crossings() {
        // high-frequency signal should have more energy than a flat one
        let flat = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let oscillating = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let e_flat = band_energy(&flat).unwrap();
        let e_osc = band_energy(&oscillating).unwrap();
        assert!(e_osc > e_flat);
    }

    #[test]
    fn band_energy_rejects_empty() {
        assert_eq!(band_energy(&[]).unwrap_err(), SignalError::EmptyInput);
    }

    #[test]
    fn waveform_symmetry_high_for_symmetric() {
        // symmetric triangle: [1,3,5,3,1]
        let values = [1.0, 3.0, 5.0, 3.0, 1.0];
        let sym = waveform_symmetry(&values).unwrap();
        assert!(sym > 0.9, "symmetric waveform should have symmetry near 1.0, got {sym}");
    }

    #[test]
    fn waveform_symmetry_low_for_asymmetric() {
        // heavily skewed: [1,1,1,1,10]
        let values = [1.0, 1.0, 1.0, 1.0, 10.0];
        let sym = waveform_symmetry(&values).unwrap();
        assert!(sym < 0.6, "asymmetric waveform should have low symmetry, got {sym}");
    }

    #[test]
    fn waveform_symmetry_rejects_short() {
        assert_eq!(waveform_symmetry(&[1.0]).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn rate_of_change_computes_pct_change() {
        let values = [100.0, 110.0, 121.0];
        let roc = rate_of_change(&values).unwrap();
        assert_eq!(roc.len(), 3);
        assert!((roc[0] - 0.0).abs() < 1e-12);
        assert!((roc[1] - 10.0).abs() < 1e-12);
        assert!((roc[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn rate_of_change_rejects_empty() {
        assert_eq!(rate_of_change(&[]).unwrap_err(), SignalError::EmptyInput);
    }

    #[test]
    fn jitter_score_low_for_regular_intervals() {
        let peaks = [0usize, 100, 200, 300, 400];
        let jitter = jitter_score(&peaks).unwrap();
        assert!(jitter < 0.05, "regular intervals should have near-zero jitter, got {jitter}");
    }

    #[test]
    fn jitter_score_high_for_irregular_intervals() {
        let peaks = [0usize, 80, 220, 270, 450];
        let jitter = jitter_score(&peaks).unwrap();
        assert!(jitter > 0.1, "irregular intervals should have higher jitter, got {jitter}");
    }

    #[test]
    fn jitter_score_rejects_short() {
        assert_eq!(jitter_score(&[0usize]).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn peak_prominence_measures_relative_height() {
        // peak at idx 2 (val 10), flanking mins ~1 and ~2 => prominence ~8-9
        let values = [1.0, 5.0, 10.0, 4.0, 2.0, 6.0, 8.0, 6.0, 1.0];
        let proms = peak_prominence(&values).unwrap();
        let (_, p) = proms[0];
        assert!(p > 5.0, "main peak prominence should be high, got {p}");
    }

    #[test]
    fn peak_prominence_rejects_short() {
        assert_eq!(peak_prominence(&[1.0, 2.0]).unwrap_err(), SignalError::InvalidLength);
    }

    #[test]
    fn signal_saturation_ratio_detects_clipping() {
        // signal in [0, 10] range, many values at bounds
        let values = [0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 3.0, 0.0];
        let ratio = signal_saturation_ratio(&values, 0.0, 10.0).unwrap();
        // 6 of 8 samples at bounds => ratio = 0.75
        assert!((ratio - 0.75).abs() < 1e-12);
    }

    #[test]
    fn signal_saturation_ratio_rejects_empty() {
        assert_eq!(
            signal_saturation_ratio(&[], 0.0, 10.0).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn signal_saturation_ratio_rejects_bad_range() {
        assert_eq!(
            signal_saturation_ratio(&[1.0, 2.0], 10.0, 0.0).unwrap_err(),
            SignalError::InvalidRange
        );
    }
}
