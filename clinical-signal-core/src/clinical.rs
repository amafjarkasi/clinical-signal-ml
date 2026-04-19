use crate::types::{
    DeteriorationFlags, EarlyWarningWindow, News2LiteScore, PatientState, RiskSummary,
    RRVariability, SignalError, SignalQualityScore, TrendDeteriorationFlags,
};
use crate::outlier::mad_outlier_flags;
use crate::rolling::trend_slope;

pub fn hrv_rmssd(rr_intervals_ms: &[f64]) -> Result<f64, SignalError> {
    if rr_intervals_ms.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if rr_intervals_ms.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(SignalError::NonFiniteValue);
    }

    let mut sum_sq = 0.0;
    for i in 1..rr_intervals_ms.len() {
        let d = rr_intervals_ms[i] - rr_intervals_ms[i - 1];
        sum_sq += d * d;
    }
    Ok((sum_sq / (rr_intervals_ms.len() - 1) as f64).sqrt())
}

pub fn hrv_sdnn(rr_intervals_ms: &[f64]) -> Result<f64, SignalError> {
    if rr_intervals_ms.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if rr_intervals_ms.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(SignalError::NonFiniteValue);
    }

    let mean = rr_intervals_ms.iter().sum::<f64>() / rr_intervals_ms.len() as f64;
    let var = rr_intervals_ms
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / rr_intervals_ms.len() as f64;
    Ok(var.sqrt())
}

pub fn rr_interval_variability(rr_intervals_ms: &[f64], cv_threshold: f64) -> Result<RRVariability, SignalError> {
    if rr_intervals_ms.len() < 2 {
        return Err(SignalError::InvalidLength);
    }
    if rr_intervals_ms.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(SignalError::NonFiniteValue);
    }
    if cv_threshold <= 0.0 || !cv_threshold.is_finite() {
        return Err(SignalError::InvalidThreshold);
    }

    let n = rr_intervals_ms.len() as f64;
    let mean = rr_intervals_ms.iter().sum::<f64>() / n;
    let var = rr_intervals_ms.iter().map(|v| { let d = v - mean; d * d }).sum::<f64>() / n;
    let cv = var.sqrt() / mean;

    if cv > cv_threshold {
        Ok(RRVariability::Irregular)
    } else {
        Ok(RRVariability::Regular)
    }
}

pub fn deterioration_flags(
    heart_rate_bpm: f64,
    systolic_bp_mmhg: f64,
    spo2_percent: f64,
    respiratory_rate_bpm: f64,
) -> Result<DeteriorationFlags, SignalError> {
    let inputs = [
        heart_rate_bpm,
        systolic_bp_mmhg,
        spo2_percent,
        respiratory_rate_bpm,
    ];
    if inputs.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(SignalError::NonFiniteValue);
    }
    if spo2_percent > 100.0 {
        return Err(SignalError::InvalidRange);
    }

    let tachycardia = heart_rate_bpm > 100.0;
    let hypotension = systolic_bp_mmhg < 90.0;
    let hypoxemia = spo2_percent < 92.0;
    let tachypnea = respiratory_rate_bpm > 22.0;
    let risk_score = [tachycardia, hypotension, hypoxemia, tachypnea]
        .iter()
        .filter(|flag| **flag)
        .count() as u8;

    Ok(DeteriorationFlags {
        tachycardia,
        hypotension,
        hypoxemia,
        tachypnea,
        risk_score,
        high_risk: risk_score >= 2,
    })
}

pub fn deterioration_trend_flags(
    heart_rate_bpm: &[f64],
    systolic_bp_mmhg: &[f64],
    spo2_percent: &[f64],
    respiratory_rate_bpm: &[f64],
    window: usize,
) -> Result<TrendDeteriorationFlags, SignalError> {
    let n = heart_rate_bpm.len();
    if n == 0
        || systolic_bp_mmhg.len() != n
        || spo2_percent.len() != n
        || respiratory_rate_bpm.len() != n
    {
        return Err(SignalError::InvalidLength);
    }
    if window < 2 || window > n {
        return Err(SignalError::InvalidWindow);
    }
    if heart_rate_bpm.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || systolic_bp_mmhg.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || respiratory_rate_bpm.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || spo2_percent.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 100.0)
    {
        return Err(SignalError::InvalidRange);
    }

    let start = n - window;
    let hr_slope = trend_slope(&heart_rate_bpm[start..])?;
    let sbp_slope = trend_slope(&systolic_bp_mmhg[start..])?;
    let spo2_slope = trend_slope(&spo2_percent[start..])?;
    let rr_slope = trend_slope(&respiratory_rate_bpm[start..])?;

    let rising_heart_rate = hr_slope > 0.0;
    let falling_systolic_bp = sbp_slope < 0.0;
    let falling_spo2 = spo2_slope < 0.0;
    let rising_respiratory_rate = rr_slope > 0.0;
    let risk_score = [
        rising_heart_rate,
        falling_systolic_bp,
        falling_spo2,
        rising_respiratory_rate,
    ]
    .iter()
    .filter(|flag| **flag)
    .count() as u8;

    Ok(TrendDeteriorationFlags {
        rising_heart_rate,
        falling_systolic_bp,
        falling_spo2,
        rising_respiratory_rate,
        risk_score,
        high_risk: risk_score >= 3,
    })
}

pub fn news2_lite_score(
    heart_rate_bpm: f64,
    respiratory_rate_bpm: f64,
    spo2_percent: f64,
    systolic_bp_mmhg: f64,
    temperature_c: f64,
    altered_consciousness: bool,
) -> Result<News2LiteScore, SignalError> {
    let inputs = [
        heart_rate_bpm,
        respiratory_rate_bpm,
        spo2_percent,
        systolic_bp_mmhg,
        temperature_c,
    ];
    if inputs.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteValue);
    }
    if heart_rate_bpm <= 0.0
        || respiratory_rate_bpm <= 0.0
        || systolic_bp_mmhg <= 0.0
        || !(0.0..=100.0).contains(&spo2_percent)
    {
        return Err(SignalError::InvalidRange);
    }

    let rr_score = if respiratory_rate_bpm <= 8.0 {
        3
    } else if respiratory_rate_bpm <= 11.0 {
        1
    } else if respiratory_rate_bpm <= 20.0 {
        0
    } else if respiratory_rate_bpm <= 24.0 {
        2
    } else {
        3
    };

    let spo2_score = if spo2_percent <= 91.0 {
        3
    } else if spo2_percent <= 93.0 {
        2
    } else if spo2_percent <= 95.0 {
        1
    } else {
        0
    };

    let temp_score = if temperature_c <= 35.0 {
        3
    } else if temperature_c <= 36.0 {
        1
    } else if temperature_c <= 38.0 {
        0
    } else if temperature_c <= 39.0 {
        1
    } else {
        2
    };

    let sbp_score = if systolic_bp_mmhg <= 90.0 {
        3
    } else if systolic_bp_mmhg <= 100.0 {
        2
    } else if systolic_bp_mmhg <= 110.0 {
        1
    } else if systolic_bp_mmhg <= 219.0 {
        0
    } else {
        3
    };

    let hr_score = if heart_rate_bpm <= 40.0 {
        3
    } else if heart_rate_bpm <= 50.0 {
        1
    } else if heart_rate_bpm <= 90.0 {
        0
    } else if heart_rate_bpm <= 110.0 {
        1
    } else if heart_rate_bpm <= 130.0 {
        2
    } else {
        3
    };

    let consciousness_score = if altered_consciousness { 3 } else { 0 };
    let score = rr_score + spo2_score + temp_score + sbp_score + hr_score + consciousness_score;

    Ok(News2LiteScore {
        score,
        high_risk: score >= 7 || consciousness_score == 3 || rr_score == 3 || spo2_score == 3,
    })
}

pub fn risk_summary(
    heart_rate_bpm: &[f64],
    systolic_bp_mmhg: &[f64],
    spo2_percent: &[f64],
    respiratory_rate_bpm: &[f64],
    temperature_c: f64,
    altered_consciousness: bool,
    trend_window: usize,
) -> Result<RiskSummary, SignalError> {
    let n = heart_rate_bpm.len();
    if n == 0
        || systolic_bp_mmhg.len() != n
        || spo2_percent.len() != n
        || respiratory_rate_bpm.len() != n
    {
        return Err(SignalError::InvalidLength);
    }

    let snapshot = deterioration_flags(
        heart_rate_bpm[n - 1],
        systolic_bp_mmhg[n - 1],
        spo2_percent[n - 1],
        respiratory_rate_bpm[n - 1],
    )?;
    let trend = deterioration_trend_flags(
        heart_rate_bpm,
        systolic_bp_mmhg,
        spo2_percent,
        respiratory_rate_bpm,
        trend_window,
    )?;
    let news2 = news2_lite_score(
        heart_rate_bpm[n - 1],
        respiratory_rate_bpm[n - 1],
        spo2_percent[n - 1],
        systolic_bp_mmhg[n - 1],
        temperature_c,
        altered_consciousness,
    )?;

    let total = snapshot
        .risk_score
        .saturating_add(trend.risk_score)
        .saturating_add(news2.score);
    Ok(RiskSummary {
        snapshot_score: snapshot.risk_score,
        trend_score: trend.risk_score,
        news2_score: news2.score,
        total_score: total,
        high_risk: snapshot.high_risk || trend.high_risk || news2.high_risk || total >= 8,
    })
}

pub fn early_warning_window(
    high_risk_windows: &[bool],
    min_consecutive: usize,
) -> Result<EarlyWarningWindow, SignalError> {
    if high_risk_windows.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if min_consecutive == 0 {
        return Err(SignalError::InvalidWindow);
    }

    let mut current = 0usize;
    let mut best = 0usize;
    for &flag in high_risk_windows {
        if flag {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 0;
        }
    }

    Ok(EarlyWarningWindow {
        should_alert: best >= min_consecutive,
        sustained_windows: best,
    })
}

pub fn patient_state_transition(
    previous_risk: u8,
    current_risk: u8,
    delta_threshold: u8,
) -> Result<PatientState, SignalError> {
    if delta_threshold == 0 {
        return Err(SignalError::InvalidThreshold);
    }
    let prev = previous_risk as i16;
    let curr = current_risk as i16;
    let delta = curr - prev;
    if delta >= delta_threshold as i16 {
        return Ok(PatientState::Worsening);
    }
    if delta <= -(delta_threshold as i16) {
        return Ok(PatientState::Improving);
    }
    Ok(PatientState::Stable)
}

pub fn signal_quality_score(data: &[f64]) -> Result<SignalQualityScore, SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if data.iter().any(|v| v.is_infinite()) {
        return Err(SignalError::NonFiniteValue);
    }

    let finite: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let completeness = finite.len() as f64 / data.len() as f64;
    if finite.len() < 2 {
        return Ok(SignalQualityScore {
            score: completeness.clamp(0.0, 1.0),
            completeness,
            stable_sampling: true,
            outlier_ratio: 0.0,
        });
    }

    let mut diffs = Vec::with_capacity(finite.len() - 1);
    for i in 1..finite.len() {
        diffs.push((finite[i] - finite[i - 1]).abs());
    }
    let diff_mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let diff_var = diffs
        .iter()
        .map(|d| {
            let x = *d - diff_mean;
            x * x
        })
        .sum::<f64>()
        / diffs.len() as f64;
    let stable_sampling = diff_var.sqrt() <= 10.0;

    let outlier_ratio = if finite.len() >= 4 {
        let flags = mad_outlier_flags(&finite, 4, 3.5)?;
        flags.iter().filter(|f| **f).count() as f64 / finite.len() as f64
    } else {
        0.0
    };

    let stability_factor = if stable_sampling { 1.0 } else { 0.85 };
    let score = (completeness * (1.0 - outlier_ratio) * stability_factor).clamp(0.0, 1.0);

    Ok(SignalQualityScore {
        score,
        completeness,
        stable_sampling,
        outlier_ratio,
    })
}
