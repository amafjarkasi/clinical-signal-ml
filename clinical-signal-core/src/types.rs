#[derive(Debug, PartialEq, Clone)]
pub enum SignalError {
    EmptyInput,
    InvalidAlpha,
    InvalidWindow,
    InvalidThreshold,
    InvalidLength,
    InvalidRange,
    NonFiniteValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeteriorationFlags {
    pub tachycardia: bool,
    pub hypotension: bool,
    pub hypoxemia: bool,
    pub tachypnea: bool,
    pub risk_score: u8,
    pub high_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct News2LiteScore {
    pub score: u8,
    pub high_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrendDeteriorationFlags {
    pub rising_heart_rate: bool,
    pub falling_systolic_bp: bool,
    pub falling_spo2: bool,
    pub rising_respiratory_rate: bool,
    pub risk_score: u8,
    pub high_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalQualityScore {
    pub score: f64,
    pub completeness: f64,
    pub stable_sampling: bool,
    pub outlier_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiskSummary {
    pub snapshot_score: u8,
    pub trend_score: u8,
    pub news2_score: u8,
    pub total_score: u8,
    pub high_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyWarningWindow {
    pub should_alert: bool,
    pub sustained_windows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatientState {
    Improving,
    Stable,
    Worsening,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RRVariability {
    Regular,
    Irregular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MACross {
    Bullish,
    Bearish,
}
