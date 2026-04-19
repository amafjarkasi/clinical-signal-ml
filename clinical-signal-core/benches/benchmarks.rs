use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use clinical_signal_core::*;

fn bench_ewma(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("ewma_1000", |b| b.iter(|| ewma(black_box(&data), black_box(0.3))));
}

fn bench_rolling_mean(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("rolling_mean_1000_w50", |b| {
        b.iter(|| rolling_mean(black_box(&data), black_box(50)))
    });
}

fn bench_rolling_variance(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("rolling_variance_1000_w50", |b| {
        b.iter(|| rolling_variance(black_box(&data), black_box(50)))
    });
}

fn bench_rolling_median(c: &mut Criterion) {
    let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("rolling_median_500_w20", |b| {
        b.iter(|| rolling_median(black_box(&data), black_box(20)))
    });
}

fn bench_detect_spikes(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("detect_spikes_zscore_1000", |b| {
        b.iter(|| detect_spikes_zscore(black_box(&data), black_box(20), black_box(3.0)))
    });
}

fn bench_mad_outlier(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("mad_outlier_flags_1000", |b| {
        b.iter(|| mad_outlier_flags(black_box(&data), black_box(20), black_box(3.5)))
    });
}

fn bench_hrv_rmssd(c: &mut Criterion) {
    let rr: Vec<f64> = (0..500).map(|i| 800.0 + (i as f64 * 0.05).sin() * 20.0).collect();
    c.bench_function("hrv_rmssd_500", |b| {
        b.iter(|| hrv_rmssd(black_box(&rr)))
    });
}

fn bench_news2(c: &mut Criterion) {
    c.bench_function("news2_lite_score", |b| {
        b.iter(|| news2_lite_score(black_box(72.0), black_box(16.0), black_box(98.0), black_box(122.0), black_box(36.8), black_box(false)))
    });
}

fn bench_fractal_dimension(c: &mut Criterion) {
    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("fractal_dimension_256", |b| {
        b.iter(|| fractal_dimension(black_box(&data), black_box(8)))
    });
}

fn bench_kalman(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| 10.0 + (i as f64 * 0.05).sin() * 3.0).collect();
    let config = kalman::KalmanConfig {
        initial_estimate: data[0],
        ..Default::default()
    };
    c.bench_function("kalman_filter_1000", |b| {
        b.iter(|| kalman::kalman_filter(black_box(&data), black_box(&config)))
    });
}

fn bench_streaming_ewma(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_ewma");
    for size in [100, 1000, 10000] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let mut s = streaming::StreamingEwma::new(0.3).unwrap();
                s.process_batch(data)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ewma,
    bench_rolling_mean,
    bench_rolling_variance,
    bench_rolling_median,
    bench_detect_spikes,
    bench_mad_outlier,
    bench_hrv_rmssd,
    bench_news2,
    bench_fractal_dimension,
    bench_kalman,
    bench_streaming_ewma,
);
criterion_main!(benches);
