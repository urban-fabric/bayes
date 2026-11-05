//! Measures of location, and the shared two-pass accumulator.
//!
//! [`mean_and_m2`] is the one everything else is built on: it returns the mean
//! and the sum of squared deviations from it in a single pass, computed *about
//! the running mean* rather than as `Sum(x^2) - (Sum x)^2/n`. That distinction is
//! the whole reason it exists — the algebraic form loses every significant digit
//! when the data sits far from the origin.
use euclid_numeric::Real;

/// Arithmetic mean. `NaN` for an empty sample.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Arithmetic mean over any [`Real`] carrier.
pub fn mean_real<R: Real>(data: &[R]) -> R {
    if data.is_empty() {
        return R::nan();
    }
    data.iter().cloned().fold(R::zero(), |acc, x| acc + x) / R::scalar(data.len() as f64)
}

/// Welford accumulation of `(mean, M₂ = Σ(xᵢ-mean)²)` in one pass.
pub(super) fn mean_and_m2(data: &[f64]) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (x - mean);
    }
    (mean, m2)
}

pub(super) fn mean_and_m2_real<R: Real>(data: &[R]) -> (R, R) {
    let mut mean = R::zero();
    let mut m2 = R::zero();
    for (i, x) in data.iter().cloned().enumerate() {
        let delta = x.clone() - mean.clone();
        mean = mean + delta.clone() / R::scalar(i as f64 + 1.0);
        m2 = m2 + delta * (x - mean.clone());
    }
    (mean, m2)
}

/// Geometric mean `(∏ xᵢ)^{1/n}` (requires `xᵢ > 0`). `NaN` if empty or any
/// non-positive value.
pub fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    (data.iter().map(|&x| x.ln()).sum::<f64>() / data.len() as f64).exp()
}

/// Geometric mean over any [`Real`] carrier.
pub fn geometric_mean_real<R: Real>(data: &[R]) -> R {
    if data.is_empty() {
        return R::nan();
    }
    (data
        .iter()
        .map(|x| x.ln())
        .fold(R::zero(), |acc, x| acc + x)
        / R::scalar(data.len() as f64))
    .exp()
}

/// Harmonic mean `n / Σ(1/xᵢ)` (requires `xᵢ > 0`). `NaN` if empty or any
/// non-positive value.
pub fn harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    data.len() as f64 / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

/// Harmonic mean over any [`Real`] carrier.
pub fn harmonic_mean_real<R: Real>(data: &[R]) -> R {
    if data.is_empty() {
        return R::nan();
    }
    R::scalar(data.len() as f64)
        / data
            .iter()
            .cloned()
            .map(|x| R::one() / x)
            .fold(R::zero(), |acc, x| acc + x)
}

pub(super) fn sorted_finite(data: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}
