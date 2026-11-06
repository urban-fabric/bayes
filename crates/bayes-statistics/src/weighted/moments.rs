//! Weighted moments.
//!
//! The defining property, and the one the relationship suite pins: an integer
//! weight of `k` must give the same answer as repeating the observation `k`
//! times. Note which normalization each carries — `weighted_variance` and
//! `weighted_covariance` are the unbiased forms, so at equal weights they land
//! on the *sample* variance, while `weighted_population_variance` lands on the
//! population one.
use super::normalize::{valid, weight_sum_real};
use euclid_numeric::Real;

/// Weighted arithmetic mean.
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> f64 {
    if !valid(data, weights) {
        return f64::NAN;
    }
    let sum_w = weights.iter().sum::<f64>();
    data.iter().zip(weights).map(|(&x, &w)| w * x).sum::<f64>() / sum_w
}

/// Weighted arithmetic mean over any [`Real`] carrier.
pub fn weighted_mean_real<R: Real>(data: &[R], weights: &[R]) -> R {
    if data.len() != weights.len() || data.is_empty() {
        return R::nan();
    }
    let sum_w = weight_sum_real(weights);
    data.iter()
        .cloned()
        .zip(weights.iter().cloned())
        .map(|(x, w)| w * x)
        .fold(R::zero(), |acc, x| acc + x)
        / sum_w
}

/// Weighted population variance, dividing by `Σw`.
pub fn weighted_population_variance(data: &[f64], weights: &[f64]) -> f64 {
    if !valid(data, weights) {
        return f64::NAN;
    }
    let m = weighted_mean(data, weights);
    let sum_w = weights.iter().sum::<f64>();
    data.iter()
        .zip(weights)
        .map(|(&x, &w)| w * (x - m).powi(2))
        .sum::<f64>()
        / sum_w
}

/// Weighted population variance over any [`Real`] carrier.
pub fn weighted_population_variance_real<R: Real>(data: &[R], weights: &[R]) -> R {
    if data.len() != weights.len() || data.is_empty() {
        return R::nan();
    }
    let m = weighted_mean_real(data, weights);
    let sum_w = weight_sum_real(weights);
    data.iter()
        .cloned()
        .zip(weights.iter().cloned())
        .map(|(x, w)| {
            let centered = x - m.clone();
            w * centered.clone() * centered
        })
        .fold(R::zero(), |acc, x| acc + x)
        / sum_w
}

/// Unbiased weighted sample variance for reliability weights.
pub fn weighted_variance(data: &[f64], weights: &[f64]) -> f64 {
    if !valid(data, weights) {
        return f64::NAN;
    }
    let m = weighted_mean(data, weights);
    let sum_w = weights.iter().sum::<f64>();
    let sum_w2 = weights.iter().map(|w| w * w).sum::<f64>();
    let denom = sum_w * sum_w - sum_w2;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let ss = data
        .iter()
        .zip(weights)
        .map(|(&x, &w)| w * (x - m).powi(2))
        .sum::<f64>();
    sum_w * ss / denom
}

/// Unbiased weighted sample variance for reliability weights over any [`Real`] carrier.
pub fn weighted_variance_real<R: Real>(data: &[R], weights: &[R]) -> R {
    if data.len() != weights.len() || data.is_empty() {
        return R::nan();
    }
    let m = weighted_mean_real(data, weights);
    let sum_w = weight_sum_real(weights);
    let sum_w2 = weights
        .iter()
        .cloned()
        .map(|w| w.clone() * w)
        .fold(R::zero(), |acc, w| acc + w);
    let denom = sum_w.clone() * sum_w.clone() - sum_w2;
    let ss = data
        .iter()
        .cloned()
        .zip(weights.iter().cloned())
        .map(|(x, w)| {
            let centered = x - m.clone();
            w * centered.clone() * centered
        })
        .fold(R::zero(), |acc, x| acc + x);
    sum_w * ss / denom
}

/// Weighted sample standard deviation.
pub fn weighted_std_dev(data: &[f64], weights: &[f64]) -> f64 {
    weighted_variance(data, weights).sqrt()
}

/// Weighted sample standard deviation over any [`Real`] carrier.
pub fn weighted_std_dev_real<R: Real>(data: &[R], weights: &[R]) -> R {
    weighted_variance_real(data, weights).sqrt()
}

/// Weighted sample covariance for paired data.
pub fn weighted_covariance(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
    if a.len() != b.len() || !valid(a, weights) || !b.iter().all(|x| x.is_finite()) {
        return f64::NAN;
    }
    let ma = weighted_mean(a, weights);
    let mb = weighted_mean(b, weights);
    let sum_w = weights.iter().sum::<f64>();
    let sum_w2 = weights.iter().map(|w| w * w).sum::<f64>();
    let denom = sum_w * sum_w - sum_w2;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let ss = a
        .iter()
        .zip(b)
        .zip(weights)
        .map(|((&x, &y), &w)| w * (x - ma) * (y - mb))
        .sum::<f64>();
    sum_w * ss / denom
}

/// Weighted sample covariance over any [`Real`] carrier.
pub fn weighted_covariance_real<R: Real>(a: &[R], b: &[R], weights: &[R]) -> R {
    if a.len() != b.len() || a.len() != weights.len() || a.is_empty() {
        return R::nan();
    }
    let ma = weighted_mean_real(a, weights);
    let mb = weighted_mean_real(b, weights);
    let sum_w = weight_sum_real(weights);
    let sum_w2 = weights
        .iter()
        .cloned()
        .map(|w| w.clone() * w)
        .fold(R::zero(), |acc, w| acc + w);
    let denom = sum_w.clone() * sum_w.clone() - sum_w2;
    let ss = a
        .iter()
        .cloned()
        .zip(b.iter().cloned())
        .zip(weights.iter().cloned())
        .map(|((x, y), w)| w * (x - ma.clone()) * (y - mb.clone()))
        .fold(R::zero(), |acc, x| acc + x);
    sum_w * ss / denom
}

/// Weighted Pearson correlation.
pub fn weighted_correlation(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
    let cov = weighted_covariance(a, b, weights);
    let denom = weighted_std_dev(a, weights) * weighted_std_dev(b, weights);
    if denom <= 0.0 || denom.is_nan() {
        f64::NAN
    } else {
        cov / denom
    }
}

/// Weighted Pearson correlation over any [`Real`] carrier.
pub fn weighted_correlation_real<R: Real>(a: &[R], b: &[R], weights: &[R]) -> R {
    weighted_covariance_real(a, b, weights)
        / (weighted_std_dev_real(a, weights) * weighted_std_dev_real(b, weights))
}
