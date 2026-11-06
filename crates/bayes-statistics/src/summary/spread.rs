//! Measures of spread.
//!
//! Both normalizations are offered and named for which they are: `variance` is
//! the unbiased `n-1` form, `population_variance` the `n` form. Getting that
//! wrong is a silent `n/(n-1)` error, which is why the relationship suite pins
//! each one against the weighted reductions that must reduce to it.
use super::location::{mean, mean_and_m2, mean_and_m2_real, mean_real, sorted_finite};
use euclid_numeric::Real;

/// Unbiased sample variance (divides by `n-1`). `NaN` for fewer than 2 points.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let (_, m2) = mean_and_m2(data);
    m2 / (data.len() as f64 - 1.0)
}

/// Unbiased sample variance over any [`Real`] carrier.
pub fn variance_real<R: Real>(data: &[R]) -> R {
    if data.len() < 2 {
        return R::nan();
    }
    let (_, m2) = mean_and_m2_real(data);
    m2 / R::scalar((data.len() - 1) as f64)
}

/// Population (biased) variance (divides by `n`). `NaN` for an empty sample.
pub fn population_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let (_, m2) = mean_and_m2(data);
    m2 / data.len() as f64
}

/// Population variance over any [`Real`] carrier.
pub fn population_variance_real<R: Real>(data: &[R]) -> R {
    if data.is_empty() {
        return R::nan();
    }
    let (_, m2) = mean_and_m2_real(data);
    m2 / R::scalar(data.len() as f64)
}

/// Sample standard deviation (`√variance`).
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Sample standard deviation over any [`Real`] carrier.
pub fn std_dev_real<R: Real>(data: &[R]) -> R {
    variance_real(data).sqrt()
}

/// Coefficient of variation `s / |x̄|`. `NaN` if the mean is zero or undefined.
pub fn coefficient_of_variation(data: &[f64]) -> f64 {
    let m = mean(data);
    if m == 0.0 || m.is_nan() {
        return f64::NAN;
    }
    std_dev(data) / m.abs()
}

/// Coefficient of variation over any [`Real`] carrier.
pub fn coefficient_of_variation_real<R: Real>(data: &[R]) -> R {
    std_dev_real(data) / mean_real(data).abs()
}

/// Mean absolute deviation around the arithmetic mean.
pub fn mean_abs_deviation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).abs()).sum::<f64>() / data.len() as f64
}

/// Mean absolute deviation over any [`Real`] carrier.
pub fn mean_abs_deviation_real<R: Real>(data: &[R]) -> R {
    if data.is_empty() {
        return R::nan();
    }
    let m = mean_real(data);
    data.iter()
        .cloned()
        .map(|x| (x - m.clone()).abs())
        .fold(R::zero(), |acc, x| acc + x)
        / R::scalar(data.len() as f64)
}

/// Gini mean difference: average absolute difference between all unordered pairs.
pub fn gini_mean_difference(data: &[f64]) -> f64 {
    let v = sorted_finite(data);
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let weighted_sum = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * x)
        .sum::<f64>();
    2.0 * weighted_sum / (n as f64 * (n as f64 - 1.0))
}

/// Gini mean difference over any [`Real`] carrier. Carriers carry no order, so this uses the
/// sort-free pairwise form `2·Σ_{i<j}|xᵢ−xⱼ| / (n(n−1))` (`O(n²)`, versus the eager
/// `O(n log n)` sorted form) — the lazy-surface equivalent that lowers onto every backend.
pub fn gini_mean_difference_real<R: Real>(data: &[R]) -> R {
    let n = data.len();
    if n < 2 {
        return R::nan();
    }
    let mut total = R::zero();
    for (i, x) in data.iter().cloned().enumerate() {
        for y in data.iter().skip(i + 1).cloned() {
            total = total + (x.clone() - y).abs();
        }
    }
    R::two() * total / R::scalar(n as f64 * (n as f64 - 1.0))
}
