//! Summary statistics — location, dispersion, shape, and association — computed
//! with numerically stable one-pass algorithms (Welford) where it matters.

/// Arithmetic mean. `NaN` for an empty sample.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Welford accumulation of `(mean, M₂ = Σ(xᵢ-mean)²)` in one pass.
fn mean_and_m2(data: &[f64]) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (x - mean);
    }
    (mean, m2)
}

/// Unbiased sample variance (divides by `n-1`). `NaN` for fewer than 2 points.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let (_, m2) = mean_and_m2(data);
    m2 / (data.len() as f64 - 1.0)
}

/// Population (biased) variance (divides by `n`). `NaN` for an empty sample.
pub fn population_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let (_, m2) = mean_and_m2(data);
    m2 / data.len() as f64
}

/// Sample standard deviation (`√variance`).
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Coefficient of variation `s / x̄`. `NaN` if the mean is zero or undefined.
pub fn coefficient_of_variation(data: &[f64]) -> f64 {
    let m = mean(data);
    if m == 0.0 || m.is_nan() {
        return f64::NAN;
    }
    std_dev(data) / m.abs()
}

/// `k`-th standardized central moment about the mean (population form).
fn standardized_moment(data: &[f64], k: i32) -> f64 {
    let (m, m2) = mean_and_m2(data);
    let n = data.len() as f64;
    let var = m2 / n;
    if var <= 0.0 {
        return f64::NAN;
    }
    let mk = data.iter().map(|&x| (x - m).powi(k)).sum::<f64>() / n;
    mk / var.powf(k as f64 / 2.0)
}

/// Sample skewness — the third standardized moment. `NaN` if fewer than 2 points
/// or zero variance.
pub fn skewness(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    standardized_moment(data, 3)
}

/// Excess kurtosis — the fourth standardized moment minus 3. `NaN` if degenerate.
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    standardized_moment(data, 4) - 3.0
}

/// Geometric mean `(∏ xᵢ)^{1/n}` (requires `xᵢ > 0`). `NaN` if empty or any
/// non-positive value.
pub fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    (data.iter().map(|&x| x.ln()).sum::<f64>() / data.len() as f64).exp()
}

/// Harmonic mean `n / Σ(1/xᵢ)` (requires `xᵢ > 0`). `NaN` if empty or any
/// non-positive value.
pub fn harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    data.len() as f64 / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

/// Mean absolute deviation around the arithmetic mean.
pub fn mean_abs_deviation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).abs()).sum::<f64>() / data.len() as f64
}

fn sorted_finite(data: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Trimmed mean after dropping `proportion` from each tail.
pub fn trimmed_mean(data: &[f64], proportion: f64) -> f64 {
    if !(0.0..0.5).contains(&proportion) {
        return f64::NAN;
    }
    let v = sorted_finite(data);
    let n = v.len();
    let k = (n as f64 * proportion).floor() as usize;
    if n == 0 || 2 * k >= n {
        return f64::NAN;
    }
    mean(&v[k..(n - k)])
}

/// Winsorized mean after clamping `proportion` from each tail to the nearest
/// retained values.
pub fn winsorized_mean(data: &[f64], proportion: f64) -> f64 {
    if !(0.0..0.5).contains(&proportion) {
        return f64::NAN;
    }
    let v = sorted_finite(data);
    let n = v.len();
    let k = (n as f64 * proportion).floor() as usize;
    if n == 0 || 2 * k >= n {
        return f64::NAN;
    }
    let lo = v[k];
    let hi = v[n - k - 1];
    let total = v
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i < k {
                lo
            } else if i >= n - k {
                hi
            } else {
                x
            }
        })
        .sum::<f64>();
    total / n as f64
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

/// Standardize each value to a z-score using the sample mean and sample standard
/// deviation. Returns an empty vector for degenerate input.
pub fn z_scores(data: &[f64]) -> Vec<f64> {
    let m = mean(data);
    let s = std_dev(data);
    if data.is_empty() || s <= 0.0 || s.is_nan() {
        return Vec::new();
    }
    data.iter().map(|&x| (x - m) / s).collect()
}

/// Sample covariance between paired samples (Bessel's `n-1`). `NaN` if the lengths
/// differ or fewer than 2 pairs.
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    let ma = mean(a);
    let mb = mean(b);
    let cov: f64 = a.iter().zip(b).map(|(&x, &y)| (x - ma) * (y - mb)).sum();
    cov / (a.len() as f64 - 1.0)
}

/// Pearson correlation coefficient `cov(a,b) / (σ_a σ_b)`, in `[-1, 1]`.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let denom = std_dev(a) * std_dev(b);
    if denom == 0.0 || denom.is_nan() {
        return f64::NAN;
    }
    covariance(a, b) / denom
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn location_and_dispersion() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        close(mean(&d), 5.0, 1e-12);
        close(population_variance(&d), 4.0, 1e-12);
        close(variance(&d), 32.0 / 7.0, 1e-12);
        close(std_dev(&d), (32.0_f64 / 7.0).sqrt(), 1e-12);
    }

    #[test]
    fn means_family() {
        let d = [1.0, 2.0, 4.0];
        close(geometric_mean(&d), 2.0, 1e-12); // (1·2·4)^{1/3} = 2
        close(harmonic_mean(&d), 3.0 / (1.0 + 0.5 + 0.25), 1e-12);
        assert!(geometric_mean(&[1.0, -1.0]).is_nan());
    }

    #[test]
    fn robust_summaries_reduce_outlier_influence() {
        let d = [1.0, 2.0, 3.0, 4.0, 100.0];
        close(trimmed_mean(&d, 0.2), 3.0, 1e-12);
        close(winsorized_mean(&d, 0.2), 3.0, 1e-12);
        assert!(mean_abs_deviation(&d) > 0.0);
        assert!(coefficient_of_variation(&d) > 0.0);
    }

    #[test]
    fn gini_difference_and_z_scores_match_known_values() {
        close(gini_mean_difference(&[1.0, 2.0, 3.0]), 4.0 / 3.0, 1e-12);
        let z = z_scores(&[1.0, 2.0, 3.0]);
        close(z.iter().sum::<f64>(), 0.0, 1e-12);
        close(z.iter().map(|x| x * x).sum::<f64>(), 2.0, 1e-12);
    }

    #[test]
    fn shape_of_symmetric_data_is_zero_skew() {
        let d = [-2.0, -1.0, 0.0, 1.0, 2.0];
        close(skewness(&d), 0.0, 1e-12);
    }

    #[test]
    fn covariance_and_correlation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0]; // perfectly linear, slope 2
        close(correlation(&a, &b), 1.0, 1e-12);
        close(covariance(&a, &b), 5.0, 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_nan_not_panics() {
        assert!(mean(&[]).is_nan());
        assert!(variance(&[1.0]).is_nan());
        assert!(correlation(&[1.0, 2.0], &[1.0]).is_nan());
    }
}
