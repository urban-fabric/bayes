//! Shape, standardization, co-movement, and the trimmed means.
//!
//! [`standardized_moment`] divides by the standard deviation *before* raising to
//! the power, not after. The two are algebraically identical and numerically
//! very different: forming `Sum((x-m)^k)` and `var^(k/2)` separately overflows for
//! data at `1e150` (both reach `1e450`, so the ratio is `inf/inf = NaN`) and
//! underflows at `1e-150`. A standardized moment is scale-free, so there is no
//! reason to leave the scale in the intermediate terms at all.
use super::location::{mean, mean_and_m2, mean_and_m2_real, mean_real, sorted_finite};
use super::spread::{std_dev, std_dev_real};
use euclid_numeric::Real;

/// `k`-th standardized central moment about the mean (population form).
///
/// The standardization happens **before** the power, not after. Forming `Σ(x−m)^k` and
/// `var^{k/2}` separately and then dividing is algebraically the same thing and numerically
/// much worse: for data at `1e150` both terms reach `1e450` and overflow, so the ratio is
/// `inf/inf = NaN`; for data at `1e-150` both underflow and it is `0/0 = NaN`. In both cases
/// the right answer is an ordinary small number, because a standardized moment is scale-free.
/// Dividing first keeps every `z` of order one, so every `z^k` is too, and the result is
/// correct across the whole representable range.
fn standardized_moment(data: &[f64], k: i32) -> f64 {
    let (m, m2) = mean_and_m2(data);
    let n = data.len() as f64;
    let var = m2 / n;
    if var <= 0.0 {
        return f64::NAN;
    }
    let sd = var.sqrt();
    data.iter().map(|&x| ((x - m) / sd).powi(k)).sum::<f64>() / n
}

/// As [`standardized_moment`], the division by the standard deviation comes first so the
/// cubed term cannot overflow at large scales or underflow at small ones.
fn standardized_moment3_real<R: Real>(data: &[R]) -> R {
    let (m, m2) = mean_and_m2_real(data);
    let n = R::scalar(data.len() as f64);
    let sd = (m2 / n.clone()).sqrt();
    data.iter()
        .cloned()
        .map(|x| {
            let z = (x - m.clone()) / sd.clone();
            z.clone() * z.clone() * z
        })
        .fold(R::zero(), |acc, x| acc + x)
        / n
}

/// As [`standardized_moment`], standardized before the power rather than after.
fn standardized_moment4_real<R: Real>(data: &[R]) -> R {
    let (m, m2) = mean_and_m2_real(data);
    let n = R::scalar(data.len() as f64);
    let sd = (m2 / n.clone()).sqrt();
    data.iter()
        .cloned()
        .map(|x| {
            let z = (x - m.clone()) / sd.clone();
            let z2 = z.clone() * z;
            z2.clone() * z2
        })
        .fold(R::zero(), |acc, x| acc + x)
        / n
}

/// Sample skewness — the third standardized moment. `NaN` if fewer than 2 points
/// or zero variance.
pub fn skewness(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    standardized_moment(data, 3)
}

/// Population skewness over any [`Real`] carrier.
pub fn skewness_real<R: Real>(data: &[R]) -> R {
    if data.len() < 2 {
        return R::nan();
    }
    standardized_moment3_real(data)
}

/// Excess kurtosis — the fourth standardized moment minus 3. `NaN` if degenerate.
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    standardized_moment(data, 4) - 3.0
}

/// Excess kurtosis over any [`Real`] carrier.
pub fn kurtosis_real<R: Real>(data: &[R]) -> R {
    if data.len() < 2 {
        return R::nan();
    }
    standardized_moment4_real(data) - R::scalar(3.0)
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

/// Standardize values to z-scores over any [`Real`] carrier.
pub fn z_scores_real<R: Real>(data: &[R]) -> Vec<R> {
    let m = mean_real(data);
    let s = std_dev_real(data);
    data.iter()
        .cloned()
        .map(|x| (x - m.clone()) / s.clone())
        .collect()
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

/// Sample covariance over any [`Real`] carrier.
pub fn covariance_real<R: Real>(a: &[R], b: &[R]) -> R {
    if a.len() != b.len() || a.len() < 2 {
        return R::nan();
    }
    let ma = mean_real(a);
    let mb = mean_real(b);
    let cov = a
        .iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(x, y)| (x - ma.clone()) * (y - mb.clone()))
        .fold(R::zero(), |acc, x| acc + x);
    cov / R::scalar((a.len() - 1) as f64)
}

/// Pearson correlation coefficient `cov(a,b) / (σ_a σ_b)`, in `[-1, 1]`.
///
/// The `[-1, 1]` bound is the covariance Cauchy–Schwarz inequality
/// `Cov(a,b)² ≤ Var(a)·Var(b)`, proved in `riemann`
/// (`RiemannLean/Bayes/Inequalities.lean`: `covariance_sq_le`,
/// `correlation_abs_le_one`).
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let denom = std_dev(a) * std_dev(b);
    if denom == 0.0 || denom.is_nan() {
        return f64::NAN;
    }
    covariance(a, b) / denom
}

/// Pearson correlation coefficient over any [`Real`] carrier.
pub fn correlation_real<R: Real>(a: &[R], b: &[R]) -> R {
    covariance_real(a, b) / (std_dev_real(a) * std_dev_real(b))
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_distribution_ext::assert_close;
    // The suite covers the whole `summary` surface, not just this module.
    use crate::summary::*;

    #[test]
    fn location_and_dispersion() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_close!(mean(&d), 5.0, 1e-12);
        assert_close!(population_variance(&d), 4.0, 1e-12);
        assert_close!(variance(&d), 32.0 / 7.0, 1e-12);
        assert_close!(std_dev(&d), (32.0_f64 / 7.0).sqrt(), 1e-12);
    }

    #[test]
    fn generic_real_summaries_match_host_f64() {
        let d = [1.0, 2.0, 4.0, 8.0];
        assert_close!(mean_real(&d), mean(&d), 1e-12);
        assert_close!(population_variance_real(&d), population_variance(&d), 1e-12);
        assert_close!(variance_real(&d), variance(&d), 1e-12);
        assert_close!(std_dev_real(&d), std_dev(&d), 1e-12);
        assert_close!(geometric_mean_real(&d), geometric_mean(&d), 1e-12);
        assert_close!(harmonic_mean_real(&d), harmonic_mean(&d), 1e-12);
        assert_close!(mean_abs_deviation_real(&d), mean_abs_deviation(&d), 1e-12);
        assert_close!(
            coefficient_of_variation_real(&d),
            coefficient_of_variation(&d),
            1e-12,
        );
        assert_close!(skewness_real(&d), skewness(&d), 1e-12);
        assert_close!(kurtosis_real(&d), kurtosis(&d), 1e-12);

        let e = [2.0, 3.0, 5.0, 11.0];
        assert_close!(covariance_real(&d, &e), covariance(&d, &e), 1e-12);
        assert_close!(correlation_real(&d, &e), correlation(&d, &e), 1e-12);

        let z = z_scores(&d);
        let generic_z = z_scores_real(&d);
        assert_eq!(generic_z.len(), z.len());
        for (got, expected) in generic_z.iter().zip(z) {
            assert_close!(*got, expected, 1e-12);
        }
    }

    #[test]
    fn correlation_lies_in_the_unit_interval() {
        // |ρ| ≤ 1, the covariance Cauchy–Schwarz bound (riemann correlation_abs_le_one).
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pairs: [&[f64]; 3] = [
            &[2.0, 4.0, 6.0, 8.0, 10.0],
            &[5.0, 1.0, 4.0, 2.0, 3.0],
            &[1.0, 0.0, 1.0, 0.0, 1.0],
        ];
        for b in pairs {
            let r = correlation(&a, b);
            assert!((-1.0 - 1e-12..=1.0 + 1e-12).contains(&r), "|ρ| > 1: {r}");
        }
        // Perfect positive linear dependence saturates ρ = 1.
        assert_close!(correlation(&a, &[2.0, 4.0, 6.0, 8.0, 10.0]), 1.0, 1e-12);
        // Perfect negative: ρ = −1.
        assert_close!(correlation(&a, &[5.0, 4.0, 3.0, 2.0, 1.0]), -1.0, 1e-12);
    }

    #[test]
    fn means_family() {
        let d = [1.0, 2.0, 4.0];
        assert_close!(geometric_mean(&d), 2.0, 1e-12);
        assert_close!(harmonic_mean(&d), 3.0 / (1.0 + 0.5 + 0.25), 1e-12);
        assert!(geometric_mean(&[1.0, -1.0]).is_nan());
    }

    #[test]
    fn robust_summaries_reduce_outlier_influence() {
        let d = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_close!(trimmed_mean(&d, 0.2), 3.0, 1e-12);
        assert_close!(winsorized_mean(&d, 0.2), 3.0, 1e-12);
        assert!(mean_abs_deviation(&d) > 0.0);
        assert!(coefficient_of_variation(&d) > 0.0);
    }

    #[test]
    fn gini_difference_and_z_scores_match_known_values() {
        assert_close!(gini_mean_difference(&[1.0, 2.0, 3.0]), 4.0 / 3.0, 1e-12);
        let z = z_scores(&[1.0, 2.0, 3.0]);
        assert_close!(z.iter().sum::<f64>(), 0.0, 1e-12);
        assert_close!(z.iter().map(|x| x * x).sum::<f64>(), 2.0, 1e-12);
    }

    #[test]
    fn shape_of_symmetric_data_is_zero_skew() {
        let d = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert_close!(skewness(&d), 0.0, 1e-12);
    }

    #[test]
    fn covariance_and_correlation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert_close!(correlation(&a, &b), 1.0, 1e-12);
        assert_close!(covariance(&a, &b), 5.0, 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_nan_not_panics() {
        assert!(mean(&[]).is_nan());
        assert!(variance(&[1.0]).is_nan());
        assert!(correlation(&[1.0, 2.0], &[1.0]).is_nan());
    }
}
