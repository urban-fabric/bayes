//! Weighted descriptive statistics.

use euclid::Real;

fn valid(data: &[f64], weights: &[f64]) -> bool {
    data.len() == weights.len()
        && !data.is_empty()
        && data.iter().all(|x| x.is_finite())
        && weights.iter().all(|w| w.is_finite() && *w >= 0.0)
        && weights.iter().sum::<f64>() > 0.0
}

fn log_sum_exp(xs: &[f64]) -> Option<f64> {
    let (&first, rest) = xs.split_first()?;
    let max = rest.iter().fold(first, |acc, &x| acc.max(x));
    let shifted_sum = xs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    Some(max + shifted_sum.ln())
}

fn softmax(xs: &[f64]) -> Option<Vec<f64>> {
    let log_z = log_sum_exp(xs)?;
    Some(xs.iter().map(|&x| (x - log_z).exp()).collect())
}

/// Sum of weights. `NaN` for malformed input.
pub fn weight_sum(weights: &[f64]) -> f64 {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return f64::NAN;
    }
    let sum = weights.iter().sum::<f64>();
    if sum > 0.0 { sum } else { f64::NAN }
}

/// Sum of weights over any [`Real`] carrier.
pub fn weight_sum_real<R: Real>(weights: &[R]) -> R {
    if weights.is_empty() {
        return R::nan();
    }
    weights.iter().cloned().fold(R::zero(), |acc, w| acc + w)
}

/// Normalize nonnegative raw weights to a probability vector.
///
/// This is the raw-weight counterpart of riemann `normWeights_isDistribution`:
/// when the return value is non-empty, its entries are nonnegative and sum to
/// one up to floating-point roundoff. Returns an empty vector for malformed,
/// all-zero, or overflowed inputs.
pub fn normalized_weights(weights: &[f64]) -> Vec<f64> {
    let sum = weight_sum(weights);
    if !sum.is_finite() {
        return Vec::new();
    }
    weights.iter().map(|&w| w / sum).collect()
}

/// Normalize raw weights over any [`Real`] carrier.
pub fn normalized_weights_real<R: Real>(weights: &[R]) -> Vec<R> {
    if weights.is_empty() {
        return Vec::new();
    }
    let sum = weight_sum_real(weights);
    weights.iter().cloned().map(|w| w / sum.clone()).collect()
}

/// Normalize log-weights to a probability vector using stable softmax.
///
/// This is the log-domain form used by importance weights, mixture
/// responsibilities, and other "weights known only up to a log constant"
/// calculations. It inherits the shift-invariance of the stable `softmax`, so adding a
/// constant to every log-weight does not change the result. Returns an empty vector on
/// empty input.
pub fn normalized_log_weights(log_weights: &[f64]) -> Vec<f64> {
    // The "empty on non-normalizable input" policy belongs here: a non-normalizable
    // input (empty, all `−∞`, `±∞`, or `NaN`) yields a non-finite vector, which we drop.
    softmax(log_weights)
        .filter(|p| p.iter().all(|x| x.is_finite()))
        .unwrap_or_default()
}

/// Effective sample size `(Σw)^2 / Σw²`.
///
/// For normalized nonnegative weights this equals the Kish ESS
/// `1 / Σᵢ pᵢ²`, and riemann proves the exact interpretation range
/// `1 ≤ ESS ≤ n` (`RiemannLean/Markov/ImportanceSampling.lean`: `one_le_ess`,
/// `ess_le_card`). The formula is scale-invariant, so raw weights may be passed
/// directly.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum = weight_sum(weights);
    if sum.is_nan() {
        return f64::NAN;
    }
    let sum_sq = weights.iter().map(|w| w * w).sum::<f64>();
    if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        f64::NAN
    }
}

/// Effective sample size `(Σw)^2 / Σw²` over any [`Real`] carrier.
pub fn effective_sample_size_real<R: Real>(weights: &[R]) -> R {
    if weights.is_empty() {
        return R::nan();
    }
    let sum = weight_sum_real(weights);
    let sum_sq = weights
        .iter()
        .cloned()
        .map(|w| w.clone() * w)
        .fold(R::zero(), |acc, w| acc + w);
    sum.clone() * sum / sum_sq
}

/// Whether an ESS value is attainable for `n` nonnegative normalized weights.
///
/// A false result flags a diagnostic outside the riemann-backed range
/// `1 ≤ ESS ≤ n`. The tolerance absorbs floating-point slack at the degenerate
/// (`ESS = 1`) and uniform (`ESS = n`) boundaries.
pub fn is_effective_sample_size_admissible(ess: f64, n: usize) -> bool {
    if n == 0 || !ess.is_finite() {
        return false;
    }
    let upper = n as f64;
    let tol = 1e-9 * upper.max(1.0);
    ess + tol >= 1.0 && ess <= upper + tol
}

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

/// Weighted empirical quantile. Uses the first ordered value whose cumulative
/// weight reaches `p * Σw`.
pub fn weighted_quantile(data: &[f64], weights: &[f64], p: f64) -> f64 {
    if !valid(data, weights) || p.is_nan() {
        return f64::NAN;
    }
    let mut pairs: Vec<(f64, f64)> = data.iter().copied().zip(weights.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if p <= 0.0 {
        return pairs[0].0;
    }
    if p >= 1.0 {
        return pairs[pairs.len() - 1].0;
    }
    let target = p * pairs.iter().map(|(_, w)| *w).sum::<f64>();
    let mut cum = 0.0;
    for (x, w) in pairs {
        cum += w;
        if cum >= target {
            return x;
        }
    }
    f64::NAN
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn equal_weights_match_unweighted_mean_variance() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        close(weighted_mean(&data, &w), 2.5, 1e-12);
        close(weighted_variance(&data, &w), 5.0 / 3.0, 1e-12);
        close(
            weighted_mean_real(&data, &w),
            weighted_mean(&data, &w),
            1e-12,
        );
        close(
            weighted_population_variance_real(&data, &w),
            weighted_population_variance(&data, &w),
            1e-12,
        );
        close(
            weighted_variance_real(&data, &w),
            weighted_variance(&data, &w),
            1e-12,
        );
        close(
            weighted_std_dev_real(&data, &w),
            weighted_std_dev(&data, &w),
            1e-12,
        );
    }

    #[test]
    fn weighted_quantile_follows_cumulative_weight() {
        let data = [1.0, 10.0];
        let weights = [9.0, 1.0];
        close(weighted_quantile(&data, &weights, 0.5), 1.0, 1e-12);
        close(effective_sample_size(&weights), 100.0 / 82.0, 1e-12);
        close(
            effective_sample_size_real(&weights),
            effective_sample_size(&weights),
            1e-12,
        );
    }

    #[test]
    fn raw_and_log_weight_normalization_are_probability_vectors() {
        let weights = [2.0, 6.0, 2.0];
        let p = normalized_weights(&weights);
        assert_eq!(p.len(), 3);
        close(p.iter().sum(), 1.0, 1e-12);
        close(p[1], 0.6, 1e-12);
        let p_real = normalized_weights_real(&weights);
        assert_eq!(p_real, p);
        close(
            effective_sample_size(&weights),
            effective_sample_size(&p),
            1e-12,
        );

        let log_p = normalized_log_weights(&[1000.0, 1000.0, 998.0]);
        assert_eq!(log_p.len(), 3);
        close(log_p.iter().sum(), 1.0, 1e-12);
        close(log_p[0], log_p[1], 1e-12);
        assert!(log_p[2] < log_p[0]);

        assert!(normalized_weights(&[0.0, 0.0]).is_empty());
        assert!(normalized_weights(&[-1.0, 2.0]).is_empty());
        assert!(normalized_log_weights(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_empty());
    }

    #[test]
    fn effective_sample_size_obeys_riemann_bounds() {
        let point_mass = [1.0, 0.0, 0.0, 0.0];
        close(effective_sample_size(&point_mass), 1.0, 1e-12);
        assert!(is_effective_sample_size_admissible(
            effective_sample_size(&point_mass),
            point_mass.len()
        ));

        let uniform = [1.0, 1.0, 1.0, 1.0];
        close(effective_sample_size(&uniform), 4.0, 1e-12);
        assert!(is_effective_sample_size_admissible(
            effective_sample_size(&uniform),
            uniform.len()
        ));

        assert!(!is_effective_sample_size_admissible(0.99, 4));
        assert!(!is_effective_sample_size_admissible(4.01, 4));
        assert!(!is_effective_sample_size_admissible(1.0, 0));
    }

    #[test]
    fn weighted_correlation_detects_linear_relationship() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let w = [1.0, 2.0, 1.0, 2.0];
        close(weighted_correlation(&x, &y, &w), 1.0, 1e-12);
        close(
            weighted_covariance_real(&x, &y, &w),
            weighted_covariance(&x, &y, &w),
            1e-12,
        );
        close(
            weighted_correlation_real(&x, &y, &w),
            weighted_correlation(&x, &y, &w),
            1e-12,
        );
    }
}
