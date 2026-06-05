//! Weighted descriptive statistics.

fn valid(data: &[f64], weights: &[f64]) -> bool {
    data.len() == weights.len()
        && !data.is_empty()
        && data.iter().all(|x| x.is_finite())
        && weights.iter().all(|w| w.is_finite() && *w >= 0.0)
        && weights.iter().sum::<f64>() > 0.0
}

/// Sum of weights. `NaN` for malformed input.
pub fn weight_sum(weights: &[f64]) -> f64 {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return f64::NAN;
    }
    let sum = weights.iter().sum::<f64>();
    if sum > 0.0 { sum } else { f64::NAN }
}

/// Effective sample size `(Σw)^2 / Σw²`.
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

/// Weighted arithmetic mean.
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> f64 {
    if !valid(data, weights) {
        return f64::NAN;
    }
    let sum_w = weights.iter().sum::<f64>();
    data.iter().zip(weights).map(|(&x, &w)| w * x).sum::<f64>() / sum_w
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

/// Weighted sample standard deviation.
pub fn weighted_std_dev(data: &[f64], weights: &[f64]) -> f64 {
    weighted_variance(data, weights).sqrt()
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
    }

    #[test]
    fn weighted_quantile_follows_cumulative_weight() {
        let data = [1.0, 10.0];
        let weights = [9.0, 1.0];
        close(weighted_quantile(&data, &weights, 0.5), 1.0, 1e-12);
        close(effective_sample_size(&weights), 100.0 / 82.0, 1e-12);
    }

    #[test]
    fn weighted_correlation_detects_linear_relationship() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let w = [1.0, 2.0, 1.0, 2.0];
        close(weighted_correlation(&x, &y, &w), 1.0, 1e-12);
    }
}
