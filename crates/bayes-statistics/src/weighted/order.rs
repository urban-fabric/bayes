//! Weighted order statistics.
//!
//! A weighted quantile has no single definition; this one interpolates the
//! cumulative weight, which reduces to the unweighted `quantile` at equal
//! weights. That reduction is the test.
use super::normalize::valid;

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
    use bayes_distribution_ext::assert_close;
    // The suite covers the whole `weighted` surface, not just this module.
    use crate::weighted::*;

    #[test]
    fn equal_weights_match_unweighted_mean_variance() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        assert_close!(weighted_mean(&data, &w), 2.5, 1e-12);
        assert_close!(weighted_variance(&data, &w), 5.0 / 3.0, 1e-12);
        assert_close!(
            weighted_mean_real(&data, &w),
            weighted_mean(&data, &w),
            1e-12,
        );
        assert_close!(
            weighted_population_variance_real(&data, &w),
            weighted_population_variance(&data, &w),
            1e-12,
        );
        assert_close!(
            weighted_variance_real(&data, &w),
            weighted_variance(&data, &w),
            1e-12,
        );
        assert_close!(
            weighted_std_dev_real(&data, &w),
            weighted_std_dev(&data, &w),
            1e-12,
        );
    }

    #[test]
    fn weighted_quantile_follows_cumulative_weight() {
        let data = [1.0, 10.0];
        let weights = [9.0, 1.0];
        assert_close!(weighted_quantile(&data, &weights, 0.5), 1.0, 1e-12);
        assert_close!(effective_sample_size(&weights), 100.0 / 82.0, 1e-12);
        assert_close!(
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
        assert_close!(p.iter().sum(), 1.0, 1e-12);
        assert_close!(p[1], 0.6, 1e-12);
        let p_real = normalized_weights_real(&weights);
        assert_eq!(p_real, p);
        assert_close!(
            effective_sample_size(&weights),
            effective_sample_size(&p),
            1e-12,
        );

        let log_p = normalized_log_weights(&[1000.0, 1000.0, 998.0]);
        assert_eq!(log_p.len(), 3);
        assert_close!(log_p.iter().sum(), 1.0, 1e-12);
        assert_close!(log_p[0], log_p[1], 1e-12);
        assert!(log_p[2] < log_p[0]);

        assert!(normalized_weights(&[0.0, 0.0]).is_empty());
        assert!(normalized_weights(&[-1.0, 2.0]).is_empty());
        assert!(normalized_log_weights(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_empty());
    }

    #[test]
    fn effective_sample_size_obeys_riemann_bounds() {
        let point_mass = [1.0, 0.0, 0.0, 0.0];
        assert_close!(effective_sample_size(&point_mass), 1.0, 1e-12);
        assert!(is_effective_sample_size_admissible(
            effective_sample_size(&point_mass),
            point_mass.len()
        ));

        let uniform = [1.0, 1.0, 1.0, 1.0];
        assert_close!(effective_sample_size(&uniform), 4.0, 1e-12);
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
        assert_close!(weighted_correlation(&x, &y, &w), 1.0, 1e-12);
        assert_close!(
            weighted_covariance_real(&x, &y, &w),
            weighted_covariance(&x, &y, &w),
            1e-12,
        );
        assert_close!(
            weighted_correlation_real(&x, &y, &w),
            weighted_correlation(&x, &y, &w),
            1e-12,
        );
    }
}
