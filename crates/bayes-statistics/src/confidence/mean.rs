//! Confidence intervals for **means**: the Student-t and known-σ normal intervals
//! for a single mean, and Welch's interval for a difference of two means.

use super::{ConfidenceInterval, alpha, invalid, valid_level, z_quantile};
use crate::{mean, variance};
use bayes_distribution_ext::{ContinuousCdf, StudentT};

/// Student-t interval for the **mean** with unknown variance (the usual small-sample
/// interval): `x̄ ± t_{1−α/2, n−1} · s/√n`.
pub fn mean_t_interval(data: &[f64], level: f64) -> ConfidenceInterval {
    let n = data.len();
    if n < 2 || !valid_level(level) {
        return invalid(level);
    }
    let q = 1.0 - (1.0 - level) / 2.0;
    let t = match StudentT::new(n as f64 - 1.0) {
        Ok(d) => d.inverse_cdf(q),
        Err(_) => z_quantile(q),
    };
    let half = t * variance(data).sqrt() / (n as f64).sqrt();
    let m = mean(data);
    ConfidenceInterval {
        lower: m - half,
        upper: m + half,
        level,
    }
}

/// Normal (z) interval for the **mean** with a *known* standard deviation `sigma`:
/// `x̄ ± z_{1−α/2} · σ/√n`.
pub fn mean_z_interval(data: &[f64], sigma: f64, level: f64) -> ConfidenceInterval {
    let n = data.len();
    if n == 0 || sigma <= 0.0 || !valid_level(level) {
        return invalid(level);
    }
    let z = z_quantile(1.0 - (1.0 - level) / 2.0);
    let half = z * sigma / (n as f64).sqrt();
    let m = mean(data);
    ConfidenceInterval {
        lower: m - half,
        upper: m + half,
        level,
    }
}

/// Welch interval for the difference of two independent means, `mean(a) - mean(b)`.
pub fn mean_difference_welch_interval(a: &[f64], b: &[f64], level: f64) -> ConfidenceInterval {
    if a.len() < 2 || b.len() < 2 || !valid_level(level) {
        return invalid(level);
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (variance(a), variance(b));
    let (sea, seb) = (va / na, vb / nb);
    let se = (sea + seb).sqrt();
    let df = (sea + seb).powi(2) / (sea * sea / (na - 1.0) + seb * seb / (nb - 1.0));
    let q = 1.0 - alpha(level) / 2.0;
    let t = match StudentT::new(df) {
        Ok(d) => d.inverse_cdf(q),
        Err(_) => f64::NAN,
    };
    let diff = mean(a) - mean(b);
    ConfidenceInterval {
        lower: diff - t * se,
        upper: diff + t * se,
        level,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_distribution_ext::assert_close;

    #[test]
    fn t_interval_brackets_the_mean() {
        let data = [4.8, 5.0, 5.2, 4.9, 5.1, 5.0];
        let ci = mean_t_interval(&data, 0.95);
        assert!(ci.contains(mean(&data)));
        assert!(ci.lower < mean(&data) && mean(&data) < ci.upper);
        assert!(mean_t_interval(&data, 0.99).width() > ci.width());
    }

    #[test]
    fn z_interval_known_width() {
        let data = [10.0, 10.0, 10.0, 10.0];
        let ci = mean_z_interval(&data, 2.0, 0.95);
        assert_close!(ci.upper - 10.0, 1.959_963_98, 1e-4);
    }

    #[test]
    fn mean_difference_interval_centers_on_observed_difference() {
        let a = [10.0, 11.0, 12.0, 13.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let ci = mean_difference_welch_interval(&a, &b, 0.95);
        assert_close!((ci.lower + ci.upper) / 2.0, mean(&a) - mean(&b), 1e-12);
    }
}
