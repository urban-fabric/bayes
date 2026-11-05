//! Chi-squared-quantile intervals: the exact **Poisson rate** interval and the
//! **variance** / **standard deviation** intervals for normally distributed data.

use super::{ConfidenceInterval, alpha, invalid, valid_level};
use crate::variance;
use bayes_distribution_ext::{ChiSquared, ContinuousCdf};

/// Exact Poisson rate interval for `count / exposure`, using chi-squared quantiles.
pub fn poisson_rate_interval(count: u64, exposure: f64, level: f64) -> ConfidenceInterval {
    if exposure <= 0.0 || !exposure.is_finite() || !valid_level(level) {
        return invalid(level);
    }
    let a = alpha(level) / 2.0;
    let lower = if count == 0 {
        0.0
    } else {
        match ChiSquared::new(2.0 * count as f64) {
            Ok(d) => 0.5 * d.inverse_cdf(a) / exposure,
            Err(_) => f64::NAN,
        }
    };
    let upper = match ChiSquared::new(2.0 * (count + 1) as f64) {
        Ok(d) => 0.5 * d.inverse_cdf(1.0 - a) / exposure,
        Err(_) => f64::NAN,
    };
    ConfidenceInterval {
        lower,
        upper,
        level,
    }
}

/// Chi-squared interval for the population variance of normally distributed data.
pub fn variance_chi_square_interval(data: &[f64], level: f64) -> ConfidenceInterval {
    let n = data.len();
    if n < 2 || !valid_level(level) {
        return invalid(level);
    }
    let df = n as f64 - 1.0;
    let s2 = variance(data);
    let Ok(chi2) = ChiSquared::new(df) else {
        return invalid(level);
    };
    let a = alpha(level) / 2.0;
    ConfidenceInterval {
        lower: df * s2 / chi2.inverse_cdf(1.0 - a),
        upper: df * s2 / chi2.inverse_cdf(a),
        level,
    }
}

/// Chi-squared interval for the population standard deviation.
pub fn std_dev_chi_square_interval(data: &[f64], level: f64) -> ConfidenceInterval {
    let ci = variance_chi_square_interval(data, level);
    ConfidenceInterval {
        lower: ci.lower.sqrt(),
        upper: ci.upper.sqrt(),
        level,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_distribution_ext::assert_close;

    #[test]
    fn variance_and_poisson_intervals_are_ordered() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance_chi_square_interval(&data, 0.95);
        assert!(v.lower > 0.0 && v.lower < variance(&data) && variance(&data) < v.upper);
        let s = std_dev_chi_square_interval(&data, 0.95);
        assert_close!(s.lower * s.lower, v.lower, 1e-12);
        let rate = poisson_rate_interval(5, 2.0, 0.95);
        assert!(rate.lower >= 0.0 && rate.upper > rate.lower);
    }
}
