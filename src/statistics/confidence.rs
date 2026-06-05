//! **Closed-form confidence intervals** — the frequentist companions to the point
//! estimates in [`summary`](super::summary). Each returns a [`ConfidenceInterval`]
//! at a requested coverage `level` (e.g. `0.95`), computed from an analytic
//! quantile — no resampling (for that, see [`resampling`](super::resampling)).

use super::summary::{mean, variance};
use crate::distribution::{Beta, ChiSquared, ContinuousCdf, StudentT};
use crate::function::erf::erf_inv;
use std::f64::consts::SQRT_2;

/// A confidence interval `[lower, upper]` at coverage `level ∈ (0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

impl ConfidenceInterval {
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
    /// Whether `x` lies within the interval (inclusive).
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lower && x <= self.upper
    }
}

/// Standard-normal quantile `Φ⁻¹(q)` via the inverse error function.
fn z_quantile(q: f64) -> f64 {
    SQRT_2 * erf_inv(2.0 * q - 1.0)
}

fn invalid(level: f64) -> ConfidenceInterval {
    ConfidenceInterval {
        lower: f64::NAN,
        upper: f64::NAN,
        level,
    }
}

fn valid_level(level: f64) -> bool {
    (0.0..1.0).contains(&level)
}

fn alpha(level: f64) -> f64 {
    1.0 - level
}

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

/// **Wald** interval for a binomial proportion: `p̂ ± z·√(p̂(1−p̂)/n)` (simple, but
/// poor near 0/1 — prefer [`proportion_wilson`] there).
pub fn proportion_wald(successes: u64, n: u64, level: f64) -> ConfidenceInterval {
    if n == 0 || successes > n || !valid_level(level) {
        return invalid(level);
    }
    let nf = n as f64;
    let p = successes as f64 / nf;
    let z = z_quantile(1.0 - (1.0 - level) / 2.0);
    let half = z * (p * (1.0 - p) / nf).sqrt();
    ConfidenceInterval {
        lower: (p - half).max(0.0),
        upper: (p + half).min(1.0),
        level,
    }
}

/// **Wilson score** interval for a binomial proportion — well-behaved for small `n`
/// and extreme `p̂`.
pub fn proportion_wilson(successes: u64, n: u64, level: f64) -> ConfidenceInterval {
    if n == 0 || successes > n || !valid_level(level) {
        return invalid(level);
    }
    let nf = n as f64;
    let p = successes as f64 / nf;
    let z = z_quantile(1.0 - (1.0 - level) / 2.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (p + z2 / (2.0 * nf)) / denom;
    let half = (z / denom) * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt();
    ConfidenceInterval {
        lower: (center - half).max(0.0),
        upper: (center + half).min(1.0),
        level,
    }
}

/// **Agresti-Coull** interval for a binomial proportion.
pub fn proportion_agresti_coull(successes: u64, n: u64, level: f64) -> ConfidenceInterval {
    if n == 0 || successes > n || !valid_level(level) {
        return invalid(level);
    }
    let z = z_quantile(1.0 - alpha(level) / 2.0);
    let z2 = z * z;
    let nf = n as f64 + z2;
    let p = (successes as f64 + z2 / 2.0) / nf;
    let half = z * (p * (1.0 - p) / nf).sqrt();
    ConfidenceInterval {
        lower: (p - half).max(0.0),
        upper: (p + half).min(1.0),
        level,
    }
}

/// Equal-tailed **Clopper-Pearson** exact interval for a binomial proportion.
pub fn proportion_clopper_pearson(successes: u64, n: u64, level: f64) -> ConfidenceInterval {
    if n == 0 || successes > n || !valid_level(level) {
        return invalid(level);
    }
    let a = alpha(level) / 2.0;
    let lower = if successes == 0 {
        0.0
    } else {
        match Beta::new(successes as f64, (n - successes + 1) as f64) {
            Ok(d) => d.inverse_cdf(a),
            Err(_) => f64::NAN,
        }
    };
    let upper = if successes == n {
        1.0
    } else {
        match Beta::new((successes + 1) as f64, (n - successes) as f64) {
            Ok(d) => d.inverse_cdf(1.0 - a),
            Err(_) => f64::NAN,
        }
    };
    ConfidenceInterval {
        lower,
        upper,
        level,
    }
}

/// Equal-tailed **Jeffreys** interval for a binomial proportion using the
/// `Beta(1/2, 1/2)` prior.
pub fn proportion_jeffreys(successes: u64, n: u64, level: f64) -> ConfidenceInterval {
    if n == 0 || successes > n || !valid_level(level) {
        return invalid(level);
    }
    let a = alpha(level) / 2.0;
    match Beta::new(successes as f64 + 0.5, (n - successes) as f64 + 0.5) {
        Ok(d) => ConfidenceInterval {
            lower: d.inverse_cdf(a),
            upper: d.inverse_cdf(1.0 - a),
            level,
        },
        Err(_) => invalid(level),
    }
}

/// Wald interval for the difference of two independent proportions, `p1 - p2`.
pub fn proportion_difference_wald(
    successes_a: u64,
    n_a: u64,
    successes_b: u64,
    n_b: u64,
    level: f64,
) -> ConfidenceInterval {
    if n_a == 0 || n_b == 0 || successes_a > n_a || successes_b > n_b || !valid_level(level) {
        return invalid(level);
    }
    let (pa, pb) = (
        successes_a as f64 / n_a as f64,
        successes_b as f64 / n_b as f64,
    );
    let z = z_quantile(1.0 - alpha(level) / 2.0);
    let half = z * (pa * (1.0 - pa) / n_a as f64 + pb * (1.0 - pb) / n_b as f64).sqrt();
    ConfidenceInterval {
        lower: pa - pb - half,
        upper: pa - pb + half,
        level,
    }
}

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

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn t_interval_brackets_the_mean() {
        let data = [4.8, 5.0, 5.2, 4.9, 5.1, 5.0];
        let ci = mean_t_interval(&data, 0.95);
        assert!(ci.contains(mean(&data)));
        assert!(ci.lower < mean(&data) && mean(&data) < ci.upper);
        // Higher confidence ⇒ wider interval.
        assert!(mean_t_interval(&data, 0.99).width() > ci.width());
    }

    #[test]
    fn z_interval_known_width() {
        // n=4, σ=2, 95% ⇒ half-width = 1.959964·2/2 = 1.959964.
        let data = [10.0, 10.0, 10.0, 10.0];
        let ci = mean_z_interval(&data, 2.0, 0.95);
        close(ci.upper - 10.0, 1.959_963_98, 1e-4);
    }

    #[test]
    fn wilson_stays_in_unit_interval_at_extremes() {
        // 0 successes of 10: Wald would dip below 0; Wilson stays valid and > 0 upper.
        let w = proportion_wilson(0, 10, 0.95);
        assert!(w.lower >= 0.0 && w.upper > 0.0 && w.upper < 1.0);
        // Half successes: centered near 0.5.
        let mid = proportion_wilson(5, 10, 0.95);
        close((mid.lower + mid.upper) / 2.0, 0.5, 0.02);
    }

    #[test]
    fn exact_and_approximate_proportion_intervals_are_valid() {
        let cp = proportion_clopper_pearson(0, 10, 0.95);
        assert_eq!(cp.lower, 0.0);
        assert!(cp.upper > 0.0 && cp.upper < 1.0);
        let j = proportion_jeffreys(5, 10, 0.95);
        assert!(j.contains(0.5));
        let ac = proportion_agresti_coull(5, 10, 0.95);
        assert!(ac.contains(0.5));
    }

    #[test]
    fn variance_and_poisson_intervals_are_ordered() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance_chi_square_interval(&data, 0.95);
        assert!(v.lower > 0.0 && v.lower < variance(&data) && variance(&data) < v.upper);
        let s = std_dev_chi_square_interval(&data, 0.95);
        close(s.lower * s.lower, v.lower, 1e-12);
        let rate = poisson_rate_interval(5, 2.0, 0.95);
        assert!(rate.lower >= 0.0 && rate.upper > rate.lower);
    }

    #[test]
    fn difference_intervals_center_on_observed_differences() {
        let a = [10.0, 11.0, 12.0, 13.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let ci = mean_difference_welch_interval(&a, &b, 0.95);
        close((ci.lower + ci.upper) / 2.0, mean(&a) - mean(&b), 1e-12);
        let pd = proportion_difference_wald(7, 10, 4, 10, 0.95);
        close((pd.lower + pd.upper) / 2.0, 0.3, 1e-12);
    }

    #[test]
    fn wald_proportion_matches_formula() {
        let ci = proportion_wald(30, 100, 0.95);
        let z = z_quantile(0.975);
        let half = z * (0.3 * 0.7 / 100.0_f64).sqrt();
        close(ci.lower, 0.3 - half, 1e-12);
        close(ci.upper, 0.3 + half, 1e-12);
    }
}
