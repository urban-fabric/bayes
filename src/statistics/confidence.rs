//! **Closed-form confidence intervals** — the frequentist companions to the point
//! estimates in [`summary`](super::summary). Each returns a [`ConfidenceInterval`]
//! at a requested coverage `level` (e.g. `0.95`), computed from an analytic
//! quantile — no resampling (for that, see [`resampling`](super::resampling)).

use super::summary::{mean, variance};
use crate::distribution::{ContinuousCdf, StudentT};
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

/// Student-t interval for the **mean** with unknown variance (the usual small-sample
/// interval): `x̄ ± t_{1−α/2, n−1} · s/√n`.
pub fn mean_t_interval(data: &[f64], level: f64) -> ConfidenceInterval {
    let n = data.len();
    if n < 2 || !(0.0..1.0).contains(&level) {
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
    if n == 0 || sigma <= 0.0 || !(0.0..1.0).contains(&level) {
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
    if n == 0 || successes > n || !(0.0..1.0).contains(&level) {
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
    if n == 0 || successes > n || !(0.0..1.0).contains(&level) {
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
    fn wald_proportion_matches_formula() {
        let ci = proportion_wald(30, 100, 0.95);
        let z = z_quantile(0.975);
        let half = z * (0.3 * 0.7 / 100.0_f64).sqrt();
        close(ci.lower, 0.3 - half, 1e-12);
        close(ci.upper, 0.3 + half, 1e-12);
    }
}
