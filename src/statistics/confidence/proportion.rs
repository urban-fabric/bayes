//! Confidence intervals for **binomial proportions**: Wald, Wilson score,
//! Agresti–Coull, exact Clopper–Pearson, and Jeffreys for a single proportion,
//! plus the two-sample Wald interval for a difference of proportions.

use super::{ConfidenceInterval, alpha, invalid, valid_level, z_quantile};
use crate::distribution::{Beta, ContinuousCdf};

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn wilson_stays_in_unit_interval_at_extremes() {
        let w = proportion_wilson(0, 10, 0.95);
        assert!(w.lower >= 0.0 && w.upper > 0.0 && w.upper < 1.0);
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
    fn proportion_difference_interval_centers_on_observed_difference() {
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
