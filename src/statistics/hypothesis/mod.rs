//! **Hypothesis tests** — each returns a [`TestResult`] (test statistic + two-sided
//! p-value, computed from `bayes`'s own distribution CDFs). They *test data against a
//! null hypothesis* — descriptive/frequentist statistics, not the iterative parameter
//! inference that is out of scope.
//!
//! Grouped by family: `parametric` (z / t / Welch / ANOVA), `contingency`
//! (χ² goodness-of-fit & independence, Fisher's exact), `nonparametric`
//! (Mann–Whitney, Wilcoxon), and `goodness_of_fit` (Kolmogorov–Smirnov,
//! Jarque–Bera). Shared p-value helpers live here and are used across the groups.

use std::f64::consts::SQRT_2;

use crate::distribution::{ChiSquared, ContinuousCdf, StudentT};
use crate::function::erf::erfc;

mod contingency;
mod goodness_of_fit;
mod nonparametric;
mod parametric;

pub use contingency::{chi_squared_goodness_of_fit, chi_squared_independence, fishers_exact};
pub use goodness_of_fit::{jarque_bera, ks_one_sample, ks_two_sample};
pub use nonparametric::{mann_whitney_u, wilcoxon_signed_rank};
pub use parametric::{one_sample_t, one_sample_z, one_way_anova, two_sample_t, welch_t};

/// The outcome of a hypothesis test: the test statistic and its p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

/// The "undefined" result returned for degenerate inputs (too few points, mismatched
/// shapes, …) — both fields `NaN`, never a panic.
fn nan_result() -> TestResult {
    TestResult {
        statistic: f64::NAN,
        p_value: f64::NAN,
    }
}

/// Standard-normal CDF `Φ(z)` (via `erfc`, no allocation).
fn std_normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / SQRT_2)
}

/// Two-sided p-value for a standard-normal statistic.
fn z_two_sided(z: f64) -> f64 {
    2.0 * std_normal_cdf(-z.abs())
}

/// Two-sided p-value for a Student-t statistic with `df` degrees of freedom.
fn t_two_sided(t: f64, df: f64) -> f64 {
    match StudentT::new(df) {
        Ok(d) => 2.0 * d.sf(t.abs()),
        Err(_) => f64::NAN,
    }
}

/// Upper-tail p-value for a χ² statistic with `df` degrees of freedom.
fn chi2_upper(x: f64, df: f64) -> f64 {
    match ChiSquared::new(df) {
        Ok(d) => d.sf(x),
        Err(_) => f64::NAN,
    }
}
