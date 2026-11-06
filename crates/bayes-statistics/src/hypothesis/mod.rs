//! **Hypothesis tests** — each returns a [`TestResult`] (test statistic + two-sided
//! p-value, computed from `bayes`'s own distribution CDFs or exact finite sums).
//! They *test data against a null hypothesis* — descriptive/frequentist statistics,
//! not the iterative parameter inference that is out of scope.
//!
//! Grouped by family: `parametric` (z / t / paired t / Welch / F / ANOVA /
//! Pearson correlation), `contingency` (χ² goodness-of-fit & independence,
//! Fisher's exact, exact binomial, McNemar), `nonparametric` (Mann–Whitney,
//! Wilcoxon, sign test), and `goodness_of_fit` (Kolmogorov–Smirnov,
//! Jarque–Bera). Shared p-value helpers live here and are used across the groups.

use std::f64::consts::SQRT_2;

use bayes_distribution_ext::{ChiSquared, ContinuousCdf, StudentT};
use euclid_numeric::{Natural, Real};

mod contingency;
mod goodness_of_fit;
mod nonparametric;
mod parametric;

pub use contingency::{
    binomial_test, chi_squared_goodness_of_fit, chi_squared_independence, fishers_exact, mcnemar,
};
pub use goodness_of_fit::{jarque_bera, ks_one_sample, ks_two_sample};
pub use nonparametric::{mann_whitney_u, sign_test, wilcoxon_signed_rank};
pub use parametric::{
    one_sample_t, one_sample_z, one_way_anova, paired_t, pearson_correlation_test, two_sample_t,
    variance_ratio_f, welch_t,
};

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
    0.5 * (-z / SQRT_2).erfc()
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

fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    (n.ln_binomial(&k) + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

fn binomial_two_sided_p_value(successes: u64, n: u64, p: f64) -> f64 {
    if successes > n || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let observed = binomial_pmf(n, successes, p);
    if observed.is_nan() {
        return f64::NAN;
    }
    let mut p_value = 0.0;
    for k in 0..=n {
        let pk = binomial_pmf(n, k, p);
        if pk <= observed * (1.0 + 1e-12) {
            p_value += pk;
        }
    }
    p_value.min(1.0)
}
