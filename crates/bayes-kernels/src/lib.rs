//! Euclid-generic closed-form probability kernels.
//!
//! This module is the `Real`/`Complex` companion to the concrete `f64`
//! distribution spine. The functions here are pure closed-form kernels: they do
//! not validate parameters, check support membership, sample, branch on CDFs, or
//! perform eager special-function work. Scalar constants are lifted through
//! `Real::scalar`; callers provide precomputed special-function values such
//! as `ln Gamma(shape)` or `ln B(alpha, beta)`.
//!
//! That shape is intentional. Euclid's lazy traits are for tensor/expression
//! backends such as `lebesgue`, where comparisons and container construction may
//! not exist, but algebraic log-density formulas should still be reusable.
//!
//! # How to read this crate
//!
//! The root is a flat surface: `bayes_kernels::normal_log_pdf(...)`,
//! `bayes_kernels::kl_normal(...)`, and every other family kernel are re-exported
//! directly (see the `pub use crate::families::{...}` block below).
//!
//! If you are thinking in distributions, browse [`families`] instead: the
//! implementation files live under `families/continuous/*.rs` and
//! `families/discrete/*.rs`, reachable as `families::normal::*`,
//! `families::poisson::*`, and similar. A handful of algorithm modules carry
//! their own implementations rather than family kernels: [`correlation`]
//! (GP covariance functions), [`glm`] (link/score/curvature), [`quantile`] (the
//! generic bisection inverse-CDF), [`transform`] (reparameterization bijections),
//! and [`primitives`] (the normal/complex-normal log-probability core).
//!
//! Naming convention: `_on_support` and `_on_unit` kernels assume the caller has
//! already checked support membership. Full-domain wrappers encode support
//! behavior when that can be expressed with Euclid's generic scalar operations.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod primitives;

use euclid_numeric::Real;

pub use crate::primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

/// `ln I₀(x) = ln I₀ᵉ(x) + |x|` — the overflow-free Bessel log shared by the von Mises
/// kernels, spelled over the [`Real::ln_bessel_i0`] method.
fn ln_i0<T: Real>(x: T) -> T {
    x.ln_bessel_i0()
}

/// `x` clamped at zero **without swallowing a `NaN`**.
///
/// `max` implements IEEE `maxNum`, which *ignores* a `NaN` operand and returns
/// the other one — so `(NaN).max(0)` is `0`. Wherever a quantity is clamped at
/// zero because rounding can push it a hair negative, that silently converts
/// "no information" into "exactly zero", and zero is rarely a neutral answer: a
/// squared Hellinger distance of `0` means *identical distributions*, and a
/// mutual information of `0` means *independent*. `hellinger_sq(Normal(0,1),
/// Normal(0, NaN))` reported `0` for exactly this reason.
///
/// `select(x < 0, 0, x)` absorbs the same rounding noise and leaves a `NaN`
/// alone, because `NaN < 0` is false.
pub fn clamp_negative_to_zero<T: Real>(x: T) -> T {
    T::select(&x.lt(&T::zero()), T::zero(), x)
}

/// `H² = max(1 − BC, 0)` — squared Hellinger from a Bhattacharyya coefficient.
/// Shared by the per-family Hellinger kernels and the categorical reduction.
fn hellinger_sq_from_bc<T: Real>(bc: T) -> T {
    clamp_negative_to_zero(T::one() - bc)
}

// `families` is the implementation home (browse by distribution: `families::normal::*`).
// The remaining modules carry their own algorithms rather than family kernels.
pub mod correlation;
pub mod families;
pub mod glm;
pub mod quantile;
pub mod transform;

// Flat root surface: every closed-form family kernel, re-exported from `families` and
// grouped by distribution. This is the public `bayes_kernels::<fn>` API; the
// implementation home for each function is `families::<dist>`.
pub use crate::families::{
    bernoulli::{
        bernoulli_excess_kurtosis, bernoulli_log_pmf, bernoulli_logit_log_pmf, bernoulli_skewness,
        chi_squared_bernoulli, hellinger_sq_bernoulli, kl_bernoulli_on_unit, renyi_bernoulli,
        total_variation_bernoulli,
    },
    beta::{
        beta_bernoulli_log_marginal_likelihood, beta_bernoulli_log_predictive,
        beta_binomial_log_predictive, beta_cdf_on_unit, beta_entropy, beta_excess_kurtosis,
        beta_log_pdf_on_unit, beta_skewness, beta_variance, hellinger_sq_beta, kl_beta,
    },
    beta_binomial::beta_binomial_log_pmf,
    binomial::{
        binomial_cdf_on_support, binomial_log_pmf, binomial_skewness, chi_squared_binomial,
        hellinger_sq_binomial, kl_binomial, renyi_binomial,
    },
    categorical::categorical_log_pmf,
    cauchy::{cauchy_cdf, cauchy_entropy, cauchy_log_pdf, cauchy_quantile, kl_cauchy},
    chi_squared::{
        chi_squared_cdf, chi_squared_cdf_on_support, chi_squared_entropy,
        chi_squared_excess_kurtosis, chi_squared_log_pdf_on_support, chi_squared_skewness,
    },
    discrete_uniform::{
        chi_squared_discrete_uniform_on_support, discrete_uniform_entropy,
        discrete_uniform_log_pmf_on_support, hellinger_sq_discrete_uniform,
        kl_discrete_uniform_on_support, renyi_discrete_uniform_on_support,
    },
    exponential::{
        chi_squared_exponential_on_support, exponential_cdf, exponential_cdf_on_support,
        exponential_entropy, exponential_ln_cdf, exponential_ln_cdf_on_support, exponential_ln_sf,
        exponential_log_pdf_on_support, exponential_log_survival_on_support, exponential_quantile,
        hellinger_sq_exponential, kl_exponential,
    },
    fisher_snedecor::{
        fisher_snedecor_cdf, fisher_snedecor_cdf_on_support, fisher_snedecor_log_pdf_on_support,
    },
    frechet::{
        frechet_cdf, frechet_cdf_on_support, frechet_ln_cdf, frechet_log_pdf_on_support,
        frechet_quantile,
    },
    gamma::{
        chi_squared_gamma_on_support, gamma_cdf, gamma_cdf_on_support, gamma_entropy,
        gamma_excess_kurtosis, gamma_exponential_log_marginal_likelihood,
        gamma_exponential_log_predictive, gamma_log_pdf_on_support,
        gamma_poisson_log_marginal_likelihood, gamma_poisson_log_predictive, gamma_skewness,
        gamma_variance, hellinger_sq_gamma, kl_gamma, renyi_gamma_on_support,
    },
    geometric::{
        geometric_excess_kurtosis, geometric_failures_log_pmf, geometric_log_pmf,
        geometric_skewness, kl_geometric,
    },
    gumbel::{
        gumbel_cdf, gumbel_entropy, gumbel_excess_kurtosis, gumbel_ln_cdf, gumbel_log_pdf,
        gumbel_quantile, kl_gumbel,
    },
    half_normal::{
        half_normal_cdf, half_normal_cdf_on_support, half_normal_entropy,
        half_normal_excess_kurtosis, half_normal_log_pdf_on_support, half_normal_quantile,
        half_normal_skewness,
    },
    hypergeometric::hypergeometric_log_pmf,
    inverse_gamma::{inverse_gamma_cdf, inverse_gamma_entropy, inverse_gamma_log_pdf_on_support},
    laplace::{kl_laplace, laplace_cdf, laplace_entropy, laplace_log_pdf, laplace_quantile},
    logistic::{
        logistic_cdf, logistic_entropy, logistic_ln_cdf, logistic_ln_sf, logistic_log_pdf,
        logistic_quantile,
    },
    lognormal::{
        lognormal_cdf, lognormal_entropy, lognormal_excess_kurtosis, lognormal_log_pdf_on_support,
        lognormal_mean, lognormal_quantile, lognormal_skewness, lognormal_variance,
    },
    lomax::{
        lomax_cdf, lomax_cdf_on_support, lomax_ln_sf, lomax_log_pdf_on_support, lomax_quantile,
    },
    negative_binomial::{
        chi_squared_negative_binomial_on_support, hellinger_sq_negative_binomial,
        kl_negative_binomial, negative_binomial_cdf, negative_binomial_excess_kurtosis,
        negative_binomial_log_pmf, negative_binomial_skewness, renyi_negative_binomial_on_support,
    },
    normal::{
        chi_squared_normal_on_support, hellinger_sq_normal, kl_normal, normal_cdf, normal_entropy,
        normal_inverse_gamma_log_marginal_likelihood,
        normal_known_variance_log_marginal_likelihood, normal_ln_cdf, normal_ln_sf, normal_log_pdf,
        normal_quantile, renyi_normal_on_support,
    },
    order_statistic::order_statistic_cdf,
    pareto::{
        kl_pareto_on_support, pareto_cdf, pareto_cdf_on_support, pareto_entropy,
        pareto_log_pdf_on_support, pareto_quantile,
    },
    poisson::{
        chi_squared_poisson, hellinger_sq_poisson, kl_poisson, poisson_cdf,
        poisson_excess_kurtosis, poisson_log_pmf, poisson_skewness,
    },
    rayleigh::{
        kl_rayleigh, rayleigh_cdf, rayleigh_cdf_on_support, rayleigh_entropy,
        rayleigh_excess_kurtosis, rayleigh_log_pdf_on_support, rayleigh_quantile,
        rayleigh_skewness,
    },
    student_t::{student_t_log_pdf, students_t_cdf, students_t_entropy},
    triangular::{triangular_cdf, triangular_entropy, triangular_pdf, triangular_quantile},
    uniform::{
        chi_squared_uniform_on_support, hellinger_sq_uniform, kl_uniform_on_support,
        renyi_uniform_on_support, uniform_cdf, uniform_entropy, uniform_log_pdf_on_support,
        uniform_quantile,
    },
    von_mises::{
        chi_squared_von_mises, hellinger_sq_von_mises, kl_von_mises, renyi_von_mises,
        von_mises_entropy, von_mises_log_pdf,
    },
    wald::{
        chi_squared_wald_on_support, hellinger_sq_wald, kl_wald, renyi_wald_on_support, wald_cdf,
        wald_excess_kurtosis, wald_log_pdf_on_support, wald_skewness,
    },
    weibull::{
        kl_weibull, weibull_cdf, weibull_cdf_on_support, weibull_entropy,
        weibull_log_pdf_on_support, weibull_quantile,
    },
};

pub use correlation::*;
pub use glm::*;
pub use quantile::*;
pub use transform::*;

/// Upper-tail kernels, grouped because they exist for one reason.
///
/// `ContinuousCdf::sf` is **derived** from `ln_sf`, and `ln_sf`'s own default
/// goes through `1 - cdf` — which is exactly zero for every argument past the
/// point where the CDF rounds to 1, around a survival of `1e-16`. These are the
/// stable closed forms that make the region beyond it computable: a chi-squared
/// p-value of `1e-40`, a Weibull's survival at ten times its scale, a normal's
/// `sf(9)`. The `*_isf` entries are the same argument for the inverse, which
/// otherwise has to form `1 - p`.
pub use crate::families::{
    cauchy::cauchy_ln_sf,
    chi_squared::chi_squared_ln_sf,
    exponential::exponential_isf,
    fisher_snedecor::fisher_snedecor_ln_sf,
    frechet::frechet_ln_sf,
    gamma::gamma_ln_sf,
    gumbel::gumbel_ln_sf,
    half_normal::{half_normal_isf, half_normal_ln_sf},
    inverse_gamma::inverse_gamma_ln_sf,
    laplace::laplace_ln_sf,
    lognormal::lognormal_ln_sf,
    pareto::{pareto_isf, pareto_ln_sf},
    rayleigh::{rayleigh_isf, rayleigh_ln_sf},
    student_t::students_t_ln_sf,
    wald::wald_ln_sf,
    weibull::{weibull_isf, weibull_ln_sf},
};
