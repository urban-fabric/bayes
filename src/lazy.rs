//! Lazy numeric probability kernels.
//!
//! This module is the `LazyReal`/`LazyComplex` companion to the concrete `f64`
//! distribution spine. The functions here are pure closed-form kernels: they do
//! not validate parameters, check support membership, sample, branch on CDFs, or
//! perform eager special-function work. Scalar constants are lifted through
//! `LazyReal::scalar`; callers provide precomputed special-function values such
//! as `ln Gamma(shape)` or `ln B(alpha, beta)`.
//!
//! That shape is intentional. Euclid's lazy traits are for tensor/expression
//! backends such as `lebesgue`, where comparisons and container construction may
//! not exist, but algebraic log-density formulas should still be reusable.

use euclid::numeric::LazyReal;

pub use crate::primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

/// Evaluate `exp(ln_density)`.
pub fn density_from_log<T: LazyReal>(ln_density: T) -> T {
    ln_density.exp()
}

/// Square a lazy scalar.
pub fn square<T: LazyReal>(x: T) -> T {
    x.clone() * x
}

/// Logistic sigmoid `1 / (1 + exp(-x))`.
pub fn sigmoid<T: LazyReal>(x: T) -> T {
    let one = T::one();
    one.clone() / (one + (-x).exp())
}

/// Logit `log(p / (1 - p))`, the inverse of [`sigmoid`] on `0 < p < 1`.
///
/// Written as `log(p) - log(1 - p)` so it composes from the lazy log directly.
/// Range membership is the caller's responsibility.
pub fn logit<T: LazyReal>(p: T) -> T {
    let one = T::one();
    p.clone().ln() - (one - p).ln()
}

/// Bernoulli log-mass for a probability parameter and a relaxed label in `[0, 1]`:
/// `y log(p) + (1-y) log(1-p)`.
///
/// This is useful for differentiable losses as well as exact labels. Support and
/// range checks are the caller's responsibility.
pub fn bernoulli_log_pmf<T: LazyReal>(y: T, p: T) -> T {
    let one = T::one();
    y.clone() * p.clone().ln() + (one.clone() - y) * (one - p).ln()
}

/// Bernoulli log-mass parameterized by a logit:
/// `y * logit - log(1 + exp(logit))`.
pub fn bernoulli_logit_log_pmf<T: LazyReal>(y: T, logit: T) -> T {
    y * logit.clone() - logit.ln_1p_exp()
}

/// Binomial log-mass for supplied success/failure counts:
/// `ln C(n,k) + k log(p) + (n-k) log(1-p)`.
pub fn binomial_log_pmf<T: LazyReal>(successes: T, failures: T, p: T, ln_binomial: T) -> T {
    let one = T::one();
    ln_binomial + successes * p.clone().ln() + failures * (one - p).ln()
}

/// Categorical log-mass for the selected class probability.
pub fn categorical_log_pmf<T: LazyReal>(selected_probability: T) -> T {
    selected_probability.ln()
}

/// Geometric log-mass for Bayes' trial-count parameterization:
/// `(trials - 1) * log(1-p) + log(p)`.
pub fn geometric_log_pmf<T: LazyReal>(trials: T, p: T) -> T {
    let one = T::one();
    (trials - one.clone()) * (one - p.clone()).ln() + p.ln()
}

/// Geometric log-mass for `failures` failures before the first success:
/// `failures * log(1-p) + log(p)`.
pub fn geometric_failures_log_pmf<T: LazyReal>(failures: T, p: T) -> T {
    let one = T::one();
    failures * (one - p.clone()).ln() + p.ln()
}

/// Poisson log-mass: `k log(lambda) - lambda - ln(k!)`.
pub fn poisson_log_pmf<T: LazyReal>(k: T, lambda: T, ln_factorial_k: T) -> T {
    k * lambda.clone().ln() - lambda - ln_factorial_k
}

/// Negative-binomial log-mass in the "failures before r successes"
/// parameterization:
/// `ln C(k+r-1,k) + r log(p) + k log(1-p)`.
pub fn negative_binomial_log_pmf<T: LazyReal>(
    failures: T,
    successes_until_stop: T,
    p: T,
    ln_coefficient: T,
) -> T {
    let one = T::one();
    ln_coefficient + successes_until_stop * p.clone().ln() + failures * (one - p).ln()
}

/// Beta-binomial log-mass:
/// `ln C(n,k) + ln B(k+alpha, n-k+beta) - ln B(alpha,beta)`.
pub fn beta_binomial_log_pmf<T: LazyReal>(
    ln_binomial: T,
    ln_beta_posterior: T,
    ln_beta_prior: T,
) -> T {
    ln_binomial + ln_beta_posterior - ln_beta_prior
}

/// Hypergeometric log-mass for drawing `k` successes:
/// `ln C(K,k) + ln C(N-K, n-k) - ln C(N,n)`.
///
/// The three log-binomials are caller supplied — `ln_binomial_successes` is
/// `ln C(K,k)` over the `K` successes, `ln_binomial_failures` is
/// `ln C(N-K, n-k)` over the `N-K` failures, and `ln_binomial_population` is
/// `ln C(N,n)` over the whole population draw.
pub fn hypergeometric_log_pmf<T: LazyReal>(
    ln_binomial_successes: T,
    ln_binomial_failures: T,
    ln_binomial_population: T,
) -> T {
    ln_binomial_successes + ln_binomial_failures - ln_binomial_population
}

/// Uniform log-density on its support: `-log(width)`.
pub fn uniform_log_pdf_on_support<T: LazyReal>(width: T) -> T {
    -width.ln()
}

/// Discrete-uniform log-mass on its support: `-log(count)`.
pub fn discrete_uniform_log_pmf_on_support<T: LazyReal>(count: T) -> T {
    -count.ln()
}

/// Normal log-density over the lazy real surface.
pub fn normal_log_pdf<T: LazyReal>(x: T, mean: T, std_dev: T) -> T {
    normal_log_prob(x, mean, std_dev)
}

/// Exponential log-density on `x >= 0`: `log(rate) - rate*x`.
pub fn exponential_log_pdf_on_support<T: LazyReal>(x: T, rate: T) -> T {
    rate.clone().ln() - rate * x
}

/// Exponential log-survival on `x >= 0`: `-rate*x`.
pub fn exponential_log_survival_on_support<T: LazyReal>(x: T, rate: T) -> T {
    -(rate * x)
}

/// Gamma log-density on `x > 0`.
///
/// `ln_gamma_shape` is `ln Gamma(shape)`.
pub fn gamma_log_pdf_on_support<T: LazyReal>(x: T, shape: T, rate: T, ln_gamma_shape: T) -> T {
    let one = T::one();
    shape.clone() * rate.clone().ln() - ln_gamma_shape + (shape - one) * x.clone().ln() - rate * x
}

/// Chi-squared log-density on `x > 0`.
///
/// This is the gamma kernel with `shape = freedom / 2` and `rate = 1/2`; callers
/// provide those lazy values and `ln_gamma_shape`.
pub fn chi_squared_log_pdf_on_support<T: LazyReal>(
    x: T,
    half_freedom: T,
    ln_gamma_half_freedom: T,
) -> T {
    gamma_log_pdf_on_support(x, half_freedom, T::half(), ln_gamma_half_freedom)
}

/// Beta log-density on `0 < x < 1`.
///
/// `ln_beta_ab` is `ln B(alpha, beta)`.
pub fn beta_log_pdf_on_unit<T: LazyReal>(x: T, alpha: T, beta: T, ln_beta_ab: T) -> T {
    let one = T::one();
    (alpha - one.clone()) * x.clone().ln() + (beta - one.clone()) * (one - x).ln() - ln_beta_ab
}

/// Dirichlet log-density on the simplex.
///
/// `ln_multivariate_beta` is `ln B(alpha)`. Returns `None` when the dimensions do
/// not match.
pub fn dirichlet_log_pdf_on_simplex<T: LazyReal>(
    x: &[T],
    alpha: &[T],
    ln_multivariate_beta: T,
) -> Option<T> {
    if x.len() != alpha.len() {
        return None;
    }

    let one = T::one();
    let mut terms = x
        .iter()
        .cloned()
        .zip(alpha.iter().cloned())
        .map(|(xi, ai)| (ai - one.clone()) * xi.ln());
    let first = terms.next();
    Some(match first {
        Some(head) => terms.fold(head, |acc, term| acc + term) - ln_multivariate_beta,
        None => -ln_multivariate_beta,
    })
}

/// Log-normal log-density on `x > 0`.
pub fn lognormal_log_pdf_on_support<T: LazyReal>(x: T, mean_log: T, std_log: T) -> T {
    let log_x = x.clone().ln();
    let z = (log_x.clone() - mean_log) / std_log.clone();
    -(T::half() * z.clone() * z) - log_x - std_log.ln() - T::half_ln_tau()
}

/// Logistic-distribution log-density on the real line.
pub fn logistic_log_pdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    let log_denom = (-z.clone()).ln_1p_exp();
    -scale.ln() - z - log_denom.clone() - log_denom
}

/// Laplace log-density on the real line.
pub fn laplace_log_pdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    T::half().ln() - scale.clone().ln() - (x - location).abs() / scale
}

/// Cauchy log-density on the real line.
pub fn cauchy_log_pdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -T::ln_pi() - scale.ln() - (T::one() + z.clone() * z).ln()
}

/// Student-t log-density for a location/scale t distribution.
///
/// `log_normalizer` is
/// `0.5*log(freedom*pi) + ln_gamma(freedom/2) - ln_gamma((freedom+1)/2)`.
pub fn student_t_log_pdf<T: LazyReal>(
    x: T,
    location: T,
    scale: T,
    freedom: T,
    log_normalizer: T,
) -> T {
    let z = (x - location) / scale.clone();
    let one = T::one();
    let inner = one.clone() + (z.clone() * z) / freedom.clone();
    -scale.ln() - log_normalizer - T::half() * (freedom + one) * inner.ln()
}

/// Fisher-Snedecor F log-density on `x > 0`.
///
/// `half_d1 = d1/2`, `half_d2 = d2/2`, and `ln_beta_half_degrees` is
/// `ln B(d1/2, d2/2)`.
pub fn fisher_snedecor_log_pdf_on_support<T: LazyReal>(
    x: T,
    d1: T,
    d2: T,
    half_d1: T,
    half_d2: T,
    ln_beta_half_degrees: T,
) -> T {
    let d1x = d1.clone() * x.clone();
    half_d1.clone() * d1x.clone().ln() + half_d2.clone() * d2.clone().ln()
        - (half_d1 + half_d2) * (d1x + d2).ln()
        - x.ln()
        - ln_beta_half_degrees
}

/// Wald / inverse-Gaussian log-density on `x > 0`.
pub fn wald_log_pdf_on_support<T: LazyReal>(x: T, mean: T, shape: T) -> T {
    let delta = x.clone() - mean.clone();
    T::half() * (shape.clone().ln() - T::ln_tau() - T::scalar(3.0) * x.clone().ln())
        - shape * delta.clone() * delta / (T::two() * mean.clone() * mean * x)
}

/// Pareto log-density on `x >= minimum`.
pub fn pareto_log_pdf_on_support<T: LazyReal>(x: T, minimum: T, shape: T) -> T {
    let one = T::one();
    shape.clone().ln() + shape.clone() * minimum.ln() - (shape + one) * x.ln()
}

/// Lomax log-density on `x >= 0`.
pub fn lomax_log_pdf_on_support<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    let one = T::one();
    shape.clone().ln() - scale.clone().ln() - (shape + one.clone()) * (one + x / scale).ln()
}

/// Inverse-gamma log-density on `x > 0`.
///
/// `ln_gamma_shape` is `ln Gamma(shape)`.
pub fn inverse_gamma_log_pdf_on_support<T: LazyReal>(
    x: T,
    shape: T,
    scale: T,
    ln_gamma_shape: T,
) -> T {
    let one = T::one();
    shape.clone() * scale.clone().ln() - ln_gamma_shape - (shape + one) * x.clone().ln() - scale / x
}

/// Rayleigh log-density on `x >= 0`.
pub fn rayleigh_log_pdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    x.clone().ln() - scale.clone().ln() - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Half-normal log-density on `x >= 0`.
pub fn half_normal_log_pdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    let log_sqrt_2_over_pi = T::scalar(0.5 * (2.0 / core::f64::consts::PI).ln());
    log_sqrt_2_over_pi - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Weibull log-density on `x >= 0`.
pub fn weibull_log_pdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    shape.clone().ln() - scale.ln() + (shape.clone() - T::one()) * log_y.clone()
        - (shape * log_y).exp()
}

/// Frechet log-density on `x > 0`.
pub fn frechet_log_pdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    let y_neg_shape = (-(shape.clone()) * log_y.clone()).exp();
    shape.clone().ln() - scale.ln() - (shape + T::one()) * log_y - y_neg_shape
}

/// Gumbel log-density on the real line.
pub fn gumbel_log_pdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -scale.ln() - z.clone() - (-z).exp()
}

/// Von Mises (circular normal) log-density on `[−π, π)`:
/// `κ·cos(x − µ) − log_normalizer`.
///
/// `log_normalizer` is `ln(2π·I₀(κ))`, the circular normalizer the caller
/// precomputes from the modified Bessel `I₀`. The kernel itself is pure analysis:
/// it leans on the lazy `cos`, which the `LazyReal` surface now carries, so the
/// circular family is reachable on every backend.
pub fn von_mises_log_pdf<T: LazyReal>(x: T, mean: T, kappa: T, log_normalizer: T) -> T {
    kappa * (x - mean).cos() - log_normalizer
}

/// Sum independent log terms. Returns `None` for an empty iterator because
/// the caller may prefer to distinguish "no terms" from the multiplicative
/// identity density.
pub fn sum_log_terms<T: LazyReal>(terms: impl IntoIterator<Item = T>) -> Option<T> {
    let mut terms = terms.into_iter();
    let first = terms.next()?;
    Some(terms.fold(first, |acc, term| acc + term))
}

/// Diagonal multivariate normal log-density.
///
/// Returns `None` when the dimensions do not match or are empty.
pub fn diagonal_normal_log_pdf<T: LazyReal>(x: &[T], mean: &[T], std_dev: &[T]) -> Option<T> {
    if x.len() != mean.len() || x.len() != std_dev.len() || x.is_empty() {
        return None;
    }

    sum_log_terms(
        x.iter()
            .cloned()
            .zip(mean.iter().cloned())
            .zip(std_dev.iter().cloned())
            .map(|((xi, mi), si)| normal_log_pdf(xi, mi, si)),
    )
}

/// Inner product `Σ aᵢ·bᵢ` of two equal-length lazy vectors. Returns `None` for an
/// empty input, matching [`sum_log_terms`]'s "no terms" convention.
fn dot_product<T: LazyReal>(a: &[T], b: &[T]) -> Option<T> {
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(ai, bi)| ai * bi)
        .reduce(|acc, term| acc + term)
}

/// Multinomial log-mass for a category-count vector and per-category log
/// probabilities: `ln_multinomial_coefficient + Σ kᵢ·log(pᵢ)`.
///
/// `ln_multinomial_coefficient` is `ln(n!) - Σ ln(kᵢ!)`, supplied by the caller so
/// no factorial work hides in the kernel. Returns `None` when the counts and log
/// probabilities have mismatched or empty lengths.
pub fn multinomial_log_pmf<T: LazyReal>(
    counts: &[T],
    log_probabilities: &[T],
    ln_multinomial_coefficient: T,
) -> Option<T> {
    if counts.len() != log_probabilities.len() {
        return None;
    }
    let weighted = dot_product(counts, log_probabilities)?;
    Some(ln_multinomial_coefficient + weighted)
}

/// Dirichlet-multinomial log-mass:
/// `ln_multinomial_coefficient + ln B(α + counts) - ln B(α)`.
///
/// `ln_multinomial_coefficient` is `ln(n!) - Σ ln(kᵢ!)`, `ln_beta_posterior` is
/// the log multivariate beta `ln B(α + counts)`, and `ln_beta_prior` is `ln B(α)`.
pub fn dirichlet_multinomial_log_pmf<T: LazyReal>(
    ln_multinomial_coefficient: T,
    ln_beta_posterior: T,
    ln_beta_prior: T,
) -> T {
    ln_multinomial_coefficient + ln_beta_posterior - ln_beta_prior
}

/// Multivariate-normal log-density from a precomputed Mahalanobis term:
/// `-½·(mahalanobis_sq + ln_det_2pi)`.
///
/// `mahalanobis_sq` is `(x-µ)ᵀ Σ⁻¹ (x-µ)` and `ln_det_2pi` is
/// `d·ln(2π) + ln|Σ|`. The quadratic form and log-determinant are the caller's
/// linear-algebra responsibility; the kernel is only the closing algebra.
pub fn multivariate_normal_log_pdf<T: LazyReal>(mahalanobis_sq: T, ln_det_2pi: T) -> T {
    -T::half() * (mahalanobis_sq + ln_det_2pi)
}

/// Multivariate Student-t log-density from a precomputed Mahalanobis term:
/// `log_normalizer - ½·(ν+d)·log(1 + mahalanobis_sq/ν)`.
///
/// `mahalanobis_sq` is `(x-µ)ᵀ Σ⁻¹ (x-µ)`, `freedom` is `ν`, `dimension` is `d`,
/// and `log_normalizer` is
/// `lnΓ((ν+d)/2) - lnΓ(ν/2) - ½·(d·ln(νπ) + ln|Σ|)`.
pub fn multivariate_student_t_log_pdf<T: LazyReal>(
    mahalanobis_sq: T,
    freedom: T,
    dimension: T,
    log_normalizer: T,
) -> T {
    let one = T::one();
    log_normalizer
        - T::half() * (freedom.clone() + dimension) * (one + mahalanobis_sq / freedom).ln()
}

/// von Mises-Fisher log-density from the precomputed mean cosine `µ·x` on the
/// sphere: `log_normalizer + κ·(µ·x)`.
///
/// `log_normalizer` is `ln C_d(κ)`, the normalized vMF constant for the ambient
/// dimension. The directional inner product is the caller's geometry.
pub fn von_mises_fisher_log_pdf<T: LazyReal>(mean_dot: T, kappa: T, log_normalizer: T) -> T {
    log_normalizer + kappa * mean_dot
}

/// Watson (antipodally symmetric) log-density from the precomputed axis cosine
/// `µ·x`: `κ·(µ·x)² - log_partition`.
///
/// `log_partition` is `ln Z(κ)`. The axial inner product is the caller's geometry.
pub fn watson_log_pdf<T: LazyReal>(axis_dot: T, kappa: T, log_partition: T) -> T {
    kappa * axis_dot.clone() * axis_dot - log_partition
}

// --- closed-form CDFs over the lazy analysis surface ---
//
// `_on_support` kernels carry only the interior closed form: a CDF is `0` below its
// support and `1` above it, and that indicator is the caller's branch. The ones
// reaching for `erf` / the regularized incomplete gamma / beta were unreachable
// before the special functions joined the lazy surface; each lowers to the matching
// tensor op on a backend.

/// Normal CDF `Φ((x − µ)/σ) = ½·erfc(−(x − µ)/(σ·√2))`, on the whole real line.
pub fn normal_cdf<T: LazyReal>(x: T, mean: T, std_dev: T) -> T {
    T::half() * (-(x - mean) / (std_dev * T::sqrt_2())).erfc()
}

/// Half-normal CDF on `x ≥ 0`: `erf(x / (σ·√2))`.
pub fn half_normal_cdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    (x / (scale * T::sqrt_2())).erf()
}

/// Exponential CDF on `x ≥ 0`: `1 − exp(−rate·x)`.
pub fn exponential_cdf_on_support<T: LazyReal>(x: T, rate: T) -> T {
    T::one() - (-(rate * x)).exp()
}

/// Logistic CDF on the real line: `σ((x − location)/scale)`.
pub fn logistic_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    sigmoid((x - location) / scale)
}

/// Gumbel CDF on the real line: `exp(−exp(−z))`, `z = (x − location)/scale`.
pub fn gumbel_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    (-((-z).exp())).exp()
}

/// Weibull CDF on `x ≥ 0`: `1 − exp(−(x/scale)^shape)`.
pub fn weibull_cdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let powered = (shape * (x / scale).ln()).exp();
    T::one() - (-powered).exp()
}

/// Frechet CDF on `x > 0`: `exp(−(scale/x)^shape)`.
pub fn frechet_cdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    (-((shape * (scale / x).ln()).exp())).exp()
}

/// Rayleigh CDF on `x ≥ 0`: `1 − exp(−x²/(2σ²))`.
pub fn rayleigh_cdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    T::one() - (-(x.clone() * x) / (T::two() * scale.clone() * scale)).exp()
}

/// Pareto CDF on `x ≥ minimum`: `1 − (minimum/x)^shape`.
pub fn pareto_cdf_on_support<T: LazyReal>(x: T, minimum: T, shape: T) -> T {
    T::one() - (shape * (minimum / x).ln()).exp()
}

/// Lomax CDF on `x ≥ 0`: `1 − (1 + x/scale)^(−shape)`.
pub fn lomax_cdf_on_support<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    T::one() - (-(shape * (T::one() + x / scale).ln())).exp()
}

/// Gamma CDF on `x > 0`: the regularized lower incomplete gamma `P(shape, rate·x)`.
pub fn gamma_cdf_on_support<T: LazyReal>(x: T, shape: T, rate: T) -> T {
    (rate * x).gamma_lr(shape)
}

/// Chi-squared CDF on `x > 0`: the gamma CDF with `shape = freedom/2`, `rate = ½`.
pub fn chi_squared_cdf_on_support<T: LazyReal>(x: T, half_freedom: T) -> T {
    gamma_cdf_on_support(x, half_freedom, T::half())
}

/// Beta CDF on `0 ≤ x ≤ 1`: the regularized incomplete beta `Iₓ(α, β)`.
pub fn beta_cdf_on_unit<T: LazyReal>(x: T, alpha: T, beta: T) -> T {
    x.beta_reg(alpha, beta)
}

/// Student-t CDF on the real line.
///
/// Written branch-free as `½ + ½·sign(z)·(1 − I_w(ν/2, ½))`, with `z = (x−loc)/scale`
/// and `w = ν/(ν + z²)`, so it lowers to a `Sign` / `RegularizedIncompleteBeta` pair.
pub fn students_t_cdf<T: LazyReal>(x: T, location: T, scale: T, freedom: T) -> T {
    let z = (x - location) / scale;
    let w = freedom.clone() / (freedom.clone() + z.clone() * z.clone());
    let ix = w.beta_reg(freedom * T::half(), T::half());
    T::half() + T::half() * z.signum() * (T::one() - ix)
}

/// Fisher–Snedecor F CDF on `x > 0`: `I_{d1·x/(d1·x + d2)}(d1/2, d2/2)`.
pub fn fisher_snedecor_cdf_on_support<T: LazyReal>(x: T, d1: T, d2: T) -> T {
    let d1x = d1.clone() * x;
    (d1x.clone() / (d1x + d2.clone())).beta_reg(d1 * T::half(), d2 * T::half())
}

/// Poisson CDF at integer `k`: `Q(k + 1, λ)` (regularized upper incomplete gamma).
pub fn poisson_cdf<T: LazyReal>(k: T, rate: T) -> T {
    rate.gamma_ur(k + T::one())
}

/// Binomial CDF at `k < n`: `I_{1−p}(n − k, k + 1)`.
pub fn binomial_cdf_on_support<T: LazyReal>(successes: T, trials: T, p: T) -> T {
    (T::one() - p).beta_reg(trials - successes.clone(), successes + T::one())
}

/// Negative-binomial CDF at `k`: `I_p(r, k + 1)`.
pub fn negative_binomial_cdf<T: LazyReal>(failures: T, successes_until_stop: T, p: T) -> T {
    p.beta_reg(successes_until_stop, failures + T::one())
}

/// k-th order-statistic CDF of `sample_size` iid draws from a base CDF value
/// `u = F(x)`: the regularized incomplete beta `I_u(rank, sample_size − rank + 1)`.
/// `rank = 1` gives the minimum (`1 − (1−u)ⁿ`), `rank = n` the maximum (`uⁿ`).
pub fn order_statistic_cdf<T: LazyReal>(base_cdf: T, rank: T, sample_size: T) -> T {
    base_cdf.beta_reg(rank.clone(), sample_size - rank + T::one())
}

// --- stable log-CDF / log-survival tails ---
//
// Branch-free log-domain forms via the lazy `ln_1m_exp` / `ln_1p_exp`, accurate in
// the tails where `cdf().ln()` would lose precision.

/// Exponential log-CDF on `x ≥ 0`: `ln(1 − exp(−rate·x))` via `ln_1m_exp`.
pub fn exponential_ln_cdf_on_support<T: LazyReal>(x: T, rate: T) -> T {
    (-(rate * x)).ln_1m_exp()
}

/// Logistic log-CDF on the real line: `−softplus(−z)`, `z = (x − location)/scale`.
pub fn logistic_ln_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).ln_1p_exp()
}

/// Logistic log-survival on the real line: `−softplus(z)`, `z = (x − location)/scale`.
pub fn logistic_ln_sf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -z.ln_1p_exp()
}

/// Gumbel log-CDF on the real line: `−exp(−z)`, `z = (x − location)/scale`.
pub fn gumbel_ln_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).exp()
}

// --- closed-form quantiles (inverse CDFs) ---
//
// The inverse-CDF maps `p ∈ (0, 1)` to the support — a total formula on the open
// interval, so (as with `logit`) clamping `p` and the `p = 0`/`p = 1` boundary are
// the caller's branch. Variable powers `(·)^{1/shape}` go through
// `exp((1/shape)·ln(·))` because the lazy `powf` takes a *constant* exponent.

/// Exponential quantile: `−ln(1 − p) / rate`.
pub fn exponential_quantile<T: LazyReal>(p: T, rate: T) -> T {
    -((T::one() - p).ln()) / rate
}

/// Logistic quantile: `location + scale · ln(p / (1 − p))`.
pub fn logistic_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (p.clone() / (T::one() - p)).ln()
}

/// Cauchy quantile: `location + scale · tan(π·(p − ½))`.
pub fn cauchy_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (T::pi() * (p - T::half())).tan()
}

/// Laplace quantile: `location − scale · sign(q) · ln(1 − 2|q|)`, `q = p − ½`.
pub fn laplace_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    let q = p - T::half();
    location - scale * q.clone().signum() * (T::one() - T::two() * q.abs()).ln()
}

/// Uniform quantile: `lo + width · p`, with `width = hi − lo`.
pub fn uniform_quantile<T: LazyReal>(p: T, lo: T, width: T) -> T {
    lo + width * p
}

/// Weibull quantile: `scale · (−ln(1 − p))^{1/shape}`.
pub fn weibull_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -((T::one() - p).ln());
    scale * ((T::one() / shape) * base.ln()).exp()
}

/// Gumbel quantile: `location − scale · ln(−ln p)`.
pub fn gumbel_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location - scale * (-(p.ln())).ln()
}

/// Fréchet quantile: `scale · (−ln p)^{−1/shape}`.
pub fn frechet_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -(p.ln());
    scale * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Pareto quantile: `minimum · (1 − p)^{−1/shape}`.
pub fn pareto_quantile<T: LazyReal>(p: T, minimum: T, shape: T) -> T {
    let base = T::one() - p;
    minimum * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Lomax quantile: `scale · ((1 − p)^{−1/shape} − 1)`.
pub fn lomax_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = T::one() - p;
    scale * (((-(T::one() / shape)) * base.ln()).exp() - T::one())
}

/// Rayleigh quantile: `scale · √(−2·ln(1 − p))`.
pub fn rayleigh_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * (-T::two() * (T::one() - p).ln()).sqrt()
}

/// Normal quantile: `mean + std·√2·erf⁻¹(2p − 1)`.
pub fn normal_quantile<T: LazyReal>(p: T, mean: T, std_dev: T) -> T {
    mean + std_dev * T::sqrt_2() * (T::two() * p - T::one()).erf_inv()
}

/// Half-normal quantile: `scale·√2·erf⁻¹(p)`.
pub fn half_normal_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erf_inv()
}

/// Log-normal quantile: the Normal quantile on the log scale, exponentiated —
/// `exp(mean_log + σ_log·√2·erf⁻¹(2p − 1))`.
pub fn lognormal_quantile<T: LazyReal>(p: T, mean_log: T, std_log: T) -> T {
    normal_quantile(p, mean_log, std_log).exp()
}

// --- closed-form differential / Shannon entropies ---
//
// Pure algebra for most families; the gamma/beta-shaped ones reach for the lazy
// `ln_gamma`/`digamma` rather than a caller-supplied normalizer. All are branch-free
// and lower unchanged to a tensor backend.

/// Normal entropy `½·ln(2π·e·σ²)`.
pub fn normal_entropy<T: LazyReal>(std_dev: T) -> T {
    T::half() * (T::tau() * T::e() * std_dev.clone() * std_dev).ln()
}

/// Log-normal entropy `µ_log + ½·ln(2π·e·σ_log²)`.
pub fn lognormal_entropy<T: LazyReal>(mean_log: T, std_log: T) -> T {
    mean_log + T::half() * (T::tau() * T::e() * std_log.clone() * std_log).ln()
}

/// Exponential entropy `1 − ln(rate)`.
pub fn exponential_entropy<T: LazyReal>(rate: T) -> T {
    T::one() - rate.ln()
}

/// Laplace entropy `ln(2·e·scale)`.
pub fn laplace_entropy<T: LazyReal>(scale: T) -> T {
    (T::two() * T::e() * scale).ln()
}

/// Logistic entropy `ln(scale) + 2`.
pub fn logistic_entropy<T: LazyReal>(scale: T) -> T {
    scale.ln() + T::two()
}

/// Uniform entropy `ln(width)`.
pub fn uniform_entropy<T: LazyReal>(width: T) -> T {
    width.ln()
}

/// Discrete-uniform entropy `ln(count)`.
pub fn discrete_uniform_entropy<T: LazyReal>(count: T) -> T {
    count.ln()
}

/// Cauchy entropy `ln(4π·scale)`.
pub fn cauchy_entropy<T: LazyReal>(scale: T) -> T {
    (T::scalar(4.0) * T::pi() * scale).ln()
}

/// Gumbel entropy `ln(scale) + γ + 1`.
pub fn gumbel_entropy<T: LazyReal>(scale: T) -> T {
    scale.ln() + T::euler_gamma() + T::one()
}

/// Rayleigh entropy `1 + ln(scale/√2) + γ/2`.
pub fn rayleigh_entropy<T: LazyReal>(scale: T) -> T {
    T::one() + (scale / T::sqrt_2()).ln() + T::euler_gamma() * T::half()
}

/// Weibull entropy `γ·(1 − 1/k) + ln(scale/k) + 1`.
pub fn weibull_entropy<T: LazyReal>(scale: T, shape: T) -> T {
    T::euler_gamma() * (T::one() - T::one() / shape.clone()) + (scale / shape).ln() + T::one()
}

/// Pareto entropy `ln(minimum/shape) + 1/shape + 1`.
pub fn pareto_entropy<T: LazyReal>(minimum: T, shape: T) -> T {
    (minimum / shape.clone()).ln() + T::one() / shape + T::one()
}

/// Triangular entropy `½ + ln((high − low)/2)`.
pub fn triangular_entropy<T: LazyReal>(low: T, high: T) -> T {
    T::half() + ((high - low) / T::two()).ln()
}

/// Half-normal entropy `½·ln(π·scale²/2) + ½`.
pub fn half_normal_entropy<T: LazyReal>(scale: T) -> T {
    T::half() * (T::pi() * scale.clone() * scale / T::two()).ln() + T::half()
}

/// Gamma differential entropy:
/// `shape − ln(rate) + ln Γ(shape) + (1 − shape)·ψ(shape)`.
///
/// Computed directly from the lazy `ln_gamma`/`digamma`, with no caller-supplied
/// normalizer.
pub fn gamma_entropy<T: LazyReal>(shape: T, rate: T) -> T {
    let one = T::one();
    shape.clone() - rate.ln() + shape.clone().ln_gamma() + (one - shape.clone()) * shape.digamma()
}

/// Log multivariate gamma `ln Γ_p(a) = (p(p−1)/4)·ln π + Σ_{j=1}^p ln Γ(a + (1−j)/2)`.
///
/// The `p`-variate normalizer behind the Wishart / multivariate-Beta families. `p` is
/// the (structural) dimension; the loop is host-side, the per-term `ln_gamma` lazy.
pub fn ln_multivariate_gamma<T: LazyReal>(dimension: usize, a: T) -> T {
    let p = dimension as f64;
    let mut acc = T::scalar(p * (p - 1.0) / 4.0) * T::ln_pi();
    for j in 1..=dimension {
        acc = acc + (a.clone() + T::scalar((1.0 - j as f64) / 2.0)).ln_gamma();
    }
    acc
}

/// Log multivariate Beta `ln B(α) = Σ ln Γ(αᵢ) − ln Γ(Σ αᵢ)` — the Dirichlet
/// normalizer. `None` on empty input.
pub fn ln_multivariate_beta<T: LazyReal>(alpha: &[T]) -> Option<T> {
    if alpha.is_empty() {
        return None;
    }
    let total = sum_log_terms(alpha.iter().cloned())?;
    Some(sum_log_terms(alpha.iter().cloned().map(|a| a.ln_gamma()))? - total.ln_gamma())
}

/// Chi-squared entropy — the gamma entropy at `shape = freedom/2`, `rate = ½`.
pub fn chi_squared_entropy<T: LazyReal>(half_freedom: T) -> T {
    gamma_entropy(half_freedom, T::half())
}

/// Inverse-gamma entropy:
/// `shape + ln(scale) + ln Γ(shape) − (1 + shape)·ψ(shape)`.
pub fn inverse_gamma_entropy<T: LazyReal>(shape: T, scale: T) -> T {
    shape.clone() + scale.ln() + shape.clone().ln_gamma()
        - (T::one() + shape.clone()) * shape.digamma()
}

/// Beta entropy:
/// `ln B(α,β) − (α−1)·ψ(α) − (β−1)·ψ(β) + (α+β−2)·ψ(α+β)`.
pub fn beta_entropy<T: LazyReal>(alpha: T, beta: T) -> T {
    let one = T::one();
    let sum = alpha.clone() + beta.clone();
    let ln_beta = alpha.clone().ln_gamma() + beta.clone().ln_gamma() - sum.clone().ln_gamma();
    ln_beta
        - (alpha.clone() - one.clone()) * alpha.digamma()
        - (beta.clone() - one) * beta.digamma()
        + (sum.clone() - T::two()) * sum.digamma()
}

/// Student-t entropy (location/scale):
/// `ln(scale) + ½·ln(νπ) + ln Γ(ν/2) − ln Γ((ν+1)/2) + (ν+1)/2·(ψ((ν+1)/2) − ψ(ν/2))`.
pub fn students_t_entropy<T: LazyReal>(scale: T, freedom: T) -> T {
    let half_nu = freedom.clone() * T::half();
    let half_nu_plus_one = (freedom.clone() + T::one()) * T::half();
    scale.ln() + T::half() * (freedom * T::pi()).ln() + half_nu.clone().ln_gamma()
        - half_nu_plus_one.clone().ln_gamma()
        + half_nu_plus_one.clone() * (half_nu_plus_one.digamma() - half_nu.digamma())
}

/// von Mises entropy: `ln(2π) + ln I₀(κ) − κ·A(κ)`, `A = I₁/I₀` the mean resultant
/// length — over the scaled Bessel surface (`ln I₀ = ln I₀ᵉ + |κ|`, `A = I₁ᵉ/I₀ᵉ`).
pub fn von_mises_entropy<T: LazyReal>(kappa: T) -> T {
    let resultant = kappa.clone().bessel_i1e() / kappa.clone().bessel_i0e();
    T::ln_tau() + ln_i0(kappa.clone()) - kappa * resultant
}

/// von Mises-Fisher entropy: `−ln C_p(κ) − κ·R`. The caller supplies the log
/// normalizer `ln C_p(κ)` and the mean resultant length `R = I_{p/2}/I_{p/2−1}`
/// (both arbitrary-order Bessel, off the scalar lazy surface) — the same caller
/// contract as [`von_mises_fisher_log_pdf`].
pub fn von_mises_fisher_entropy<T: LazyReal>(
    log_normalizer: T,
    kappa: T,
    mean_resultant_length: T,
) -> T {
    -log_normalizer - kappa * mean_resultant_length
}

/// Watson entropy: `ln Z(κ) − κ·E[(µᵀx)²]`. The caller supplies the log partition
/// `ln Z(κ)` and the axial second moment `E[(µᵀx)²]` (a Kummer-function ratio off
/// the scalar lazy surface) — the same caller contract as [`watson_log_pdf`].
pub fn watson_entropy<T: LazyReal>(log_partition: T, kappa: T, axial_second_moment: T) -> T {
    log_partition - kappa * axial_second_moment
}

/// von Mises-Fisher cross-entropy `H(p, q) = −ln C_p(κ_q) − κ_q·R_p·(µ_p·µ_q)`. The
/// caller supplies `q`'s log normalizer, `p`'s mean resultant length `R_p`, and the
/// mean-direction cosine — the same caller contract as [`von_mises_fisher_entropy`].
/// KL is then `cross_entropy(p, q) − entropy(p)`.
pub fn von_mises_fisher_cross_entropy<T: LazyReal>(
    log_normalizer_other: T,
    kappa_other: T,
    mean_resultant_length_self: T,
    mean_cosine: T,
) -> T {
    -log_normalizer_other - kappa_other * mean_resultant_length_self * mean_cosine
}

/// Watson cross-entropy `H(p, q) = ln Z(κ_q) − κ_q·E_p[(µ_q·x)²]`. The caller supplies
/// `q`'s log partition and `p`'s second moment along `q`'s axis. KL is then
/// `cross_entropy(p, q) − entropy(p)`.
pub fn watson_cross_entropy<T: LazyReal>(
    log_partition_other: T,
    kappa_other: T,
    second_moment_self_along_other: T,
) -> T {
    log_partition_other - kappa_other * second_moment_self_along_other
}

/// Watson log unnormalized-density difference `κ·((µ·x)² − (µ·y)²)` from the two axis
/// cosines — the partition function cancels, so no normalizer is needed.
pub fn watson_ln_density_difference<T: LazyReal>(kappa: T, axis_dot_x: T, axis_dot_y: T) -> T {
    kappa * (square(axis_dot_x) - square(axis_dot_y))
}

// --- closed-form Kullback–Leibler divergences `D(self ‖ other)` ---
//
// Same-family closed forms in nats. The gamma/beta-shaped ones reach for the lazy
// `ln_gamma`/`digamma` rather than a caller-supplied normalizer, so no normalizer
// is threaded through. Forms whose divergence is `+∞` outside a support-containment
// region carry an `_on_support` suffix and the interior closed form only — the
// containment test is the caller's branch — which keeps them traceable.

/// Gaussian KL `D(N(m₀,s₀) ‖ N(m₁,s₁)) = ln(s₁/s₀) + (s₀² + (m₀−m₁)²)/(2 s₁²) − ½`.
///
/// Also the LogNormal KL on the log-space `(µ, σ)`, and the HalfNormal KL with both
/// means `0`.
pub fn kl_normal<T: LazyReal>(mean_self: T, std_self: T, mean_other: T, std_other: T) -> T {
    let dm = mean_self - mean_other;
    (std_other.clone() / std_self.clone()).ln()
        + (std_self.clone() * std_self + dm.clone() * dm)
            / (T::two() * std_other.clone() * std_other)
        - T::half()
}

/// Exponential KL: `ln(λ₀/λ₁) + λ₁/λ₀ − 1`.
pub fn kl_exponential<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    (rate_self.clone() / rate_other.clone()).ln() + rate_other / rate_self - T::one()
}

/// Poisson KL: `λ₀·ln(λ₀/λ₁) − λ₀ + λ₁`.
pub fn kl_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    rate_self.clone() * (rate_self.clone() / rate_other.clone()).ln() - rate_self + rate_other
}

/// Geometric KL: `ln(p₀/p₁) + ((1−p₀)/p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_geometric<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    (p_self.clone() / p_other).ln() + (q_self.clone() / p_self) * (q_self / q_other).ln()
}

/// Bernoulli KL on `0 < p < 1`: `p₀·ln(p₀/p₁) + (1−p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_bernoulli_on_unit<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    p_self.clone() * (p_self / p_other).ln() + q_self.clone() * (q_self / q_other).ln()
}

/// Negative-binomial KL for a matched number of successes `r`:
/// `r·ln(p₀/p₁) + r·((1−p₀)/p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_negative_binomial<T: LazyReal>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    successes.clone() * (p_self.clone() / p_other).ln()
        + successes * (q_self.clone() / p_self) * (q_self / q_other).ln()
}

/// Binomial KL for a matched trial count `n`: `n · D_KL(Bernoulli(p₀) ‖ Bernoulli(p₁))`
/// — the binomial coefficient cancels in the log-ratio. A differing trial count is
/// out of family and is the caller's branch.
pub fn kl_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    trials * kl_bernoulli_on_unit(p_self, p_other)
}

/// Cauchy KL: `ln[((γ₀+γ₁)² + (x₀−x₁)²) / (4 γ₀ γ₁)]`.
pub fn kl_cauchy<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let scale_sum = scale_self.clone() + scale_other.clone();
    let dm = loc_self - loc_other;
    ((scale_sum.clone() * scale_sum + dm.clone() * dm)
        / (T::scalar(4.0) * scale_self * scale_other))
        .ln()
}

/// Rayleigh KL: `2·ln(σ₁/σ₀) + (σ₀/σ₁)² − 1`.
pub fn kl_rayleigh<T: LazyReal>(scale_self: T, scale_other: T) -> T {
    let ratio = scale_self.clone() / scale_other.clone();
    T::two() * (scale_other / scale_self).ln() + ratio.clone() * ratio - T::one()
}

/// Laplace KL: `ln(b₁/b₀) + (b₀·exp(−d/b₀) + d)/b₁ − 1`, `d = |m₀ − m₁|`.
pub fn kl_laplace<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let d = (loc_self - loc_other).abs();
    (scale_other.clone() / scale_self.clone()).ln()
        + (scale_self.clone() * (-(d.clone() / scale_self)).exp() + d) / scale_other
        - T::one()
}

/// Wald / inverse-Gaussian KL (mean µ, shape λ):
/// `½ln(λ₀/λ₁) + λ₁(µ₀−2µ₁)/(2µ₁²) + ½λ₁(1/µ₀ + 1/λ₀) − ½`.
pub fn kl_wald<T: LazyReal>(mean_self: T, shape_self: T, mean_other: T, shape_other: T) -> T {
    let one = T::one();
    T::half() * (shape_self.clone() / shape_other.clone()).ln()
        + shape_other.clone() * (mean_self.clone() - T::two() * mean_other.clone())
            / (T::two() * mean_other.clone() * mean_other)
        + T::half() * shape_other * (one.clone() / mean_self + one / shape_self)
        - T::half()
}

/// Gamma KL: `(a₀−a₁)·ψ(a₀) − ln Γ(a₀) + ln Γ(a₁) + a₁·ln(b₀/b₁) + a₀·(b₁−b₀)/b₀`.
///
/// Also the ChiSquared KL (shape `k/2`, rate `½`) and the InverseGamma KL (shape,
/// rate ≡ scale), both of which reduce to this form.
pub fn kl_gamma<T: LazyReal>(shape_self: T, rate_self: T, shape_other: T, rate_other: T) -> T {
    (shape_self.clone() - shape_other.clone()) * shape_self.clone().digamma()
        - shape_self.clone().ln_gamma()
        + shape_other.clone().ln_gamma()
        + shape_other * (rate_self.clone().ln() - rate_other.clone().ln())
        + shape_self * (rate_other - rate_self.clone()) / rate_self
}

/// Beta KL:
/// `ln B(a₁,b₁) − ln B(a₀,b₀) + (a₀−a₁)ψ(a₀) + (b₀−b₁)ψ(b₀) + (a₁−a₀+b₁−b₀)ψ(a₀+b₀)`.
pub fn kl_beta<T: LazyReal>(alpha_self: T, beta_self: T, alpha_other: T, beta_other: T) -> T {
    let sum_self = alpha_self.clone() + beta_self.clone();
    let ln_beta_self =
        alpha_self.clone().ln_gamma() + beta_self.clone().ln_gamma() - sum_self.clone().ln_gamma();
    let ln_beta_other = alpha_other.clone().ln_gamma() + beta_other.clone().ln_gamma()
        - (alpha_other.clone() + beta_other.clone()).ln_gamma();
    let psi_alpha = alpha_self.clone().digamma();
    let psi_beta = beta_self.clone().digamma();
    let psi_sum = sum_self.digamma();
    ln_beta_other - ln_beta_self
        + (alpha_self.clone() - alpha_other.clone()) * psi_alpha
        + (beta_self.clone() - beta_other.clone()) * psi_beta
        + (alpha_other - alpha_self + beta_other - beta_self) * psi_sum
}

/// Weibull KL (Bauckhage's closed form, shape `k`, scale `λ`):
/// `(ln k₀ − k₀ln λ₀) − (ln k₁ − k₁ln λ₁) + (k₀−k₁)(ln λ₀ − γ/k₀)
///  + (λ₀/λ₁)^{k₁}·Γ(1 + k₁/k₀) − 1`.
///
/// Also the Frechet KL through the `x ↦ 1/x` bijection (scales inverted).
pub fn kl_weibull<T: LazyReal>(shape_self: T, scale_self: T, shape_other: T, scale_other: T) -> T {
    let (k0, l0, k1, l1) = (shape_self, scale_self, shape_other, scale_other);
    let term0 = k0.clone().ln() - k0.clone() * l0.clone().ln();
    let term1 = k1.clone().ln() - k1.clone() * l1.clone().ln();
    let cross = (k0.clone() - k1.clone()) * (l0.clone().ln() - T::euler_gamma() / k0.clone());
    let ratio_pow = (k1.clone() * (l0 / l1).ln()).exp();
    let gamma_term = (T::one() + k1 / k0).ln_gamma().exp();
    term0 - term1 + cross + ratio_pow * gamma_term - T::one()
}

/// Gumbel KL (location µ, scale β):
/// `ln(β₁/β₀) + γ(β₀/β₁ − 1) + (µ₀−µ₁)/β₁ + exp((µ₁−µ₀)/β₁)·Γ(1 + β₀/β₁) − 1`.
pub fn kl_gumbel<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let (mu0, b0, mu1, b1) = (loc_self, scale_self, loc_other, scale_other);
    (b1.clone() / b0.clone()).ln()
        + T::euler_gamma() * (b0.clone() / b1.clone() - T::one())
        + (mu0.clone() - mu1.clone()) / b1.clone()
        + ((mu1 - mu0) / b1.clone()).exp() * (T::one() + b0 / b1).ln_gamma().exp()
        - T::one()
}

/// Pareto KL on nested support (`xₘ₀ ≥ xₘ₁`):
/// `ln(α₀/α₁) + α₁·ln(xₘ₀/xₘ₁) + α₁/α₀ − 1`.
pub fn kl_pareto_on_support<T: LazyReal>(
    minimum_self: T,
    shape_self: T,
    minimum_other: T,
    shape_other: T,
) -> T {
    let (xm0, a0, xm1, a1) = (minimum_self, shape_self, minimum_other, shape_other);
    (a0.clone() / a1.clone()).ln() + a1.clone() * (xm0 / xm1).ln() + a1 / a0 - T::one()
}

/// `ln I₀(x)` over the lazy Bessel surface: `ln I₀ᵉ(x) + |x|`, overflow-free.
fn ln_i0<T: LazyReal>(x: T) -> T {
    x.clone().bessel_i0e().ln() + x.abs()
}

/// von Mises KL: `ln I₀(κ₁) − ln I₀(κ₀) + (I₁/I₀)(κ₀)·(κ₀ − κ₁·cos Δµ)`.
///
/// Reachable now that `cos` and the scaled Bessel `I₀ᵉ`/`I₁ᵉ` are on the lazy
/// surface: `ln I₀ = ln I₀ᵉ + |·|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`.
pub fn kl_von_mises<T: LazyReal>(mean_self: T, kappa_self: T, mean_other: T, kappa_other: T) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let resultant = k0.clone().bessel_i1e() / k0.clone().bessel_i0e();
    ln_i0(k1.clone()) - ln_i0(k0.clone()) + resultant * (k0 - k1 * (mu0 - mu1).cos())
}

/// Uniform KL on nested support (`[a₀,b₀] ⊆ [a₁,b₁]`): `ln(w₁/w₀)`, `w = b − a`. The
/// containment test (`+∞` otherwise) is the caller's branch.
pub fn kl_uniform_on_support<T: LazyReal>(width_self: T, width_other: T) -> T {
    (width_other / width_self).ln()
}

/// Discrete-uniform KL on nested support: `ln(n₁/n₀)`, `n` the support count.
pub fn kl_discrete_uniform_on_support<T: LazyReal>(count_self: T, count_other: T) -> T {
    (count_other / count_self).ln()
}

// --- closed-form squared Hellinger distances `H² = 1 − BC` ---
//
// Each is built from the family Bhattacharyya coefficient `BC = ∫√(pq)`; the
// `(1 − BC).max(0)` only absorbs `O(ε)` rounding (`BC ≤ 1` is a theorem).

/// Gaussian Bhattacharyya coefficient `√(2 s₀ s₁/v)·exp(−¼(m₀−m₁)²/v)`, `v=s₀²+s₁²`.
fn gaussian_bc<T: LazyReal>(m0: T, s0: T, m1: T, s1: T) -> T {
    let v = s0.clone() * s0.clone() + s1.clone() * s1.clone();
    let dm = m0 - m1;
    (T::two() * s0 * s1 / v.clone()).sqrt() * (-T::scalar(0.25) * dm.clone() * dm / v).exp()
}

/// Gamma Bhattacharyya coefficient (shape `a`, rate `b`), via `ln_gamma`.
fn gamma_bc<T: LazyReal>(a0: T, b0: T, a1: T, b1: T) -> T {
    let a = T::half() * (a0.clone() + a1.clone());
    let b = T::half() * (b0.clone() + b1.clone());
    let ln_bc = a.clone().ln_gamma() + T::half() * (a0.clone() * b0.ln() + a1.clone() * b1.ln())
        - T::half() * (a0.ln_gamma() + a1.ln_gamma())
        - a * b.ln();
    ln_bc.exp()
}

/// `H² = (1 − BC).max(0)`.
fn hellinger_sq_from_bc<T: LazyReal>(bc: T) -> T {
    (T::one() - bc).max(T::zero())
}

/// Gaussian squared Hellinger (Normal, LogNormal on log-params, HalfNormal at m=0).
pub fn hellinger_sq_normal<T: LazyReal>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
) -> T {
    hellinger_sq_from_bc(gaussian_bc(mean_self, std_self, mean_other, std_other))
}

/// Exponential squared Hellinger: `BC = 2√(λ₀λ₁)/(λ₀+λ₁)`.
pub fn hellinger_sq_exponential<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    hellinger_sq_from_bc(T::two() * (l0.clone() * l1.clone()).sqrt() / (l0 + l1))
}

/// Poisson squared Hellinger: `BC = exp(−½(√λ₀ − √λ₁)²)`.
pub fn hellinger_sq_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let d = rate_self.sqrt() - rate_other.sqrt();
    hellinger_sq_from_bc((-T::half() * d.clone() * d).exp())
}

/// Bernoulli squared Hellinger: `BC = √(p₀p₁) + √((1−p₀)(1−p₁))`.
pub fn hellinger_sq_bernoulli<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    hellinger_sq_from_bc(bc)
}

/// Gamma squared Hellinger (Gamma, ChiSquared, InverseGamma).
pub fn hellinger_sq_gamma<T: LazyReal>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    hellinger_sq_from_bc(gamma_bc(shape_self, rate_self, shape_other, rate_other))
}

/// Beta squared Hellinger:
/// `BC = exp(ln B(ā, b̄) − ½(ln B(a₀,b₀) + ln B(a₁,b₁)))`, `ā=(a₀+a₁)/2`.
pub fn hellinger_sq_beta<T: LazyReal>(
    alpha_self: T,
    beta_self: T,
    alpha_other: T,
    beta_other: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    let am = T::half() * (alpha_self.clone() + alpha_other.clone());
    let bm = T::half() * (beta_self.clone() + beta_other.clone());
    let ln_bc = ln_beta(am, bm)
        - T::half() * (ln_beta(alpha_self, beta_self) + ln_beta(alpha_other, beta_other));
    hellinger_sq_from_bc(ln_bc.exp())
}

/// Wald / inverse-Gaussian squared Hellinger (mean µ, shape λ).
pub fn hellinger_sq_wald<T: LazyReal>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let s = ((l0.clone() / (m0.clone() * m0.clone()) + l1.clone() / (m1.clone() * m1.clone()))
        * (l0.clone() + l1.clone()))
    .sqrt();
    let ln_bc = T::scalar(0.25) * (l0.clone() * l1.clone()).ln()
        + T::half() * (T::two() / (l0.clone() + l1.clone())).ln()
        + T::half() * (l0 / m0 + l1 / m1)
        - T::half() * s;
    hellinger_sq_from_bc(ln_bc.exp())
}

/// von Mises squared Hellinger: `BC = I₀(R)/√(I₀(κ₀)I₀(κ₁))`,
/// `R = ½√(κ₀² + κ₁² + 2κ₀κ₁ cos Δµ)`.
pub fn hellinger_sq_von_mises<T: LazyReal>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let r = T::half()
        * (k0.clone() * k0.clone()
            + k1.clone() * k1.clone()
            + T::two() * k0.clone() * k1.clone() * cos)
            .sqrt();
    let ln_bc = ln_i0(r) - T::half() * (ln_i0(k0) + ln_i0(k1));
    hellinger_sq_from_bc(ln_bc.exp())
}

/// Binomial squared Hellinger for a matched trial count `n`: the Bhattacharyya
/// coefficient factorizes over the `n` iid Bernoulli trials, `BC = BC_Bernoulli^n`,
/// so `H² = 1 − BC`. A differing trial count is the caller's out-of-family branch.
pub fn hellinger_sq_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc_bernoulli = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    hellinger_sq_from_bc((trials * bc_bernoulli.ln()).exp())
}

/// Negative-binomial squared Hellinger for a matched number of successes `r`:
/// `BC = [√(p₀p₁) / (1 − √((1−p₀)(1−p₁)))]^r`, so `H² = 1 − BC`.
pub fn hellinger_sq_negative_binomial<T: LazyReal>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        / (one.clone() - ((one.clone() - p_self) * (one - p_other)).sqrt());
    hellinger_sq_from_bc((successes * bc.ln()).exp())
}

/// Uniform squared Hellinger: `BC = overlap / √(w₀·w₁)`, the overlap
/// `max(min(b₀,b₁) − max(a₀,a₁), 0)` and widths `w = b − a`. Bounded, so valid for
/// any two uniforms (the clamp absorbs disjoint support).
pub fn hellinger_sq_uniform<T: LazyReal>(lo_self: T, hi_self: T, lo_other: T, hi_other: T) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    hellinger_sq_from_bc(overlap / (width_self * width_other).sqrt())
}

/// Discrete-uniform squared Hellinger: `BC = overlap / √(n₀·n₁)`, integer overlap
/// `max(min(b₀,b₁) − max(a₀,a₁) + 1, 0)` and counts `n = b − a + 1`.
pub fn hellinger_sq_discrete_uniform<T: LazyReal>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    hellinger_sq_from_bc(overlap / (n_self * n_other).sqrt())
}

// --- closed-form χ² (Pearson) divergences ---
//
// `_on_support` forms carry the interior only; the `2λ₀>λ₁`-style convergence test
// (`+∞` otherwise) is the caller's branch.

/// Bernoulli χ²: `(p₀−p₁)² / (p₁(1−p₁))`.
pub fn chi_squared_bernoulli<T: LazyReal>(p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    diff.clone() * diff / (p_other.clone() * (T::one() - p_other))
}

/// Poisson χ²: `exp((λ₀−λ₁)²/λ₁) − 1`.
pub fn chi_squared_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let diff = rate_self - rate_other.clone();
    (diff.clone() * diff / rate_other).exp() - T::one()
}

/// Gaussian χ² interior (`2s₁² > s₀²`): `s₁²/(s₀·√d)·exp((m₀−m₁)²/d) − 1`, `d=2s₁²−s₀²`.
pub fn chi_squared_normal_on_support<T: LazyReal>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
) -> T {
    let (m0, s0, m1, s1) = (mean_self, std_self, mean_other, std_other);
    let denom = T::two() * s1.clone() * s1.clone() - s0.clone() * s0.clone();
    let dm = m0 - m1;
    s1.clone() * s1 / (s0 * denom.clone().sqrt()) * (dm.clone() * dm / denom).exp() - T::one()
}

/// Exponential χ² interior (`2λ₀ > λ₁`): `λ₀²/(λ₁(2λ₀−λ₁)) − 1`.
pub fn chi_squared_exponential_on_support<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    l0.clone() * l0.clone() / (l1.clone() * (T::two() * l0 - l1)) - T::one()
}

/// Gamma χ² interior (`2a₀>a₁`, `2b₀>b₁`), via `ln_gamma`.
pub fn chi_squared_gamma_on_support<T: LazyReal>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let a = T::two() * a0.clone() - a1.clone();
    let b = T::two() * b0.clone() - b1.clone();
    let ln_term = T::two() * a0.clone() * b0.clone().ln() - T::two() * a0.ln_gamma()
        + a1.clone().ln_gamma()
        - a1 * b1.ln()
        + a.clone().ln_gamma()
        - a * b.ln();
    ln_term.exp() - T::one()
}

/// von Mises χ² (always finite): `I₀(R)·I₀(κ₁)/I₀(κ₀)² − 1`,
/// `R = √(4κ₀² + κ₁² − 4κ₀κ₁ cos Δµ)`.
pub fn chi_squared_von_mises<T: LazyReal>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let r = (T::scalar(4.0) * k0.clone() * k0.clone() + k1.clone() * k1.clone()
        - T::scalar(4.0) * k0.clone() * k1.clone() * cos)
        .sqrt();
    (ln_i0(r) + ln_i0(k1) - T::two() * ln_i0(k0)).exp() - T::one()
}

/// Binomial χ² for a matched trial count `n`: `(1 + χ²_Bernoulli)^n − 1`, always
/// finite. A differing trial count is the caller's out-of-family branch.
pub fn chi_squared_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    let chi_bernoulli = diff.clone() * diff / (p_other.clone() * (T::one() - p_other));
    (trials * (T::one() + chi_bernoulli).ln()).exp() - T::one()
}

/// Negative-binomial χ² interior for a matched number of successes `r`
/// (finite when `(1−p₀)² < 1−p₁`): `exp(r·(2 ln p₀ − ln p₁ − ln(1 − z))) − 1`,
/// `z = (1−p₀)²/(1−p₁)`. The convergence test is the caller's branch.
pub fn chi_squared_negative_binomial_on_support<T: LazyReal>(
    successes: T,
    p_self: T,
    p_other: T,
) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let z = q_self.clone() * q_self / (one.clone() - p_other.clone());
    (successes * (T::two() * p_self.ln() - p_other.ln() - (one - z).ln())).exp() - T::one()
}

/// Wald / inverse-Gaussian χ² interior (mean µ, shape λ; finite when `2λ₀ > λ₁` and
/// `2λ₀/µ₀² > λ₁/µ₁²`): `exp(ln λ₀ − ½ln λ₁ + 2λ₀/µ₀ − λ₁/µ₁ − ½ln la − √(aa·la)) − 1`,
/// `la = 2λ₀ − λ₁`, `aa = 2λ₀/µ₀² − λ₁/µ₁²`. The convergence test is the caller's branch.
pub fn chi_squared_wald_on_support<T: LazyReal>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let la = T::two() * l0.clone() - l1.clone();
    let aa =
        T::two() * l0.clone() / (m0.clone() * m0.clone()) - l1.clone() / (m1.clone() * m1.clone());
    let ln_int = l0.clone().ln() - T::half() * l1.clone().ln() + T::two() * l0 / m0
        - l1 / m1
        - T::half() * la.clone().ln()
        - (aa * la).sqrt();
    ln_int.exp() - T::one()
}

/// Uniform χ² on nested support: `w₁/w₀ − 1`, `w = b − a`. The containment test is
/// the caller's branch.
pub fn chi_squared_uniform_on_support<T: LazyReal>(width_self: T, width_other: T) -> T {
    width_other / width_self - T::one()
}

/// Discrete-uniform χ² on nested support: `n₁/n₀ − 1`, `n` the support count.
pub fn chi_squared_discrete_uniform_on_support<T: LazyReal>(count_self: T, count_other: T) -> T {
    count_other / count_self - T::one()
}

// --- closed-form Rényi divergences `D_α` (α ≠ 1) ---

/// Bernoulli Rényi: `ln(p₀^α p₁^{1−α} + (1−p₀)^α (1−p₁)^{1−α}) / (α−1)`.
pub fn renyi_bernoulli<T: LazyReal>(p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    s.ln() / (alpha - T::one())
}

/// Gaussian Rényi interior (`σ_α² = α s₁² + (1−α)s₀² > 0`):
/// `α(m₀−m₁)²/(2σ_α²) + ((1−α)ln s₀ + α ln s₁ − ½ln σ_α²)/(α−1)`.
pub fn renyi_normal_on_support<T: LazyReal>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
    alpha: T,
) -> T {
    let (m0, s0, m1, s1) = (mean_self, std_self, mean_other, std_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let var_alpha =
        alpha.clone() * s1.clone() * s1.clone() + one_minus_alpha.clone() * s0.clone() * s0.clone();
    let dm = m0 - m1;
    let mean_term = alpha.clone() * dm.clone() * dm / (T::two() * var_alpha.clone());
    let log_term = (one_minus_alpha * s0.ln() + alpha.clone() * s1.ln()
        - T::half() * var_alpha.ln())
        / (alpha - T::one());
    mean_term + log_term
}

/// Gamma Rényi interior (`a_α, b_α > 0`), via `ln_gamma`.
pub fn renyi_gamma_on_support<T: LazyReal>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
    alpha: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let a_alpha = alpha.clone() * a0.clone() + one_minus_alpha.clone() * a1.clone();
    let b_alpha = alpha.clone() * b0.clone() + one_minus_alpha.clone() * b1.clone();
    let num = alpha.clone() * (a0.clone() * b0.ln() - a0.ln_gamma())
        + one_minus_alpha * (a1.clone() * b1.ln() - a1.ln_gamma())
        + a_alpha.clone().ln_gamma()
        - a_alpha * b_alpha.ln();
    num / (alpha - T::one())
}

/// von Mises Rényi interior:
/// `(ln I₀(R_α) − α ln I₀(κ₀) − (1−α) ln I₀(κ₁)) / (α−1)`,
/// `R_α = √((ακ₀)² + ((1−α)κ₁)² + 2α(1−α)κ₀κ₁ cos Δµ)`.
pub fn renyi_von_mises<T: LazyReal>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
    alpha: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let one_minus_alpha = T::one() - alpha.clone();
    let term0 = alpha.clone() * k0.clone();
    let term1 = one_minus_alpha.clone() * k1.clone();
    let r_alpha = (term0.clone() * term0
        + term1.clone() * term1
        + T::two() * alpha.clone() * one_minus_alpha.clone() * k0.clone() * k1.clone() * cos)
        .sqrt();
    (ln_i0(r_alpha) - alpha.clone() * ln_i0(k0) - one_minus_alpha * ln_i0(k1)) / (alpha - T::one())
}

/// Binomial Rényi for a matched trial count `n`: `n ·` the per-trial Bernoulli
/// Rényi, `n · ln(p₀^α p₁^{1−α} + (1−p₀)^α (1−p₁)^{1−α}) / (α−1)`. A differing trial
/// count is the caller's out-of-family branch.
pub fn renyi_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    trials * s.ln() / (alpha - T::one())
}

/// Negative-binomial Rényi interior for matched successes `r`
/// (finite when `w = (1−p₀)^α (1−p₁)^{1−α} < 1`):
/// `r · (α ln p₀ + (1−α) ln p₁ − ln(1 − w)) / (α−1)`. The convergence test is the
/// caller's branch.
pub fn renyi_negative_binomial_on_support<T: LazyReal>(
    successes: T,
    p_self: T,
    p_other: T,
    alpha: T,
) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let w = (alpha.clone() * (one.clone() - p_self.clone()).ln()).exp()
        * (one_minus_alpha.clone() * (one.clone() - p_other.clone()).ln()).exp();
    successes * (alpha.clone() * p_self.ln() + one_minus_alpha * p_other.ln() - (one - w).ln())
        / (alpha - T::one())
}

/// Wald / inverse-Gaussian Rényi interior (mean µ, shape λ; finite when
/// `la = α λ₀ + (1−α) λ₁ > 0` and `aa = α λ₀/µ₀² + (1−α) λ₁/µ₁² > 0`):
/// `[½(α ln λ₀ + (1−α) ln λ₁) + α λ₀/µ₀ + (1−α) λ₁/µ₁ − ½ ln la − √(aa·la)] / (α−1)`.
/// The convergence test is the caller's branch.
pub fn renyi_wald_on_support<T: LazyReal>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
    alpha: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let la = alpha.clone() * l0.clone() + one_minus_alpha.clone() * l1.clone();
    let aa = alpha.clone() * l0.clone() / (m0.clone() * m0.clone())
        + one_minus_alpha.clone() * l1.clone() / (m1.clone() * m1.clone());
    let ln_int = T::half()
        * (alpha.clone() * l0.clone().ln() + one_minus_alpha.clone() * l1.clone().ln())
        + alpha.clone() * l0 / m0
        + one_minus_alpha * l1 / m1
        - T::half() * la.clone().ln()
        - (aa * la).sqrt();
    ln_int / (alpha - T::one())
}

/// Uniform Rényi interior: `ln(overlap · w₀^{−α} · w₁^{−(1−α)}) / (α−1)`, `w = b − a`.
/// The `+∞` boundary (`α>1` outside containment, or disjoint support) is the caller's
/// branch.
pub fn renyi_uniform_on_support<T: LazyReal>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    let one_minus_alpha = T::one() - alpha.clone();
    let s = overlap
        * (-alpha.clone() * width_self.ln()).exp()
        * (-one_minus_alpha * width_other.ln()).exp();
    s.ln() / (alpha - T::one())
}

/// Discrete-uniform Rényi interior: `ln(overlap · n₀^{−α} · n₁^{−(1−α)}) / (α−1)`,
/// integer overlap and `n = b − a + 1`. The `+∞` boundary is the caller's branch.
pub fn renyi_discrete_uniform_on_support<T: LazyReal>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    let one_minus_alpha = T::one() - alpha.clone();
    let s =
        overlap * (-alpha.clone() * n_self.ln()).exp() * (-one_minus_alpha * n_other.ln()).exp();
    s.ln() / (alpha - T::one())
}

// --- closed-form total variation ---

/// Bernoulli total variation: `|p₀ − p₁|`.
pub fn total_variation_bernoulli<T: LazyReal>(p_self: T, p_other: T) -> T {
    (p_self - p_other).abs()
}

// --- vector / reduction kernels over probability simplices ---
//
// Reductions over equal-length, on-support (strictly positive) probability vectors:
// the per-element guard against `0·log 0` is the caller's responsibility, so each is
// a branch-free fold that lowers to a tensor `Reduce`. They return `None` on length
// mismatch (or an empty input), matching the simplex/log-term kernels above.

/// Stable log-sum-exp `m + ln Σ exp(xᵢ − m)`, `m = maxᵢ xᵢ` — the mixture /
/// categorical normalizer. `None` for an empty slice.
pub fn log_sum_exp<T: LazyReal>(values: &[T]) -> Option<T> {
    let max = values.iter().cloned().reduce(|a, b| a.max(b))?;
    let sum = sum_log_terms(values.iter().cloned().map(|x| (x - max.clone()).exp()))?;
    Some(max + sum.ln())
}

/// Log-mean-exp `log_sum_exp(values) − ln(n)`.
pub fn log_mean_exp<T: LazyReal>(values: &[T]) -> Option<T> {
    let count = values.len();
    Some(log_sum_exp(values)? - T::scalar(count as f64).ln())
}

/// Log-softmax `xᵢ − log_sum_exp(x)`, a log-probability vector. `None` if empty.
pub fn log_softmax<T: LazyReal>(values: &[T]) -> Option<Vec<T>> {
    let log_z = log_sum_exp(values)?;
    Some(values.iter().cloned().map(|x| x - log_z.clone()).collect())
}

/// Softmax `exp(xᵢ − log_sum_exp(x))`, a probability vector. `None` if empty.
pub fn softmax<T: LazyReal>(values: &[T]) -> Option<Vec<T>> {
    Some(log_softmax(values)?.into_iter().map(|x| x.exp()).collect())
}

/// Shannon entropy `−Σ pᵢ·ln pᵢ` of an on-support probability vector.
pub fn shannon_entropy<T: LazyReal>(p: &[T]) -> Option<T> {
    Some(-sum_log_terms(p.iter().map(|pi| pi.clone() * pi.ln()))?)
}

/// Cross-entropy `−Σ pᵢ·ln qᵢ` on the shared support.
pub fn cross_entropy<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    Some(-sum_log_terms(
        p.iter()
            .cloned()
            .zip(q.iter().cloned())
            .map(|(pi, qi)| pi * qi.ln()),
    )?)
}

/// Categorical KL `Σ pᵢ·ln(pᵢ/qᵢ)` on the shared support.
pub fn categorical_kl<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    sum_log_terms(
        p.iter()
            .cloned()
            .zip(q.iter().cloned())
            .map(|(pi, qi)| pi.clone() * (pi / qi).ln()),
    )
}

/// Bhattacharyya coefficient `Σ √(pᵢ qᵢ)`.
pub fn bhattacharyya_coefficient<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    sum_log_terms(
        p.iter()
            .cloned()
            .zip(q.iter().cloned())
            .map(|(pi, qi)| (pi * qi).sqrt()),
    )
}

/// Categorical squared Hellinger `1 − Σ √(pᵢ qᵢ)`.
pub fn hellinger_sq_categorical<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    Some(hellinger_sq_from_bc(bhattacharyya_coefficient(p, q)?))
}

/// Categorical χ² `Σ (pᵢ − qᵢ)²/qᵢ` on the shared support.
pub fn categorical_chi_squared<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    sum_log_terms(p.iter().cloned().zip(q.iter().cloned()).map(|(pi, qi)| {
        let diff = pi - qi.clone();
        diff.clone() * diff / qi
    }))
}

/// Categorical total variation `½ Σ |pᵢ − qᵢ|`.
pub fn categorical_total_variation<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    Some(
        T::half()
            * sum_log_terms(
                p.iter()
                    .cloned()
                    .zip(q.iter().cloned())
                    .map(|(pi, qi)| (pi - qi).abs()),
            )?,
    )
}

/// Categorical Rényi `ln(Σ pᵢ^α qᵢ^{1−α}) / (α−1)` on the shared support.
pub fn categorical_renyi<T: LazyReal>(p: &[T], q: &[T], alpha: T) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    let one_minus_alpha = T::one() - alpha.clone();
    let s = sum_log_terms(p.iter().cloned().zip(q.iter().cloned()).map(|(pi, qi)| {
        (alpha.clone() * pi.ln()).exp() * (one_minus_alpha.clone() * qi.ln()).exp()
    }))?;
    Some(s.ln() / (alpha - T::one()))
}

/// Jensen–Shannon divergence `½ Σ [pᵢ ln(pᵢ/mᵢ) + qᵢ ln(qᵢ/mᵢ)]`, `mᵢ = (pᵢ+qᵢ)/2`,
/// on the shared support.
pub fn jensen_shannon<T: LazyReal>(p: &[T], q: &[T]) -> Option<T> {
    if p.len() != q.len() {
        return None;
    }
    let terms = p.iter().cloned().zip(q.iter().cloned()).map(|(pi, qi)| {
        let m = T::half() * (pi.clone() + qi.clone());
        pi.clone() * (pi / m.clone()).ln() + qi.clone() * (qi / m).ln()
    });
    Some(T::half() * sum_log_terms(terms)?)
}

/// Dirichlet KL `D(α ‖ β)` on the simplex:
/// `ln Γ(α₀) − ln Γ(β₀) + Σ[ln Γ(βᵢ) − ln Γ(αᵢ) + (αᵢ−βᵢ)(ψ(αᵢ) − ψ(α₀))]`.
pub fn dirichlet_kl<T: LazyReal>(alpha: &[T], beta: &[T]) -> Option<T> {
    if alpha.len() != beta.len() || alpha.is_empty() {
        return None;
    }
    let a0 = sum_log_terms(alpha.iter().cloned())?;
    let b0 = sum_log_terms(beta.iter().cloned())?;
    let psi_a0 = a0.clone().digamma();
    let head = a0.ln_gamma() - b0.ln_gamma();
    let terms = alpha
        .iter()
        .cloned()
        .zip(beta.iter().cloned())
        .map(|(ai, bi)| {
            bi.clone().ln_gamma() - ai.clone().ln_gamma()
                + (ai.clone() - bi) * (ai.digamma() - psi_a0.clone())
        });
    Some(head + sum_log_terms(terms)?)
}

/// Dirichlet entropy:
/// `ln B(α) + (α₀ − K)·ψ(α₀) − Σ(αᵢ − 1)·ψ(αᵢ)`, `ln B(α) = Σ ln Γ(αᵢ) − ln Γ(α₀)`.
pub fn dirichlet_entropy<T: LazyReal>(alpha: &[T]) -> Option<T> {
    if alpha.is_empty() {
        return None;
    }
    let a0 = sum_log_terms(alpha.iter().cloned())?;
    let dimension = T::scalar(alpha.len() as f64);
    let ln_b = sum_log_terms(alpha.iter().cloned().map(|a| a.ln_gamma()))? - a0.clone().ln_gamma();
    let sum_terms = sum_log_terms(
        alpha
            .iter()
            .cloned()
            .map(|a| (a.clone() - T::one()) * a.digamma()),
    )?;
    Some(ln_b + (a0.clone() - dimension) * a0.digamma() - sum_terms)
}

// --- proper scoring rules for a categorical forecast ---
//
// `observed` is the realized class index — a structural label, not a lazy value —
// so the truth indicator is a host-side branch, leaving the per-element algebra
// branch-free. `None` when the index is out of range.

/// Brier score `Σ (pᵢ − 𝟙[i = observed])²` — the categorical mean squared error.
pub fn brier_score<T: LazyReal>(p: &[T], observed: usize) -> Option<T> {
    if observed >= p.len() {
        return None;
    }
    sum_log_terms(p.iter().enumerate().map(|(i, pi)| {
        let truth = if i == observed { T::one() } else { T::zero() };
        let residual = pi.clone() - truth;
        residual.clone() * residual
    }))
}

/// Spherical score `p_observed / ‖p‖₂`.
pub fn spherical_score<T: LazyReal>(p: &[T], observed: usize) -> Option<T> {
    if observed >= p.len() {
        return None;
    }
    let norm = sum_log_terms(p.iter().map(|pi| square(pi.clone())))?.sqrt();
    Some(p[observed].clone() / norm)
}

// --- reparameterization transforms (constrained ↔ unconstrained ℝ) ---
//
// The bijections an inference layer differentiates through: a constrained value is
// mapped to ℝ, worked on there, and mapped back, with the density corrected by the
// transform's log-Jacobian. Each is pure analysis on the interior (boundary `±∞`
// is the caller's branch), so the whole reparameterization traces to a backend.

/// Positive support `(0, ∞)` → ℝ: `ln(x)`.
pub fn positive_unconstrain<T: LazyReal>(x: T) -> T {
    x.ln()
}

/// ℝ → positive support: `exp(y)`.
pub fn positive_constrain<T: LazyReal>(y: T) -> T {
    y.exp()
}

/// `ln|d exp(y)/dy| = y`, the positive-support Jacobian correction.
pub fn positive_constrain_log_jacobian<T: LazyReal>(y: T) -> T {
    y
}

/// Unit interval `(0, 1)` → ℝ: the logit `ln(x) − ln(1 − x)`.
pub fn unit_interval_unconstrain<T: LazyReal>(x: T) -> T {
    logit(x)
}

/// ℝ → unit interval: the sigmoid `1/(1 + exp(−y))`.
pub fn unit_interval_constrain<T: LazyReal>(y: T) -> T {
    sigmoid(y)
}

/// `ln|d σ(y)/dy| = log σ(y) + log(1 − σ(y)) = −softplus(−y) − softplus(y)`.
pub fn unit_interval_constrain_log_jacobian<T: LazyReal>(y: T) -> T {
    -(-y.clone()).ln_1p_exp() - y.ln_1p_exp()
}

/// Bounded interval `(lo, hi)` → ℝ: `logit((x − lo)/(hi − lo))`.
pub fn interval_unconstrain<T: LazyReal>(x: T, lo: T, hi: T) -> T {
    logit((x - lo.clone()) / (hi - lo))
}

/// ℝ → bounded interval: `lo + (hi − lo)·σ(y)`.
pub fn interval_constrain<T: LazyReal>(y: T, lo: T, hi: T) -> T {
    lo.clone() + (hi - lo) * sigmoid(y)
}

/// `ln|d constrain(y)/dy| = ln(hi − lo) + log σ(y) + log(1 − σ(y))`.
pub fn interval_constrain_log_jacobian<T: LazyReal>(y: T, lo: T, hi: T) -> T {
    (hi - lo).ln() + (-(-y.clone()).ln_1p_exp() - y.ln_1p_exp())
}

// --- conjugate posterior-predictives and log marginal likelihoods ---
//
// Closed-form Bayesian evidence and posterior-predictives. The lazy `ln_gamma`
// supplies every `ln B`, `ln k!`, and `ln Γ`, so no normalizer is threaded in; each
// is branch-free over sufficient statistics or a single future datum.

/// Beta-Bernoulli posterior-predictive log-probability for a relaxed label
/// `y ∈ [0, 1]`: `y·ln(α/(α+β)) + (1−y)·ln(β/(α+β))`.
pub fn beta_bernoulli_log_predictive<T: LazyReal>(y: T, alpha: T, beta: T) -> T {
    let total = alpha.clone() + beta.clone();
    y.clone() * (alpha / total.clone()).ln() + (T::one() - y) * (beta / total).ln()
}

/// Beta-Bernoulli log marginal likelihood for `s` successes and `f` failures:
/// `ln B(α+s, β+f) − ln B(α, β)`.
pub fn beta_bernoulli_log_marginal_likelihood<T: LazyReal>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}

/// Gamma-Poisson (negative-binomial) posterior-predictive log-mass for a `count`
/// over `exposure`:
/// `ln Γ(α+k) − ln Γ(α) − ln k! + α·ln β + k·ln e − (α+k)·ln(β+e)`.
pub fn gamma_poisson_log_predictive<T: LazyReal>(count: T, shape: T, rate: T, exposure: T) -> T {
    (shape.clone() + count.clone()).ln_gamma()
        - shape.clone().ln_gamma()
        - (count.clone() + T::one()).ln_gamma()
        + shape.clone() * rate.clone().ln()
        + count.clone() * exposure.clone().ln()
        - (shape + count) * (rate + exposure).ln()
}

/// Gamma-Exponential posterior-predictive log-density (a Lomax) on `x ≥ 0`:
/// `ln α + α·ln β − (α+1)·ln(β + x)`.
pub fn gamma_exponential_log_predictive<T: LazyReal>(x: T, shape: T, rate: T) -> T {
    shape.clone().ln() + shape.clone() * rate.clone().ln() - (shape + T::one()) * (rate + x).ln()
}

/// Normal log marginal likelihood with known observation variance and a Normal
/// prior on the mean, from the sufficient statistics `(n, Σx, Σx²)`.
///
/// The Gaussian integral from completing the square in `µ`:
/// `−½n·ln(2π σ²) + ½(ln λ₀ − ln λₙ) − ½·quad`, with `λ₀ = 1/τ₀²`, `λₙ = λ₀ + n/σ²`,
/// `ηₙ = λ₀µ₀ + Σx/σ²`, and `quad = Σx²/σ² + λ₀µ₀² − ηₙ²/λₙ`.
pub fn normal_known_variance_log_marginal_likelihood<T: LazyReal>(
    count: T,
    sum: T,
    sum_sq: T,
    prior_mean: T,
    prior_std: T,
    obs_std: T,
) -> T {
    let sigma2 = obs_std.clone() * obs_std;
    let lambda0 = T::one() / (prior_std.clone() * prior_std);
    let eta_n = lambda0.clone() * prior_mean.clone() + sum / sigma2.clone();
    let lambda_n = lambda0.clone() + count.clone() / sigma2.clone();
    let quad = sum_sq / sigma2.clone() + lambda0.clone() * prior_mean.clone() * prior_mean
        - eta_n.clone() * eta_n / lambda_n.clone();
    -T::half() * count * (T::tau() * sigma2).ln() + T::half() * (lambda0.ln() - lambda_n.ln())
        - T::half() * quad
}

/// Gamma-Poisson log marginal likelihood over count sufficient statistics
/// `(Σk, n)`, the rate integrated out:
/// `α·ln β − ln Γ(α) − Σ ln kᵢ! + ln Γ(α+Σk) − (α+Σk)·ln(β+n)`.
///
/// `ln_factorial_sum = Σ ln kᵢ!` is the data-combinatorial term and is caller
/// supplied (like the `ln_binomial` in [`binomial_log_pmf`]). For Poisson exposure
/// offsets, pass `n = Σ eᵢ` and add the caller's `Σ kᵢ·ln eᵢ` to the result.
pub fn gamma_poisson_log_marginal_likelihood<T: LazyReal>(
    shape: T,
    rate: T,
    count_sum: T,
    count_n: T,
    ln_factorial_sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma() - ln_factorial_sum
        + (shape.clone() + count_sum.clone()).ln_gamma()
        - (shape + count_sum) * (rate + count_n).ln()
}

/// Gamma-Exponential log marginal likelihood over sufficient statistics `(n, Σx)`,
/// the rate integrated out:
/// `α·ln β − ln Γ(α) + ln Γ(α+n) − (α+n)·ln(β+Σx)`.
///
/// The same form carries the right-censored/exposure case: pass the failure count
/// for `n` and the total exposure for `sum`.
pub fn gamma_exponential_log_marginal_likelihood<T: LazyReal>(
    shape: T,
    rate: T,
    count_n: T,
    sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma()
        + (shape.clone() + count_n.clone()).ln_gamma()
        - (shape + count_n) * (rate + sum).ln()
}

/// Beta-Binomial posterior-predictive log-probability for one future fixed-trial
/// binomial observation (`successes`, `failures = trials − successes`):
/// `ln C(trials, successes) + ln B(α+successes, β+failures) − ln B(α, β)`.
///
/// `ln_binomial = ln C(trials, successes)` is caller supplied.
pub fn beta_binomial_log_predictive<T: LazyReal>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
    ln_binomial: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    ln_binomial + ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}

/// Dirichlet-categorical log marginal likelihood for category `counts` under a
/// `Dirichlet(α)` prior: `ln B(α + counts) − ln B(α)`, with the multivariate Beta
/// `ln B(v) = Σ ln Γ(vᵢ) − ln Γ(Σ vᵢ)`. `None` on length mismatch or empty input.
pub fn dirichlet_categorical_log_marginal_likelihood<T: LazyReal>(
    alpha: &[T],
    counts: &[T],
) -> Option<T> {
    if alpha.len() != counts.len() || alpha.is_empty() {
        return None;
    }
    let posterior: Vec<T> = alpha
        .iter()
        .cloned()
        .zip(counts.iter().cloned())
        .map(|(a, c)| a + c)
        .collect();
    Some(ln_multivariate_beta(&posterior)? - ln_multivariate_beta(alpha)?)
}

/// Normal-Inverse-Gamma log marginal likelihood (both mean and variance integrated
/// out) over sufficient statistics `(n, Σx, Σx²)` and the prior `(µ₀, λ₀, α₀, β₀)`:
/// `ln Γ(αₙ) − ln Γ(α₀) + α₀·ln β₀ − αₙ·ln βₙ + ½(ln λ₀ − ln λₙ) − ½n·ln(2π)`.
///
/// The posterior is recovered in the additive natural coordinates
/// `(λ, λµ, 2α, 2β+λµ²)`: `λₙ = λ₀+n`, `αₙ = α₀+n/2`, `λµₙ = λ₀µ₀+Σx`,
/// `2βₙ = (2β₀+λ₀µ₀²+Σx²) − λµₙ²/λₙ`. This is also the **Normal-Gamma** evidence —
/// the precision-parametrized twin shares the identical closed form.
pub fn normal_inverse_gamma_log_marginal_likelihood<T: LazyReal>(
    count: T,
    sum: T,
    sum_sq: T,
    prior_mean: T,
    prior_lambda: T,
    prior_alpha: T,
    prior_beta: T,
) -> T {
    let lambda_n = prior_lambda.clone() + count.clone();
    let lambda_mu_n = prior_lambda.clone() * prior_mean.clone() + sum;
    let alpha_n = prior_alpha.clone() + T::half() * count.clone();
    let sum_sq_nat = T::two() * prior_beta.clone()
        + prior_lambda.clone() * prior_mean.clone() * prior_mean
        + sum_sq;
    let beta_n = T::half() * (sum_sq_nat - lambda_mu_n.clone() * lambda_mu_n / lambda_n.clone());
    alpha_n.clone().ln_gamma() - prior_alpha.clone().ln_gamma() + prior_alpha * prior_beta.ln()
        - alpha_n * beta_n.ln()
        + T::half() * (prior_lambda.ln() - lambda_n.ln())
        - T::half() * count * T::ln_tau()
}

// --- analytic higher moments (skewness, excess kurtosis, non-trivial mean/var) ---
//
// The closed-form moments that are genuine functions of the shape parameters —
// pure algebra. (The constant ones — Normal skew 0, exponential skew 2 — need no
// kernel; the trivial location/scale means are the parameters themselves.)

/// Log-normal mean `exp(µ + σ²/2)`.
pub fn lognormal_mean<T: LazyReal>(mean_log: T, std_log: T) -> T {
    (mean_log + std_log.clone() * std_log * T::half()).exp()
}

/// Log-normal variance `(exp(σ²) − 1)·exp(2µ + σ²)`.
pub fn lognormal_variance<T: LazyReal>(mean_log: T, std_log: T) -> T {
    let var_log = std_log.clone() * std_log;
    (var_log.clone().exp() - T::one()) * (T::two() * mean_log + var_log).exp()
}

/// Gamma variance `shape / rate²`.
pub fn gamma_variance<T: LazyReal>(shape: T, rate: T) -> T {
    shape / (rate.clone() * rate)
}

/// Beta variance `αβ / ((α+β)²·(α+β+1))`.
pub fn beta_variance<T: LazyReal>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    (alpha * beta) / (sum.clone() * sum.clone() * (sum + T::one()))
}

/// Gamma skewness `2/√shape`.
pub fn gamma_skewness<T: LazyReal>(shape: T) -> T {
    T::two() / shape.sqrt()
}

/// Gamma excess kurtosis `6/shape`.
pub fn gamma_excess_kurtosis<T: LazyReal>(shape: T) -> T {
    T::scalar(6.0) / shape
}

/// Beta skewness `2(β−α)·√(α+β+1) / ((α+β+2)·√(αβ))`.
pub fn beta_skewness<T: LazyReal>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    T::two() * (beta.clone() - alpha.clone()) * (sum.clone() + T::one()).sqrt()
        / ((sum + T::two()) * (alpha * beta).sqrt())
}

/// Chi-squared skewness `√(8/freedom)`.
pub fn chi_squared_skewness<T: LazyReal>(freedom: T) -> T {
    (T::scalar(8.0) / freedom).sqrt()
}

/// Chi-squared excess kurtosis `12/freedom`.
pub fn chi_squared_excess_kurtosis<T: LazyReal>(freedom: T) -> T {
    T::scalar(12.0) / freedom
}

/// Wald / inverse-Gaussian skewness `3√(µ/λ)`.
pub fn wald_skewness<T: LazyReal>(mean: T, shape: T) -> T {
    T::scalar(3.0) * (mean / shape).sqrt()
}

/// Wald / inverse-Gaussian excess kurtosis `15µ/λ`.
pub fn wald_excess_kurtosis<T: LazyReal>(mean: T, shape: T) -> T {
    T::scalar(15.0) * mean / shape
}

/// Poisson skewness `1/√λ`.
pub fn poisson_skewness<T: LazyReal>(rate: T) -> T {
    T::one() / rate.sqrt()
}

/// Poisson excess kurtosis `1/λ`.
pub fn poisson_excess_kurtosis<T: LazyReal>(rate: T) -> T {
    T::one() / rate
}

/// Bernoulli skewness `(q − p)/√(p·q)`, `q = 1 − p`.
pub fn bernoulli_skewness<T: LazyReal>(p: T) -> T {
    let q = T::one() - p.clone();
    (q.clone() - p.clone()) / (p * q).sqrt()
}

/// Binomial skewness `(1 − 2p)/√(n·p·(1−p))`.
pub fn binomial_skewness<T: LazyReal>(trials: T, p: T) -> T {
    let variance = trials * p.clone() * (T::one() - p.clone());
    (T::one() - T::two() * p) / variance.sqrt()
}

/// Geometric skewness `(2 − p)/√(1 − p)`.
pub fn geometric_skewness<T: LazyReal>(p: T) -> T {
    (T::two() - p.clone()) / (T::one() - p).sqrt()
}

/// Negative-binomial skewness `(2 − p)/√(r·(1 − p))`.
pub fn negative_binomial_skewness<T: LazyReal>(successes: T, p: T) -> T {
    (T::two() - p.clone()) / (successes * (T::one() - p)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The scalar/lazy kernels are exercised end-to-end here against bayes' own concrete
    // distributions (euclid-only families). The multivariate-matrix, conjugate-matrix,
    // and S² directional families moved to the sibling crate `ito`; the cross-checks that
    // these same kernels agree with `ito::MultivariateNormal` / `ito::VonMisesFisher` /
    // `ito::Dirichlet` (etc.) live in ito's integration test
    // `tests/bayes_kernels_on_eager.rs`.
    use crate::distribution::{
        Bernoulli, Beta, BetaBernoulli, BetaBinomialModel, Binomial, Categorical, Cauchy,
        ChiSquared, ChiSquaredDivergence, Continuous, ContinuousCdf, Discrete, DiscreteCdf,
        DiscreteUniform, Exponential, FisherSnedecor, Frechet, Gamma, GammaExponential,
        GammaPoisson, Geometric, Gumbel, HalfNormal, HellingerDistance, Hypergeometric,
        InverseGamma, KlDivergence, Laplace, LogNormal, Logistic, Lomax, Moments,
        NegativeBinomial, Normal, OrderStatistic, Pareto, Poisson, Rayleigh, RenyiDivergence,
        StudentT, TotalVariation, Triangular, Uniform, VonMises, Wald, Weibull,
    };
    use crate::function::bessel::ln_i0;
    use crate::function::beta::ln_beta;
    use crate::function::factorial::{ln_binomial, ln_factorial};
    use crate::function::gamma::ln_gamma;
    use crate::function::multivariate::ln_multivariate_beta;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol + tol * b.abs(),
            "observed {a}, expected {b}"
        );
    }

    #[test]
    fn lazy_real_continuous_kernels_match_distribution_log_densities() {
        let normal = Normal::new(1.0, 2.0).unwrap();
        close(normal_log_pdf(0.25, 1.0, 2.0), normal.ln_pdf(0.25), 1e-12);

        let uniform = Uniform::new(-2.0, 3.0).unwrap();
        close(uniform_log_pdf_on_support(5.0), uniform.ln_pdf(0.25), 1e-12);

        let exponential = Exponential::new(1.7).unwrap();
        close(
            exponential_log_pdf_on_support(0.4, 1.7),
            exponential.ln_pdf(0.4),
            1e-12,
        );

        let gamma = Gamma::new(2.5, 1.2).unwrap();
        close(
            gamma_log_pdf_on_support(0.8, 2.5, 1.2, ln_gamma(2.5)),
            gamma.ln_pdf(0.8),
            1e-12,
        );

        let beta = Beta::new(2.0, 5.0).unwrap();
        close(
            beta_log_pdf_on_unit(0.35, 2.0, 5.0, ln_beta(2.0, 5.0)),
            beta.ln_pdf(0.35),
            1e-12,
        );

        let lognormal = LogNormal::new(0.2, 0.7).unwrap();
        close(
            lognormal_log_pdf_on_support(1.4, 0.2, 0.7),
            lognormal.ln_pdf(1.4),
            1e-12,
        );

        let logistic = Logistic::new(-0.5, 1.3).unwrap();
        close(
            logistic_log_pdf(0.2, -0.5, 1.3),
            logistic.ln_pdf(0.2),
            1e-12,
        );

        let laplace = Laplace::new(0.5, 1.7).unwrap();
        close(laplace_log_pdf(-0.2, 0.5, 1.7), laplace.ln_pdf(-0.2), 1e-12);

        let cauchy = Cauchy::new(0.5, 1.7).unwrap();
        close(cauchy_log_pdf(-0.2, 0.5, 1.7), cauchy.ln_pdf(-0.2), 1e-12);

        let student = StudentT::new(6.0).unwrap();
        let log_normalizer =
            0.5 * (6.0 * core::f64::consts::PI).ln() + ln_gamma(3.0) - ln_gamma(3.5);
        close(
            student_t_log_pdf(0.4, 0.0, 1.0, 6.0, log_normalizer),
            student.ln_pdf(0.4),
            1e-12,
        );

        let pareto = Pareto::new(2.0, 3.0).unwrap();
        close(
            pareto_log_pdf_on_support(4.0, 2.0, 3.0),
            pareto.ln_pdf(4.0),
            1e-12,
        );

        let rayleigh = Rayleigh::new(1.4).unwrap();
        close(
            rayleigh_log_pdf_on_support(0.9, 1.4),
            rayleigh.ln_pdf(0.9),
            1e-12,
        );

        let weibull = Weibull::new(1.4, 2.2).unwrap();
        close(
            weibull_log_pdf_on_support(0.9, 1.4, 2.2),
            weibull.ln_pdf(0.9),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_discrete_kernels_match_distribution_log_masses() {
        let binomial = Binomial::new(12, 0.3).unwrap();
        close(
            binomial_log_pmf(4.0, 8.0, 0.3, ln_binomial(12, 4)),
            binomial.ln_pmf(4),
            1e-12,
        );

        let geometric = Geometric::new(0.25).unwrap();
        close(geometric_log_pmf(5.0, 0.25), geometric.ln_pmf(5), 1e-12);

        let poisson = Poisson::new(3.5).unwrap();
        close(
            poisson_log_pmf(6.0, 3.5, ln_factorial(6)),
            poisson.ln_pmf(6),
            1e-12,
        );

        let discrete_uniform = DiscreteUniform::new(-2, 5).unwrap();
        close(
            discrete_uniform_log_pmf_on_support(7.0),
            discrete_uniform.ln_pmf(3),
            1e-12,
        );

        close(
            bernoulli_logit_log_pmf(1.0, 2.0),
            -2.0_f64.ln_1p_exp() + 2.0,
            1e-12,
        );
    }

    #[test]
    fn lazy_real_vector_kernels_sum_component_log_densities() {
        let got = diagonal_normal_log_pdf(&[0.0, 2.0], &[0.0, 1.0], &[1.0, 2.0]).unwrap();
        let expected =
            Normal::new(0.0, 1.0).unwrap().ln_pdf(0.0) + Normal::new(1.0, 2.0).unwrap().ln_pdf(2.0);
        close(got, expected, 1e-12);

        // The Dirichlet-on-simplex cross-check (against the concrete distribution) moved to
        // ito; here we cross-check the kernel against the bayes-local `ln_multivariate_beta`
        // normalizer it consumes — `ln f = (α−1)·ln x − ln B(α)`.
        let alpha = [2.0, 3.0, 5.0];
        let x = [0.2, 0.3, 0.5];
        let got =
            dirichlet_log_pdf_on_simplex(&x, &alpha, ln_multivariate_beta(&alpha)).unwrap();
        let expected = alpha
            .iter()
            .zip(&x)
            .map(|(&a, &xi)| (a - 1.0) * xi.ln())
            .sum::<f64>()
            - ln_multivariate_beta(&alpha);
        close(got, expected, 1e-12);
    }

    #[test]
    fn lazy_real_moment_kernels_match_distribution() {
        let lognormal = LogNormal::new(0.2, 0.7).unwrap();
        close(lognormal_mean(0.2, 0.7), lognormal.mean().unwrap(), 1e-12);
        close(
            lognormal_variance(0.2, 0.7),
            lognormal.variance().unwrap(),
            1e-12,
        );

        let gamma = Gamma::new(2.5, 1.2).unwrap();
        close(gamma_variance(2.5, 1.2), gamma.variance().unwrap(), 1e-12);
        close(gamma_skewness(2.5), gamma.skewness().unwrap(), 1e-12);
        close(gamma_excess_kurtosis(2.5), gamma.kurtosis().unwrap(), 1e-12);

        let beta = Beta::new(2.0, 5.0).unwrap();
        close(beta_variance(2.0, 5.0), beta.variance().unwrap(), 1e-12);
        close(beta_skewness(2.0, 5.0), beta.skewness().unwrap(), 1e-12);

        let chi = ChiSquared::new(5.0).unwrap();
        close(chi_squared_skewness(5.0), chi.skewness().unwrap(), 1e-12);
        close(
            chi_squared_excess_kurtosis(5.0),
            chi.kurtosis().unwrap(),
            1e-12,
        );

        let wald = Wald::new(1.0, 2.0).unwrap();
        close(wald_skewness(1.0, 2.0), wald.skewness().unwrap(), 1e-12);
        close(
            wald_excess_kurtosis(1.0, 2.0),
            wald.kurtosis().unwrap(),
            1e-12,
        );

        let poisson = Poisson::new(3.5).unwrap();
        close(poisson_skewness(3.5), poisson.skewness().unwrap(), 1e-12);
        close(
            poisson_excess_kurtosis(3.5),
            poisson.kurtosis().unwrap(),
            1e-12,
        );

        let binomial = Binomial::new(12, 0.3).unwrap();
        close(
            binomial_skewness(12.0, 0.3),
            binomial.skewness().unwrap(),
            1e-12,
        );

        let geometric = Geometric::new(0.25).unwrap();
        close(
            geometric_skewness(0.25),
            geometric.skewness().unwrap(),
            1e-12,
        );

        let negbin = NegativeBinomial::new(5.0, 0.4).unwrap();
        close(
            negative_binomial_skewness(5.0, 0.4),
            negbin.skewness().unwrap(),
            1e-12,
        );

        let bernoulli = Bernoulli::new(0.3).unwrap();
        close(
            bernoulli_skewness(0.3),
            bernoulli.skewness().unwrap(),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_conjugate_kernels_match_distribution() {
        // Beta-Bernoulli predictive (success and failure) and marginal likelihood.
        let bb = BetaBernoulli::new(Beta::new(2.0, 3.0).unwrap());
        close(
            beta_bernoulli_log_predictive(1.0, 2.0, 3.0),
            bb.ln_predictive(true),
            1e-12,
        );
        close(
            beta_bernoulli_log_predictive(0.0, 2.0, 3.0),
            bb.ln_predictive(false),
            1e-12,
        );
        // 4 successes, 2 failures.
        let data = [true, true, true, true, false, false];
        close(
            beta_bernoulli_log_marginal_likelihood(2.0, 3.0, 4.0, 2.0),
            bb.ln_marginal_likelihood(&data),
            1e-12,
        );

        // Gamma-Poisson predictive (exposure 1 and a custom exposure).
        let gp = GammaPoisson::new(Gamma::new(2.5, 1.2).unwrap());
        close(
            gamma_poisson_log_predictive(3.0, 2.5, 1.2, 1.0),
            gp.ln_predictive(3),
            1e-12,
        );
        close(
            gamma_poisson_log_predictive(3.0, 2.5, 1.2, 0.7),
            gp.ln_predictive_with_exposure(3, 0.7),
            1e-12,
        );

        // Gamma-Exponential predictive (a Lomax).
        let ge = GammaExponential::new(Gamma::new(2.0, 1.5).unwrap());
        close(
            gamma_exponential_log_predictive(0.8, 2.0, 1.5),
            ge.ln_predictive(0.8),
            1e-12,
        );

        // Normal known-variance / Normal-Inverse-Gamma / Normal-Gamma cross-checks against
        // their concrete conjugate models moved to ito (the matrix-conjugate crate); the
        // marginal-likelihood kernels themselves stay here and are exercised there.

        // Gamma-Poisson marginal likelihood over count sufficient statistics.
        let counts = [2u64, 0, 3, 1];
        let count_sum = counts.iter().sum::<u64>() as f64;
        let ln_factorial_sum: f64 = counts.iter().map(|&k| ln_factorial(k)).sum();
        close(
            gamma_poisson_log_marginal_likelihood(
                2.5,
                1.2,
                count_sum,
                counts.len() as f64,
                ln_factorial_sum,
            ),
            gp.ln_marginal_likelihood(&counts),
            1e-12,
        );

        // Gamma-Exponential marginal likelihood over (n, Σx).
        let durations = [0.8, 1.2, 0.3];
        let dur_sum: f64 = durations.iter().sum();
        close(
            gamma_exponential_log_marginal_likelihood(2.0, 1.5, durations.len() as f64, dur_sum),
            ge.ln_marginal_likelihood(&durations),
            1e-12,
        );

        // Beta-Binomial predictive (10 trials, 4 successes ⇒ 6 failures).
        let bbm = BetaBinomialModel::new(Beta::new(2.0, 3.0).unwrap(), 10);
        close(
            beta_binomial_log_predictive(2.0, 3.0, 4.0, 6.0, ln_binomial(10, 4)),
            bbm.ln_predictive(4),
            1e-12,
        );

        // Dirichlet-categorical marginal likelihood from category counts — cross-checked
        // against the explicit `ln B(α + counts) − ln B(α)` (the concrete
        // `DirichletCategorical` cross-check moved to ito).
        let alpha = [1.0, 2.0, 0.5];
        let category_counts = [2.0, 3.0, 1.0];
        let posterior: Vec<f64> = alpha
            .iter()
            .zip(&category_counts)
            .map(|(&a, &c)| a + c)
            .collect();
        close(
            dirichlet_categorical_log_marginal_likelihood(&alpha, &category_counts).unwrap(),
            ln_multivariate_beta(&posterior) - ln_multivariate_beta(&alpha),
            1e-12,
        );

        // The Normal-Inverse-Gamma / Normal-Gamma marginal-likelihood cross-check (against
        // the concrete conjugate models) moved to ito.

        // Multivariate Beta normalizer (the Dirichlet log-normalizer).
        let mvbeta_alpha = [1.5, 2.0, 0.7];
        close(
            crate::lazy::ln_multivariate_beta(&mvbeta_alpha).unwrap(),
            ln_multivariate_beta(&mvbeta_alpha),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_scoring_rules_match_distribution() {
        let p = [0.2, 0.5, 0.3];
        let observed = 1;
        close(
            brier_score(&p, observed).unwrap(),
            crate::distribution::brier_score(&p, observed),
            1e-12,
        );
        close(
            spherical_score(&p, observed).unwrap(),
            crate::distribution::spherical_score(&p, observed),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_reparameterization_transforms_match_constraint() {
        use crate::distribution::Constraint;

        let positive = Constraint::Positive;
        close(
            positive_unconstrain(1.5),
            positive.to_unconstrained(1.5),
            1e-12,
        );
        close(positive_constrain(0.7), positive.to_constrained(0.7), 1e-12);
        close(
            positive_constrain_log_jacobian(0.7),
            positive.log_abs_det_jacobian(0.7),
            1e-12,
        );

        let unit = Constraint::UnitInterval;
        close(
            unit_interval_unconstrain(0.4),
            unit.to_unconstrained(0.4),
            1e-12,
        );
        close(
            unit_interval_constrain(0.6),
            unit.to_constrained(0.6),
            1e-12,
        );
        close(
            unit_interval_constrain_log_jacobian(0.6),
            unit.log_abs_det_jacobian(0.6),
            1e-12,
        );

        let interval = Constraint::Interval { lo: -2.0, hi: 3.0 };
        close(
            interval_unconstrain(0.5, -2.0, 3.0),
            interval.to_unconstrained(0.5),
            1e-12,
        );
        close(
            interval_constrain(0.4, -2.0, 3.0),
            interval.to_constrained(0.4),
            1e-12,
        );
        close(
            interval_constrain_log_jacobian(0.4, -2.0, 3.0),
            interval.log_abs_det_jacobian(0.4),
            1e-12,
        );

        // Round-trip: constrain ∘ unconstrain = id.
        close(positive_constrain(positive_unconstrain(2.3)), 2.3, 1e-12);
        close(
            unit_interval_constrain(unit_interval_unconstrain(0.35)),
            0.35,
            1e-12,
        );
        close(
            interval_constrain(interval_unconstrain(1.2, -2.0, 3.0), -2.0, 3.0),
            1.2,
            1e-12,
        );
    }

    #[test]
    fn lazy_real_log_sum_exp_family_matches_stable() {
        use crate::function::stable;
        let xs = [1.0, 2.0, 0.5, -1.0, 3.2];
        close(log_sum_exp(&xs).unwrap(), stable::ln_sum_exp(&xs), 1e-12);
        close(log_mean_exp(&xs).unwrap(), stable::ln_mean_exp(&xs), 1e-12);
        for (got, want) in log_softmax(&xs)
            .unwrap()
            .iter()
            .zip(&stable::log_softmax(&xs))
        {
            close(*got, *want, 1e-12);
        }
        for (got, want) in softmax(&xs).unwrap().iter().zip(&stable::softmax(&xs)) {
            close(*got, *want, 1e-12);
        }
    }

    #[test]
    fn lazy_real_probability_vector_divergences_match_distribution() {
        let p = [0.2, 0.5, 0.3];
        let q = [0.1, 0.6, 0.3];
        let alpha = 0.5;

        // Information-theory free functions.
        close(
            shannon_entropy(&p).unwrap(),
            crate::distribution::shannon_entropy(&p),
            1e-12,
        );
        close(
            cross_entropy(&p, &q).unwrap(),
            crate::distribution::cross_entropy(&p, &q),
            1e-12,
        );
        close(
            jensen_shannon(&p, &q).unwrap(),
            crate::distribution::jensen_shannon(&p, &q),
            1e-12,
        );
        let bc = crate::distribution::bhattacharyya_coefficient(&p, &q);
        close(bhattacharyya_coefficient(&p, &q).unwrap(), bc, 1e-12);
        close(hellinger_sq_categorical(&p, &q).unwrap(), 1.0 - bc, 1e-12);

        // Categorical divergences via the distribution.
        let cp = Categorical::new(&p).unwrap();
        let cq = Categorical::new(&q).unwrap();
        close(
            categorical_kl(&p, &q).unwrap(),
            cp.kl_divergence(&cq),
            1e-12,
        );
        close(
            categorical_chi_squared(&p, &q).unwrap(),
            cp.chi_squared_divergence(&cq),
            1e-12,
        );
        close(
            categorical_total_variation(&p, &q).unwrap(),
            cp.total_variation(&cq),
            1e-12,
        );
        close(
            categorical_renyi(&p, &q, alpha).unwrap(),
            cp.renyi_divergence(&cq, alpha),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_dirichlet_vector_kernels_match_explicit_closed_form() {
        // The concrete-distribution cross-check moved to ito; here the Dirichlet KL and
        // entropy kernels are checked against their explicit closed forms built from the
        // bayes-local `ln_multivariate_beta` / `digamma` (no concrete distribution).
        use crate::function::gamma::digamma;
        let alpha = [2.0, 3.0, 5.0];
        let beta = [1.5, 2.0, 4.0];
        let a0: f64 = alpha.iter().sum();

        // D(α ‖ β) = ln B(β) − ln B(α) + Σ (αᵢ − βᵢ)·(ψ(αᵢ) − ψ(α₀)).
        let expected_kl = ln_multivariate_beta(&beta) - ln_multivariate_beta(&alpha)
            + alpha
                .iter()
                .zip(&beta)
                .map(|(&a, &b)| (a - b) * (digamma(a) - digamma(a0)))
                .sum::<f64>();
        close(dirichlet_kl(&alpha, &beta).unwrap(), expected_kl, 1e-12);

        // H(α) = ln B(α) + (α₀ − K)·ψ(α₀) − Σ (αᵢ − 1)·ψ(αᵢ).
        let k = alpha.len() as f64;
        let expected_entropy = ln_multivariate_beta(&alpha) + (a0 - k) * digamma(a0)
            - alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum::<f64>();
        close(dirichlet_entropy(&alpha).unwrap(), expected_entropy, 1e-12);
    }

    #[test]
    fn lazy_real_stable_log_tails_match_distribution() {
        let exponential = Exponential::new(1.7).unwrap();
        close(
            exponential_ln_cdf_on_support(0.4, 1.7),
            exponential.ln_cdf(0.4),
            1e-12,
        );

        let logistic = Logistic::new(-0.5, 1.3).unwrap();
        close(logistic_ln_cdf(0.2, -0.5, 1.3), logistic.ln_cdf(0.2), 1e-12);
        close(logistic_ln_sf(0.2, -0.5, 1.3), logistic.ln_sf(0.2), 1e-12);

        // Gumbel has no ln_cdf method; check against ln(cdf).
        let gumbel = Gumbel::new(1.0, 2.0).unwrap();
        close(gumbel_ln_cdf(1.5, 1.0, 2.0), gumbel.cdf(1.5).ln(), 1e-12);
    }

    #[test]
    fn lazy_real_hellinger_kernels_match_distribution() {
        let normal = (
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(1.0, 2.0).unwrap(),
        );
        close(
            hellinger_sq_normal(0.0, 1.0, 1.0, 2.0),
            normal.0.hellinger_sq(&normal.1),
            1e-12,
        );

        let exponential = (
            Exponential::new(1.5).unwrap(),
            Exponential::new(0.7).unwrap(),
        );
        close(
            hellinger_sq_exponential(1.5, 0.7),
            exponential.0.hellinger_sq(&exponential.1),
            1e-12,
        );

        let poisson = (Poisson::new(2.5).unwrap(), Poisson::new(4.0).unwrap());
        close(
            hellinger_sq_poisson(2.5, 4.0),
            poisson.0.hellinger_sq(&poisson.1),
            1e-12,
        );

        let bernoulli = (Bernoulli::new(0.3).unwrap(), Bernoulli::new(0.7).unwrap());
        close(
            hellinger_sq_bernoulli(0.3, 0.7),
            bernoulli.0.hellinger_sq(&bernoulli.1),
            1e-12,
        );

        let gamma = (Gamma::new(2.5, 1.2).unwrap(), Gamma::new(3.0, 0.8).unwrap());
        close(
            hellinger_sq_gamma(2.5, 1.2, 3.0, 0.8),
            gamma.0.hellinger_sq(&gamma.1),
            1e-12,
        );

        let beta = (Beta::new(2.0, 5.0).unwrap(), Beta::new(3.0, 4.0).unwrap());
        close(
            hellinger_sq_beta(2.0, 5.0, 3.0, 4.0),
            beta.0.hellinger_sq(&beta.1),
            1e-12,
        );

        let wald = (Wald::new(1.0, 2.0).unwrap(), Wald::new(1.5, 3.0).unwrap());
        close(
            hellinger_sq_wald(1.0, 2.0, 1.5, 3.0),
            wald.0.hellinger_sq(&wald.1),
            1e-12,
        );

        // Binomial with a matched trial count n = 12 (BC_Bernoulli^n).
        let binomial = (
            Binomial::new(12, 0.3).unwrap(),
            Binomial::new(12, 0.5).unwrap(),
        );
        close(
            hellinger_sq_binomial(12.0, 0.3, 0.5),
            binomial.0.hellinger_sq(&binomial.1),
            1e-12,
        );

        // Negative-binomial with matched successes r = 5.
        let negbin = (
            NegativeBinomial::new(5.0, 0.4).unwrap(),
            NegativeBinomial::new(5.0, 0.6).unwrap(),
        );
        close(
            hellinger_sq_negative_binomial(5.0, 0.4, 0.6),
            negbin.0.hellinger_sq(&negbin.1),
            1e-12,
        );

        // Uniform / discrete-uniform squared Hellinger (general overlap).
        let uniform = (
            Uniform::new(-1.0, 2.0).unwrap(),
            Uniform::new(0.0, 4.0).unwrap(),
        );
        close(
            hellinger_sq_uniform(-1.0, 2.0, 0.0, 4.0),
            uniform.0.hellinger_sq(&uniform.1),
            1e-12,
        );
        let du = (
            DiscreteUniform::new(0, 3).unwrap(),
            DiscreteUniform::new(2, 6).unwrap(),
        );
        close(
            hellinger_sq_discrete_uniform(0.0, 3.0, 2.0, 6.0),
            du.0.hellinger_sq(&du.1),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_chi_squared_renyi_tv_kernels_match_distribution() {
        let alpha = 0.5;

        let bernoulli = (Bernoulli::new(0.3).unwrap(), Bernoulli::new(0.7).unwrap());
        close(
            chi_squared_bernoulli(0.3, 0.7),
            bernoulli.0.chi_squared_divergence(&bernoulli.1),
            1e-12,
        );
        close(
            renyi_bernoulli(0.3, 0.7, alpha),
            bernoulli.0.renyi_divergence(&bernoulli.1, alpha),
            1e-12,
        );
        close(
            total_variation_bernoulli(0.3, 0.7),
            bernoulli.0.total_variation(&bernoulli.1),
            1e-12,
        );

        let poisson = (Poisson::new(2.5).unwrap(), Poisson::new(4.0).unwrap());
        close(
            chi_squared_poisson(2.5, 4.0),
            poisson.0.chi_squared_divergence(&poisson.1),
            1e-12,
        );

        // χ² / Rényi interiors in their convergence region.
        let normal = (
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(0.5, 1.2).unwrap(),
        );
        close(
            chi_squared_normal_on_support(0.0, 1.0, 0.5, 1.2),
            normal.0.chi_squared_divergence(&normal.1),
            1e-12,
        );
        close(
            renyi_normal_on_support(0.0, 1.0, 0.5, 1.2, alpha),
            normal.0.renyi_divergence(&normal.1, alpha),
            1e-12,
        );

        let exponential = (
            Exponential::new(1.5).unwrap(),
            Exponential::new(1.0).unwrap(),
        );
        close(
            chi_squared_exponential_on_support(1.5, 1.0),
            exponential.0.chi_squared_divergence(&exponential.1),
            1e-12,
        );

        let gamma = (Gamma::new(2.5, 1.2).unwrap(), Gamma::new(3.0, 0.8).unwrap());
        close(
            chi_squared_gamma_on_support(2.5, 1.2, 3.0, 0.8),
            gamma.0.chi_squared_divergence(&gamma.1),
            1e-12,
        );
        close(
            renyi_gamma_on_support(2.5, 1.2, 3.0, 0.8, alpha),
            gamma.0.renyi_divergence(&gamma.1, alpha),
            1e-12,
        );

        // Binomial with a matched trial count n = 12 (always finite).
        let binomial = (
            Binomial::new(12, 0.3).unwrap(),
            Binomial::new(12, 0.5).unwrap(),
        );
        close(
            chi_squared_binomial(12.0, 0.3, 0.5),
            binomial.0.chi_squared_divergence(&binomial.1),
            1e-12,
        );
        close(
            renyi_binomial(12.0, 0.3, 0.5, alpha),
            binomial.0.renyi_divergence(&binomial.1, alpha),
            1e-12,
        );

        // Negative-binomial interiors in their convergence region (matched r = 5).
        let negbin = (
            NegativeBinomial::new(5.0, 0.6).unwrap(),
            NegativeBinomial::new(5.0, 0.4).unwrap(),
        );
        close(
            chi_squared_negative_binomial_on_support(5.0, 0.6, 0.4),
            negbin.0.chi_squared_divergence(&negbin.1),
            1e-12,
        );
        close(
            renyi_negative_binomial_on_support(5.0, 0.6, 0.4, alpha),
            negbin.0.renyi_divergence(&negbin.1, alpha),
            1e-12,
        );

        // Wald / inverse-Gaussian interiors in their convergence region.
        let wald = (Wald::new(1.0, 3.0).unwrap(), Wald::new(1.2, 2.0).unwrap());
        close(
            chi_squared_wald_on_support(1.0, 3.0, 1.2, 2.0),
            wald.0.chi_squared_divergence(&wald.1),
            1e-12,
        );
        close(
            renyi_wald_on_support(1.0, 3.0, 1.2, 2.0, alpha),
            wald.0.renyi_divergence(&wald.1, alpha),
            1e-12,
        );

        // Uniform / discrete-uniform χ² and Rényi on nested support.
        let uniform = (
            Uniform::new(-1.0, 2.0).unwrap(),
            Uniform::new(-2.0, 3.0).unwrap(),
        );
        close(
            chi_squared_uniform_on_support(3.0, 5.0),
            uniform.0.chi_squared_divergence(&uniform.1),
            1e-12,
        );
        close(
            renyi_uniform_on_support(-1.0, 2.0, -2.0, 3.0, alpha),
            uniform.0.renyi_divergence(&uniform.1, alpha),
            1e-12,
        );
        let du = (
            DiscreteUniform::new(0, 3).unwrap(),
            DiscreteUniform::new(-1, 5).unwrap(),
        );
        close(
            chi_squared_discrete_uniform_on_support(4.0, 7.0),
            du.0.chi_squared_divergence(&du.1),
            1e-12,
        );
        close(
            renyi_discrete_uniform_on_support(0.0, 3.0, -1.0, 5.0, alpha),
            du.0.renyi_divergence(&du.1, alpha),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_von_mises_divergences_match_distribution() {
        // κ ≤ 15 keeps the distribution on its `i0(x).ln()` branch, so the scaled-
        // Bessel reformulation `ln I₀ᵉ + |x|` agrees to rounding.
        let vm = (
            VonMises::new(0.3, 2.0).unwrap(),
            VonMises::new(0.8, 1.5).unwrap(),
        );
        close(
            kl_von_mises(0.3, 2.0, 0.8, 1.5),
            vm.0.kl_divergence(&vm.1),
            1e-9,
        );
        close(
            hellinger_sq_von_mises(0.3, 2.0, 0.8, 1.5),
            vm.0.hellinger_sq(&vm.1),
            1e-9,
        );
        close(
            chi_squared_von_mises(0.3, 2.0, 0.8, 1.5),
            vm.0.chi_squared_divergence(&vm.1),
            1e-9,
        );
        close(
            renyi_von_mises(0.3, 2.0, 0.8, 1.5, 0.5),
            vm.0.renyi_divergence(&vm.1, 0.5),
            1e-9,
        );
    }

    #[test]
    fn lazy_real_kl_kernels_match_distribution() {
        // Gaussian family (Normal / LogNormal / HalfNormal share one closed form).
        let normal = (
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(1.0, 2.0).unwrap(),
        );
        close(
            kl_normal(0.0, 1.0, 1.0, 2.0),
            normal.0.kl_divergence(&normal.1),
            1e-12,
        );

        let exponential = (
            Exponential::new(1.5).unwrap(),
            Exponential::new(0.7).unwrap(),
        );
        close(
            kl_exponential(1.5, 0.7),
            exponential.0.kl_divergence(&exponential.1),
            1e-12,
        );

        let poisson = (Poisson::new(2.5).unwrap(), Poisson::new(4.0).unwrap());
        close(
            kl_poisson(2.5, 4.0),
            poisson.0.kl_divergence(&poisson.1),
            1e-12,
        );

        let geometric = (Geometric::new(0.3).unwrap(), Geometric::new(0.6).unwrap());
        close(
            kl_geometric(0.3, 0.6),
            geometric.0.kl_divergence(&geometric.1),
            1e-12,
        );

        let bernoulli = (Bernoulli::new(0.3).unwrap(), Bernoulli::new(0.7).unwrap());
        close(
            kl_bernoulli_on_unit(0.3, 0.7),
            bernoulli.0.kl_divergence(&bernoulli.1),
            1e-12,
        );

        // Negative-binomial with matched successes r = 5.
        let negbin = (
            NegativeBinomial::new(5.0, 0.4).unwrap(),
            NegativeBinomial::new(5.0, 0.6).unwrap(),
        );
        close(
            kl_negative_binomial(5.0, 0.4, 0.6),
            negbin.0.kl_divergence(&negbin.1),
            1e-12,
        );

        let cauchy = (
            Cauchy::new(0.0, 1.0).unwrap(),
            Cauchy::new(0.5, 1.7).unwrap(),
        );
        close(
            kl_cauchy(0.0, 1.0, 0.5, 1.7),
            cauchy.0.kl_divergence(&cauchy.1),
            1e-12,
        );

        let rayleigh = (Rayleigh::new(1.2).unwrap(), Rayleigh::new(2.0).unwrap());
        close(
            kl_rayleigh(1.2, 2.0),
            rayleigh.0.kl_divergence(&rayleigh.1),
            1e-12,
        );

        let laplace = (
            Laplace::new(0.0, 1.0).unwrap(),
            Laplace::new(0.5, 1.7).unwrap(),
        );
        close(
            kl_laplace(0.0, 1.0, 0.5, 1.7),
            laplace.0.kl_divergence(&laplace.1),
            1e-12,
        );

        // Wald::new(mu, shape).
        let wald = (Wald::new(1.0, 2.0).unwrap(), Wald::new(1.5, 3.0).unwrap());
        close(
            kl_wald(1.0, 2.0, 1.5, 3.0),
            wald.0.kl_divergence(&wald.1),
            1e-12,
        );

        // Gamma family (Gamma / ChiSquared / InverseGamma share one closed form).
        let gamma = (Gamma::new(2.5, 1.2).unwrap(), Gamma::new(3.0, 0.8).unwrap());
        close(
            kl_gamma(2.5, 1.2, 3.0, 0.8),
            gamma.0.kl_divergence(&gamma.1),
            1e-12,
        );

        let beta = (Beta::new(2.0, 5.0).unwrap(), Beta::new(3.0, 4.0).unwrap());
        close(
            kl_beta(2.0, 5.0, 3.0, 4.0),
            beta.0.kl_divergence(&beta.1),
            1e-12,
        );

        // Weibull::new(scale, shape); the kernel takes (shape, scale).
        let weibull = (
            Weibull::new(1.4, 2.2).unwrap(),
            Weibull::new(1.0, 1.6).unwrap(),
        );
        close(
            kl_weibull(2.2, 1.4, 1.6, 1.0),
            weibull.0.kl_divergence(&weibull.1),
            1e-12,
        );

        let gumbel = (
            Gumbel::new(0.0, 1.0).unwrap(),
            Gumbel::new(0.5, 1.5).unwrap(),
        );
        close(
            kl_gumbel(0.0, 1.0, 0.5, 1.5),
            gumbel.0.kl_divergence(&gumbel.1),
            1e-12,
        );

        // Pareto::new(minimum, shape) with nested support (xm0 ≥ xm1).
        let pareto = (
            Pareto::new(3.0, 2.0).unwrap(),
            Pareto::new(2.0, 1.5).unwrap(),
        );
        close(
            kl_pareto_on_support(3.0, 2.0, 2.0, 1.5),
            pareto.0.kl_divergence(&pareto.1),
            1e-12,
        );

        // Binomial with a matched trial count n = 12 (n × the Bernoulli KL).
        let binomial = (
            Binomial::new(12, 0.3).unwrap(),
            Binomial::new(12, 0.5).unwrap(),
        );
        close(
            kl_binomial(12.0, 0.3, 0.5),
            binomial.0.kl_divergence(&binomial.1),
            1e-12,
        );

        // Uniform / discrete-uniform on nested support.
        let uniform = (
            Uniform::new(-1.0, 2.0).unwrap(),
            Uniform::new(-2.0, 3.0).unwrap(),
        );
        close(
            kl_uniform_on_support(3.0, 5.0),
            uniform.0.kl_divergence(&uniform.1),
            1e-12,
        );
        let du = (
            DiscreteUniform::new(0, 3).unwrap(),
            DiscreteUniform::new(-1, 5).unwrap(),
        );
        close(
            kl_discrete_uniform_on_support(4.0, 7.0),
            du.0.kl_divergence(&du.1),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_cdf_kernels_match_distribution() {
        let normal = Normal::new(1.0, 2.0).unwrap();
        close(normal_cdf(0.25, 1.0, 2.0), normal.cdf(0.25), 1e-12);

        let half_normal = HalfNormal::new(1.4).unwrap();
        close(
            half_normal_cdf_on_support(0.9, 1.4),
            half_normal.cdf(0.9),
            1e-12,
        );

        let exponential = Exponential::new(1.7).unwrap();
        close(
            exponential_cdf_on_support(0.4, 1.7),
            exponential.cdf(0.4),
            1e-12,
        );

        let logistic = Logistic::new(-0.5, 1.3).unwrap();
        close(logistic_cdf(0.2, -0.5, 1.3), logistic.cdf(0.2), 1e-12);

        let gumbel = Gumbel::new(1.0, 2.0).unwrap();
        close(gumbel_cdf(1.5, 1.0, 2.0), gumbel.cdf(1.5), 1e-12);

        // Weibull::new(scale, shape); Frechet/Lomax::new(shape, scale).
        let weibull = Weibull::new(1.4, 2.2).unwrap();
        close(
            weibull_cdf_on_support(0.9, 1.4, 2.2),
            weibull.cdf(0.9),
            1e-12,
        );

        let frechet = Frechet::new(3.0, 2.0).unwrap();
        close(
            frechet_cdf_on_support(2.5, 2.0, 3.0),
            frechet.cdf(2.5),
            1e-12,
        );

        let rayleigh = Rayleigh::new(1.4).unwrap();
        close(rayleigh_cdf_on_support(0.9, 1.4), rayleigh.cdf(0.9), 1e-12);

        let pareto = Pareto::new(2.0, 3.0).unwrap();
        close(pareto_cdf_on_support(4.0, 2.0, 3.0), pareto.cdf(4.0), 1e-12);

        let lomax = Lomax::new(3.0, 2.0).unwrap();
        close(lomax_cdf_on_support(1.3, 3.0, 2.0), lomax.cdf(1.3), 1e-12);

        let gamma = Gamma::new(2.5, 1.2).unwrap();
        close(gamma_cdf_on_support(0.8, 2.5, 1.2), gamma.cdf(0.8), 1e-12);

        let chi_squared = ChiSquared::new(5.0).unwrap();
        close(
            chi_squared_cdf_on_support(4.0, 2.5),
            chi_squared.cdf(4.0),
            1e-12,
        );

        let beta = Beta::new(2.0, 5.0).unwrap();
        close(beta_cdf_on_unit(0.35, 2.0, 5.0), beta.cdf(0.35), 1e-12);

        let student = StudentT::new_location_scale(6.0, -0.5, 1.3).unwrap();
        close(students_t_cdf(0.2, -0.5, 1.3, 6.0), student.cdf(0.2), 1e-10);

        let fisher = FisherSnedecor::new(5.0, 10.0).unwrap();
        close(
            fisher_snedecor_cdf_on_support(1.5, 5.0, 10.0),
            fisher.cdf(1.5),
            1e-12,
        );

        // Discrete CDFs via the incomplete gamma / beta.
        let poisson = Poisson::new(3.5).unwrap();
        close(poisson_cdf(6.0, 3.5), poisson.cdf(6), 1e-12);

        let binomial = Binomial::new(12, 0.3).unwrap();
        close(
            binomial_cdf_on_support(4.0, 12.0, 0.3),
            binomial.cdf(4),
            1e-12,
        );

        let negative_binomial = NegativeBinomial::new(5.0, 0.4).unwrap();
        close(
            negative_binomial_cdf(3.0, 5.0, 0.4),
            negative_binomial.cdf(3),
            1e-12,
        );

        // k-th order-statistic CDF via the regularized incomplete beta.
        let os_base = Normal::new(0.0, 1.0).unwrap();
        let os_u = os_base.cdf(0.4);
        let os = OrderStatistic::new(os_base, 7, 3).unwrap();
        close(order_statistic_cdf(os_u, 3.0, 7.0), os.cdf(0.4), 1e-10);
    }

    #[test]
    fn lazy_real_quantile_kernels_match_distribution() {
        let p = 0.37;

        let exponential = Exponential::new(1.7).unwrap();
        close(
            exponential_quantile(p, 1.7),
            exponential.inverse_cdf(p),
            1e-12,
        );

        let logistic = Logistic::new(-0.5, 1.3).unwrap();
        close(
            logistic_quantile(p, -0.5, 1.3),
            logistic.inverse_cdf(p),
            1e-12,
        );

        let cauchy = Cauchy::new(0.5, 1.7).unwrap();
        close(cauchy_quantile(p, 0.5, 1.7), cauchy.inverse_cdf(p), 1e-12);

        // Both sides of the median to exercise the `sign(p − ½)` branch.
        let laplace = Laplace::new(0.5, 1.7).unwrap();
        close(laplace_quantile(p, 0.5, 1.7), laplace.inverse_cdf(p), 1e-12);
        close(
            laplace_quantile(0.8, 0.5, 1.7),
            laplace.inverse_cdf(0.8),
            1e-12,
        );

        // Uniform::new(lo, hi); the kernel takes (p, lo, width = hi − lo).
        let uniform = Uniform::new(-2.0, 3.0).unwrap();
        close(
            uniform_quantile(p, -2.0, 5.0),
            uniform.inverse_cdf(p),
            1e-12,
        );

        // Weibull::new(scale, shape); the kernel takes (p, scale, shape).
        let weibull = Weibull::new(1.4, 2.2).unwrap();
        close(weibull_quantile(p, 1.4, 2.2), weibull.inverse_cdf(p), 1e-12);

        let gumbel = Gumbel::new(1.0, 2.0).unwrap();
        close(gumbel_quantile(p, 1.0, 2.0), gumbel.inverse_cdf(p), 1e-12);

        // Frechet/Lomax::new(shape, scale); the kernels take (p, scale, shape).
        let frechet = Frechet::new(3.0, 2.0).unwrap();
        close(frechet_quantile(p, 2.0, 3.0), frechet.inverse_cdf(p), 1e-12);

        // Pareto::new(minimum, shape); the kernel takes (p, minimum, shape).
        let pareto = Pareto::new(2.0, 3.0).unwrap();
        close(pareto_quantile(p, 2.0, 3.0), pareto.inverse_cdf(p), 1e-12);

        let lomax = Lomax::new(3.0, 2.0).unwrap();
        close(lomax_quantile(p, 2.0, 3.0), lomax.inverse_cdf(p), 1e-12);

        let rayleigh = Rayleigh::new(1.4).unwrap();
        close(rayleigh_quantile(p, 1.4), rayleigh.inverse_cdf(p), 1e-12);

        let normal = Normal::new(1.0, 2.0).unwrap();
        close(normal_quantile(p, 1.0, 2.0), normal.inverse_cdf(p), 1e-12);

        let half_normal = HalfNormal::new(1.4).unwrap();
        close(
            half_normal_quantile(p, 1.4),
            half_normal.inverse_cdf(p),
            1e-12,
        );

        let lognormal = LogNormal::new(0.2, 0.7).unwrap();
        close(
            lognormal_quantile(p, 0.2, 0.7),
            lognormal.inverse_cdf(p),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_entropy_kernels_match_distribution() {
        let normal = Normal::new(1.0, 2.0).unwrap();
        close(normal_entropy(2.0), normal.entropy().unwrap(), 1e-12);

        let lognormal = LogNormal::new(0.2, 0.7).unwrap();
        close(
            lognormal_entropy(0.2, 0.7),
            lognormal.entropy().unwrap(),
            1e-12,
        );

        let exponential = Exponential::new(1.7).unwrap();
        close(
            exponential_entropy(1.7),
            exponential.entropy().unwrap(),
            1e-12,
        );

        let laplace = Laplace::new(0.5, 1.7).unwrap();
        close(laplace_entropy(1.7), laplace.entropy().unwrap(), 1e-12);

        let logistic = Logistic::new(-0.5, 1.3).unwrap();
        close(logistic_entropy(1.3), logistic.entropy().unwrap(), 1e-12);

        let uniform = Uniform::new(-2.0, 3.0).unwrap();
        close(uniform_entropy(5.0), uniform.entropy().unwrap(), 1e-12);

        let discrete_uniform = DiscreteUniform::new(-2, 5).unwrap();
        close(
            discrete_uniform_entropy(7.0),
            discrete_uniform.entropy().unwrap(),
            1e-12,
        );

        let cauchy = Cauchy::new(0.5, 1.7).unwrap();
        close(cauchy_entropy(1.7), cauchy.entropy().unwrap(), 1e-12);

        let gumbel = Gumbel::new(1.0, 2.0).unwrap();
        close(gumbel_entropy(2.0), gumbel.entropy().unwrap(), 1e-12);

        let rayleigh = Rayleigh::new(1.4).unwrap();
        close(rayleigh_entropy(1.4), rayleigh.entropy().unwrap(), 1e-12);

        let weibull = Weibull::new(1.4, 2.2).unwrap();
        close(weibull_entropy(1.4, 2.2), weibull.entropy().unwrap(), 1e-12);

        let pareto = Pareto::new(2.0, 3.0).unwrap();
        close(pareto_entropy(2.0, 3.0), pareto.entropy().unwrap(), 1e-12);

        let triangular = Triangular::new(-1.0, 4.0, 1.0).unwrap();
        close(
            triangular_entropy(-1.0, 4.0),
            triangular.entropy().unwrap(),
            1e-12,
        );

        let half_normal = HalfNormal::new(1.4).unwrap();
        close(
            half_normal_entropy(1.4),
            half_normal.entropy().unwrap(),
            1e-12,
        );

        let gamma = Gamma::new(2.5, 1.2).unwrap();
        close(gamma_entropy(2.5, 1.2), gamma.entropy().unwrap(), 1e-12);

        let chi_squared = ChiSquared::new(5.0).unwrap();
        close(
            chi_squared_entropy(2.5),
            chi_squared.entropy().unwrap(),
            1e-12,
        );

        let inverse_gamma = InverseGamma::new(3.0, 2.0).unwrap();
        close(
            inverse_gamma_entropy(3.0, 2.0),
            inverse_gamma.entropy().unwrap(),
            1e-12,
        );

        let beta = Beta::new(2.0, 5.0).unwrap();
        close(beta_entropy(2.0, 5.0), beta.entropy().unwrap(), 1e-12);

        let student = StudentT::new_location_scale(6.0, -0.5, 1.3).unwrap();
        close(
            students_t_entropy(1.3, 6.0),
            student.entropy().unwrap(),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_von_mises_kernel_matches_distribution() {
        // The circular family: now reachable because `cos` lives on the lazy tier.
        let von_mises = VonMises::new(0.5, 2.0).unwrap();
        let log_normalizer = core::f64::consts::TAU.ln() + ln_i0(2.0);
        close(
            von_mises_log_pdf(0.3, 0.5, 2.0, log_normalizer),
            von_mises.ln_pdf(0.3),
            1e-12,
        );
    }

    #[test]
    fn logit_inverts_sigmoid() {
        for &x in &[-2.3_f64, -0.5, 0.0, 0.7, 3.1] {
            close(logit(sigmoid(x)), x, 1e-12);
        }
    }

    #[test]
    fn lazy_real_hypergeometric_kernel_matches_distribution() {
        let hyper = Hypergeometric::new(50, 20, 10).unwrap();
        let k = 4u64;
        let got = hypergeometric_log_pmf(
            ln_binomial(20, k),
            ln_binomial(50 - 20, 10 - k),
            ln_binomial(50, 10),
        );
        close(got, hyper.ln_pmf(k), 1e-12);
    }

    #[test]
    fn lazy_real_count_vector_kernels_match_explicit_closed_form() {
        // The Multinomial / Dirichlet-multinomial cross-checks against the concrete
        // distributions moved to ito; here the count-vector kernels are checked against
        // their explicit closed forms (the multinomial coefficient is supplied by the
        // bayes-local `ln_factorial`, the Dirichlet-multinomial normalizer by
        // `ln_multivariate_beta`).
        let probabilities = [0.2, 0.5, 0.3];
        let counts = [2u64, 5, 3];
        let n: u64 = counts.iter().sum();
        let ln_coefficient = ln_factorial(n) - counts.iter().map(|&k| ln_factorial(k)).sum::<f64>();
        let counts_real: Vec<f64> = counts.iter().map(|&k| k as f64).collect();

        // Multinomial: ln C + Σ kᵢ·ln pᵢ.
        let log_probabilities: Vec<f64> = probabilities.iter().map(|p| p.ln()).collect();
        let got = multinomial_log_pmf(&counts_real, &log_probabilities, ln_coefficient).unwrap();
        let expected_multinomial = ln_coefficient
            + counts_real
                .iter()
                .zip(&log_probabilities)
                .map(|(&k, &lp)| k * lp)
                .sum::<f64>();
        close(got, expected_multinomial, 1e-12);

        // Dirichlet-multinomial: ln C + ln B(α + counts) − ln B(α).
        let alpha = [1.5, 2.0, 0.7];
        let posterior: Vec<f64> = alpha
            .iter()
            .zip(&counts)
            .map(|(&a, &c)| a + c as f64)
            .collect();
        let got = dirichlet_multinomial_log_pmf(
            ln_coefficient,
            ln_multivariate_beta(&posterior),
            ln_multivariate_beta(&alpha),
        );
        close(
            got,
            ln_coefficient + ln_multivariate_beta(&posterior) - ln_multivariate_beta(&alpha),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_multivariate_gaussian_kernels_match_explicit_closed_form() {
        // The cross-checks against the concrete `MultivariateNormal` / `MultivariateStudentT`
        // moved to ito; here the scalar multivariate kernels are checked against their own
        // closed forms with a diagonal covariance (separable determinant / Mahalanobis term).
        use core::f64::consts::{PI, TAU};
        let mean = [0.5, -1.0];
        let variance = [2.0, 0.75];
        let x = [1.2, -0.3];

        let mahalanobis_sq: f64 = (0..2)
            .map(|i| (x[i] - mean[i]) * (x[i] - mean[i]) / variance[i])
            .sum();
        let ln_det_2pi = 2.0 * TAU.ln() + variance.iter().map(|v| v.ln()).sum::<f64>();
        close(
            multivariate_normal_log_pdf(mahalanobis_sq, ln_det_2pi),
            -0.5 * (mahalanobis_sq + ln_det_2pi),
            1e-12,
        );

        let freedom = 6.0;
        let dimension = 2.0;
        let ln_det_scale = variance.iter().map(|v| v.ln()).sum::<f64>();
        let log_normalizer = ln_gamma((freedom + dimension) / 2.0)
            - ln_gamma(freedom / 2.0)
            - 0.5 * (dimension * (freedom * PI).ln() + ln_det_scale);
        close(
            multivariate_student_t_log_pdf(mahalanobis_sq, freedom, dimension, log_normalizer),
            log_normalizer
                - 0.5 * (freedom + dimension) * (1.0 + mahalanobis_sq / freedom).ln(),
            1e-12,
        );
    }

    #[test]
    fn lazy_real_directional_entropy_kernel_matches_distribution() {
        // The S² directional families (von Mises-Fisher / Watson) moved to ito; their lazy
        // kernel cross-checks live in ito's `tests/bayes_kernels_on_eager.rs`. The circular
        // von Mises stays in bayes, so its entropy kernel is exercised here.
        let vm = VonMises::new(0.3, 2.0).unwrap();
        close(von_mises_entropy(2.0), vm.entropy().unwrap(), 1e-12);
    }
}
