use super::*;

/// Bernoulli log-mass for a probability parameter and a relaxed label in `[0, 1]`:
/// `y log(p) + (1-y) log(1-p)`.
///
/// This is useful for differentiable losses as well as exact labels. Support and
/// range checks are the caller's responsibility.
pub fn bernoulli_log_pmf<T: Real>(y: T, p: T) -> T {
    let one = T::one();
    y.clone() * p.clone().ln() + (one.clone() - y) * (one - p).ln()
}

/// Bernoulli log-mass parameterized by a logit:
/// `y * logit - log(1 + exp(logit))`.
pub fn bernoulli_logit_log_pmf<T: Real>(y: T, logit: T) -> T {
    y * logit.clone() - logit.ln_1p_exp()
}

/// Binomial log-mass for supplied success/failure **counts**:
/// `ln C(n,k) + k log(p) + (n-k) log(1-p)`. The counts are integer-typed
/// ([`Integer`]) — a fractional `k` is unrepresentable — and cross into the real
/// surface via [`to_real`](Integer::to_real).
pub fn binomial_log_pmf<I, T>(successes: I, failures: I, p: T, ln_binomial: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    ln_binomial + successes.to_real() * p.clone().ln() + failures.to_real() * (one - p).ln()
}

/// Categorical log-mass for the selected class probability.
pub fn categorical_log_pmf<T: Real>(selected_probability: T) -> T {
    selected_probability.ln()
}

/// Geometric log-mass for Bayes' trial-count parameterization:
/// `(trials - 1) * log(1-p) + log(p)`. `trials` is an integer count
/// ([`Integer`]).
pub fn geometric_log_pmf<I, T>(trials: I, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    (trials.to_real() - one.clone()) * (one - p.clone()).ln() + p.ln()
}

/// Geometric log-mass for `failures` failures before the first success:
/// `failures * log(1-p) + log(p)`. `failures` is an integer count ([`Integer`]).
pub fn geometric_failures_log_pmf<I, T>(failures: I, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    failures.to_real() * (one - p.clone()).ln() + p.ln()
}

/// Poisson log-mass: `k log(lambda) - lambda - ln(k!)`. The count `k` is integer-typed
/// ([`Integer`]) — there is no Poisson mass at a fractional `k` — and crosses into
/// the real surface via [`to_real`](Integer::to_real); `lambda` and `ln(k!)` are real.
pub fn poisson_log_pmf<I, T>(k: I, lambda: T, ln_factorial_k: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    k.to_real() * lambda.clone().ln() - lambda - ln_factorial_k
}

/// Negative-binomial log-mass in the "failures before r successes"
/// parameterization:
/// `ln C(k+r-1,k) + r log(p) + k log(1-p)`. The `failures` count is integer-typed
/// ([`Integer`]); `successes_until_stop` (`r`) stays real, as it is a continuous
/// shape parameter in the Pólya / Gamma–Poisson generalization.
pub fn negative_binomial_log_pmf<I, T>(
    failures: I,
    successes_until_stop: T,
    p: T,
    ln_coefficient: T,
) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    ln_coefficient + successes_until_stop * p.clone().ln() + failures.to_real() * (one - p).ln()
}

/// Beta-binomial log-mass:
/// `ln C(n,k) + ln B(k+alpha, n-k+beta) - ln B(alpha,beta)`.
pub fn beta_binomial_log_pmf<T: Real>(ln_binomial: T, ln_beta_posterior: T, ln_beta_prior: T) -> T {
    ln_binomial + ln_beta_posterior - ln_beta_prior
}

/// Hypergeometric log-mass for drawing `k` successes:
/// `ln C(K,k) + ln C(N-K, n-k) - ln C(N,n)`.
///
/// The three log-binomials are caller supplied — `ln_binomial_successes` is
/// `ln C(K,k)` over the `K` successes, `ln_binomial_failures` is
/// `ln C(N-K, n-k)` over the `N-K` failures, and `ln_binomial_population` is
/// `ln C(N,n)` over the whole population draw.
pub fn hypergeometric_log_pmf<T: Real>(
    ln_binomial_successes: T,
    ln_binomial_failures: T,
    ln_binomial_population: T,
) -> T {
    ln_binomial_successes + ln_binomial_failures - ln_binomial_population
}

/// Uniform log-density on its support: `-log(width)`.
pub fn uniform_log_pdf_on_support<T: Real>(width: T) -> T {
    -width.ln()
}

/// Discrete-uniform log-mass on its support: `-log(count)`.
pub fn discrete_uniform_log_pmf_on_support<T: Real>(count: T) -> T {
    -count.ln()
}

/// Normal log-density over the lazy real surface.
pub fn normal_log_pdf<T: Real>(x: T, mean: T, std_dev: T) -> T {
    normal_log_prob(x, mean, std_dev)
}

/// Exponential log-density on `x >= 0`: `log(rate) - rate*x`.
pub fn exponential_log_pdf_on_support<T: Real>(x: T, rate: T) -> T {
    rate.clone().ln() - rate * x
}

/// Exponential log-survival on `x >= 0`: `-rate*x`.
pub fn exponential_log_survival_on_support<T: Real>(x: T, rate: T) -> T {
    -(rate * x)
}

/// Gamma log-density on `x > 0`.
///
/// `ln_gamma_shape` is `ln Gamma(shape)`.
pub fn gamma_log_pdf_on_support<T: Real>(x: T, shape: T, rate: T, ln_gamma_shape: T) -> T {
    let one = T::one();
    shape.clone() * rate.clone().ln() - ln_gamma_shape + (shape - one) * x.clone().ln() - rate * x
}

/// Chi-squared log-density on `x > 0`.
///
/// This is the gamma kernel with `shape = freedom / 2` and `rate = 1/2`; callers
/// provide those lazy values and `ln_gamma_shape`.
pub fn chi_squared_log_pdf_on_support<T: Real>(
    x: T,
    half_freedom: T,
    ln_gamma_half_freedom: T,
) -> T {
    gamma_log_pdf_on_support(x, half_freedom, T::half(), ln_gamma_half_freedom)
}

/// Beta log-density on `0 < x < 1`.
///
/// `ln_beta_ab` is `ln B(alpha, beta)`.
pub fn beta_log_pdf_on_unit<T: Real>(x: T, alpha: T, beta: T, ln_beta_ab: T) -> T {
    let one = T::one();
    (alpha - one.clone()) * x.clone().ln() + (beta - one.clone()) * (one - x).ln() - ln_beta_ab
}

/// Log-normal log-density on `x > 0`.
pub fn lognormal_log_pdf_on_support<T: Real>(x: T, mean_log: T, std_log: T) -> T {
    let log_x = x.clone().ln();
    let z = (log_x.clone() - mean_log) / std_log.clone();
    -(T::half() * z.clone() * z) - log_x - std_log.ln() - T::half_ln_tau()
}

/// Logistic-distribution log-density on the real line.
pub fn logistic_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    let log_denom = (-z.clone()).ln_1p_exp();
    -scale.ln() - z - log_denom.clone() - log_denom
}

/// Laplace log-density on the real line.
pub fn laplace_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    T::half().ln() - scale.clone().ln() - (x - location).abs() / scale
}

/// Cauchy log-density on the real line.
pub fn cauchy_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -T::ln_pi() - scale.ln() - (T::one() + z.clone() * z).ln()
}

/// Student-t log-density for a location/scale t distribution.
///
/// `log_normalizer` is
/// `0.5*log(freedom*pi) + ln_gamma(freedom/2) - ln_gamma((freedom+1)/2)`.
pub fn student_t_log_pdf<T: Real>(x: T, location: T, scale: T, freedom: T, log_normalizer: T) -> T {
    let z = (x - location) / scale.clone();
    let one = T::one();
    let inner = one.clone() + (z.clone() * z) / freedom.clone();
    -scale.ln() - log_normalizer - T::half() * (freedom + one) * inner.ln()
}

/// Fisher-Snedecor F log-density on `x > 0`.
///
/// `half_d1 = d1/2`, `half_d2 = d2/2`, and `ln_beta_half_degrees` is
/// `ln B(d1/2, d2/2)`.
pub fn fisher_snedecor_log_pdf_on_support<T: Real>(
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
pub fn wald_log_pdf_on_support<T: Real>(x: T, mean: T, shape: T) -> T {
    let delta = x.clone() - mean.clone();
    T::half() * (shape.clone().ln() - T::ln_tau() - T::scalar(3.0) * x.clone().ln())
        - shape * delta.clone() * delta / (T::two() * mean.clone() * mean * x)
}

/// Pareto log-density on `x >= minimum`.
pub fn pareto_log_pdf_on_support<T: Real>(x: T, minimum: T, shape: T) -> T {
    let one = T::one();
    shape.clone().ln() + shape.clone() * minimum.ln() - (shape + one) * x.ln()
}

/// Lomax log-density on `x >= 0`.
pub fn lomax_log_pdf_on_support<T: Real>(x: T, shape: T, scale: T) -> T {
    let one = T::one();
    shape.clone().ln() - scale.clone().ln() - (shape + one.clone()) * (one + x / scale).ln()
}

/// Inverse-gamma log-density on `x > 0`.
///
/// `ln_gamma_shape` is `ln Gamma(shape)`.
pub fn inverse_gamma_log_pdf_on_support<T: Real>(x: T, shape: T, scale: T, ln_gamma_shape: T) -> T {
    let one = T::one();
    shape.clone() * scale.clone().ln() - ln_gamma_shape - (shape + one) * x.clone().ln() - scale / x
}

/// Rayleigh log-density on `x >= 0`.
pub fn rayleigh_log_pdf_on_support<T: Real>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    x.clone().ln() - scale.clone().ln() - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Half-normal log-density on `x >= 0`.
pub fn half_normal_log_pdf_on_support<T: Real>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    let log_sqrt_2_over_pi = T::scalar(0.5 * (2.0 / core::f64::consts::PI).ln());
    log_sqrt_2_over_pi - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Weibull log-density on `x >= 0`.
pub fn weibull_log_pdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    shape.clone().ln() - scale.ln() + (shape.clone() - T::one()) * log_y.clone()
        - (shape * log_y).exp()
}

/// Frechet log-density on `x > 0`.
pub fn frechet_log_pdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    let y_neg_shape = (-(shape.clone()) * log_y.clone()).exp();
    shape.clone().ln() - scale.ln() - (shape + T::one()) * log_y - y_neg_shape
}

/// Gumbel log-density on the real line.
pub fn gumbel_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -scale.ln() - z.clone() - (-z).exp()
}

/// Von Mises (circular normal) log-density on `[−π, π)`:
/// `κ·cos(x − µ) − log_normalizer`.
///
/// `log_normalizer` is `ln(2π·I₀(κ))`, the circular normalizer the caller
/// precomputes from the modified Bessel `I₀`. The kernel itself is pure analysis:
/// it leans on the lazy `cos`, which the `Real` surface now carries, so the
/// circular family is reachable on every backend.
pub fn von_mises_log_pdf<T: Real>(x: T, mean: T, kappa: T, log_normalizer: T) -> T {
    kappa * (x - mean).cos() - log_normalizer
}
