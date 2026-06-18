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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{
        Beta, Binomial, Cauchy, Continuous, Dirichlet, Discrete, DiscreteUniform, Exponential,
        Gamma, Geometric, Laplace, LogNormal, Logistic, MultivariateContinuous, Normal, Pareto,
        Poisson, Rayleigh, StudentT, Uniform, Weibull,
    };
    use crate::function::beta::ln_beta;
    use crate::function::factorial::{ln_binomial, ln_factorial};
    use crate::function::gamma::ln_gamma;

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

        let dirichlet = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        let x = [0.2, 0.3, 0.5];
        let got =
            dirichlet_log_pdf_on_simplex(&x, dirichlet.alpha(), dirichlet.ln_multivariate_beta())
                .unwrap();
        close(got, dirichlet.ln_pdf(&x), 1e-12);
    }
}
