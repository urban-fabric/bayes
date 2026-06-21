//! Cross-checks: `bayes-kernels` closed-form kernels vs the concrete distribution
//! families. Relocated from `bayes-kernels` unit tests — the lazy kernels sit *below*
//! the families in the crate DAG, so this cross-check runs at the facade level
//! (`bayes`), where both the kernels and every family are visible.
#![allow(
    unused_imports,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

// The lazy kernels are generic over `euclid::numeric::LazyReal`; the cross-checks
// instantiate them at `f64` and call LazyReal methods (e.g. `ln_1p_exp`) directly.
#[allow(unused_imports)]
use euclid::numeric::{LazyInteger, LazyNatural, LazyReal};

use bayes::kernels::*;
// The scalar/lazy kernels are exercised end-to-end here against bayes' own concrete
// distributions (euclid-only families). The multivariate-matrix, conjugate-matrix,
// and S² directional families moved to the sibling crate `ito`; the cross-checks that
// these same kernels agree with `ito::MultivariateNormal` / `ito::VonMisesFisher` /
// `ito::Dirichlet` (etc.) live in ito's integration test
// `tests/bayes_kernels_on_eager.rs`.
use bayes::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomialModel, Binomial, Categorical, Cauchy, ChiSquared,
    ChiSquaredDivergence, Continuous, ContinuousCdf, Discrete, DiscreteCdf, DiscreteUniform,
    Exponential, FisherSnedecor, Frechet, Gamma, GammaExponential, GammaPoisson, Geometric, Gumbel,
    HalfNormal, HellingerDistance, Hypergeometric, IntegerDiscrete, InverseGamma, KlDivergence,
    Laplace, LogNormal, Logistic, Lomax, Moments, NegativeBinomial, Normal, OrderStatistic, Pareto,
    Poisson, Rayleigh, RenyiDivergence, StudentT, TotalVariation, Triangular, Uniform, VonMises,
    Wald, Weibull,
};

fn ln_beta(a: f64, b: f64) -> f64 {
    a.ln_beta(b)
}

fn ln_factorial(n: u64) -> f64 {
    n.ln_factorial()
}

fn ln_binomial(n: u64, k: u64) -> f64 {
    n.ln_binomial(&k)
}

fn ln_gamma(x: f64) -> f64 {
    x.ln_gamma()
}

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
    let log_normalizer = 0.5 * (6.0 * core::f64::consts::PI).ln() + ln_gamma(3.0) - ln_gamma(3.5);
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
        binomial_log_pmf(4_i64, 8_i64, 0.3, ln_binomial(12, 4)),
        binomial.ln_pmf(4_u64),
        1e-12,
    );

    let geometric = Geometric::new(0.25).unwrap();
    close(
        geometric_log_pmf(5_i64, 0.25),
        geometric.ln_pmf(5_u64),
        1e-12,
    );

    let poisson = Poisson::new(3.5).unwrap();
    close(
        poisson_log_pmf(6_i64, 3.5, ln_factorial(6)),
        poisson.ln_pmf(6_u64),
        1e-12,
    );

    let discrete_uniform = DiscreteUniform::new(-2, 5).unwrap();
    close(
        discrete_uniform_log_pmf_on_support(7.0),
        discrete_uniform.ln_pmf(3_i64),
        1e-12,
    );

    close(
        bernoulli_logit_log_pmf(1.0, 2.0),
        -2.0_f64.ln_1p_exp() + 2.0,
        1e-12,
    );
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
