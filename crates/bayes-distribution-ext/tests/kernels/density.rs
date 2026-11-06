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
use bayes_core_ext::assert_close_scaled;

// The lazy kernels are generic over `euclid_numeric::Real`; the cross-checks
// instantiate them at `f64` and call Real methods (e.g. `ln_1p_exp`) directly.
#[allow(unused_imports)]
use euclid_numeric::{Integer, Natural, Real};

use bayes_kernels::*;
// The scalar/lazy kernels are exercised end-to-end here against bayes' own concrete
// distributions (euclid-only families). The multivariate-matrix, conjugate-matrix,
// and S² directional families moved to the sibling crate `ito`; the cross-checks that
// these same kernels agree with `ito::MultivariateNormal` / `ito::VonMisesFisher` /
// `ito::Dirichlet` (etc.) live in ito's integration test
// `tests/bayes_kernels_on_eager.rs`.
use bayes_distribution_ext::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomialModel, Binomial, Categorical, Cauchy, ChiSquared,
    Continuous, ContinuousCdf, Discrete, DiscreteCdf, DiscreteUniform, Exponential, FisherSnedecor,
    Frechet, Gamma, GammaExponential, GammaPoisson, Geometric, Gumbel, HalfNormal, Hypergeometric,
    IntegerDiscrete, InverseGamma, Laplace, LogNormal, Logistic, Lomax, Moments, NegativeBinomial,
    Normal, OrderStatistic, Pareto, Poisson, Rayleigh, StudentT, Triangular, Uniform, VonMises,
    Wald, Weibull,
};
use bayes_divergence::{
    ChiSquaredDivergence, HellingerDistance, KlDivergence, RenyiDivergence, TotalVariation,
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

#[test]
fn lazy_real_continuous_kernels_match_distribution_log_densities() {
    let normal = Normal::new(1.0, 2.0).unwrap();
    assert_close_scaled!(normal_log_pdf(0.25, 1.0, 2.0), normal.ln_pdf(0.25), 1e-12);

    let uniform = Uniform::new(-2.0, 3.0).unwrap();
    assert_close_scaled!(uniform_log_pdf_on_support(5.0), uniform.ln_pdf(0.25), 1e-12);

    let exponential = Exponential::new(1.7).unwrap();
    assert_close_scaled!(
        exponential_log_pdf_on_support(0.4, 1.7),
        exponential.ln_pdf(0.4),
        1e-12,
    );

    let gamma = Gamma::new(2.5, 1.2).unwrap();
    assert_close_scaled!(
        gamma_log_pdf_on_support(0.8, 2.5, 1.2, ln_gamma(2.5)),
        gamma.ln_pdf(0.8),
        1e-12,
    );

    let beta = Beta::new(2.0, 5.0).unwrap();
    assert_close_scaled!(
        beta_log_pdf_on_unit(0.35, 2.0, 5.0, ln_beta(2.0, 5.0)),
        beta.ln_pdf(0.35),
        1e-12,
    );

    let lognormal = LogNormal::new(0.2, 0.7).unwrap();
    assert_close_scaled!(
        lognormal_log_pdf_on_support(1.4, 0.2, 0.7),
        lognormal.ln_pdf(1.4),
        1e-12,
    );

    let logistic = Logistic::new(-0.5, 1.3).unwrap();
    assert_close_scaled!(
        logistic_log_pdf(0.2, -0.5, 1.3),
        logistic.ln_pdf(0.2),
        1e-12,
    );

    let laplace = Laplace::new(0.5, 1.7).unwrap();
    assert_close_scaled!(laplace_log_pdf(-0.2, 0.5, 1.7), laplace.ln_pdf(-0.2), 1e-12);

    let cauchy = Cauchy::new(0.5, 1.7).unwrap();
    assert_close_scaled!(cauchy_log_pdf(-0.2, 0.5, 1.7), cauchy.ln_pdf(-0.2), 1e-12);

    let student = StudentT::new(6.0).unwrap();
    let log_normalizer = 0.5 * (6.0 * core::f64::consts::PI).ln() + ln_gamma(3.0) - ln_gamma(3.5);
    assert_close_scaled!(
        student_t_log_pdf(0.4, 0.0, 1.0, 6.0, log_normalizer),
        student.ln_pdf(0.4),
        1e-12,
    );

    let pareto = Pareto::new(2.0, 3.0).unwrap();
    assert_close_scaled!(
        pareto_log_pdf_on_support(4.0, 2.0, 3.0),
        pareto.ln_pdf(4.0),
        1e-12,
    );

    let rayleigh = Rayleigh::new(1.4).unwrap();
    assert_close_scaled!(
        rayleigh_log_pdf_on_support(0.9, 1.4),
        rayleigh.ln_pdf(0.9),
        1e-12,
    );

    let weibull = Weibull::new(1.4, 2.2).unwrap();
    assert_close_scaled!(
        weibull_log_pdf_on_support(0.9, 1.4, 2.2),
        weibull.ln_pdf(0.9),
        1e-12,
    );
}

#[test]
fn lazy_real_discrete_kernels_match_distribution_log_masses() {
    let binomial = Binomial::new(12, 0.3).unwrap();
    assert_close_scaled!(
        binomial_log_pmf(4_i64, 8_i64, 0.3, ln_binomial(12, 4)),
        binomial.ln_pmf(4_u64),
        1e-12,
    );

    let geometric = Geometric::new(0.25).unwrap();
    assert_close_scaled!(
        geometric_log_pmf(5_i64, 0.25),
        geometric.ln_pmf(5_u64),
        1e-12,
    );

    let poisson = Poisson::new(3.5).unwrap();
    assert_close_scaled!(
        poisson_log_pmf(6_i64, 3.5, ln_factorial(6)),
        poisson.ln_pmf(6_u64),
        1e-12,
    );

    let discrete_uniform = DiscreteUniform::new(-2, 5).unwrap();
    assert_close_scaled!(
        discrete_uniform_log_pmf_on_support(7.0),
        discrete_uniform.ln_pmf(3_i64),
        1e-12,
    );

    assert_close_scaled!(
        bernoulli_logit_log_pmf(1.0, 2.0),
        -2.0_f64.ln_1p_exp() + 2.0,
        1e-12,
    );
}

#[test]
fn lazy_real_moment_kernels_match_distribution() {
    let lognormal = LogNormal::new(0.2, 0.7).unwrap();
    assert_close_scaled!(lognormal_mean(0.2, 0.7), lognormal.mean(), 1e-12);
    assert_close_scaled!(lognormal_variance(0.2, 0.7), lognormal.variance(), 1e-12);
    assert_close_scaled!(lognormal_skewness(0.7), lognormal.skewness(), 1e-12);
    assert_close_scaled!(lognormal_excess_kurtosis(0.7), lognormal.kurtosis(), 1e-12);

    let gamma = Gamma::new(2.5, 1.2).unwrap();
    assert_close_scaled!(gamma_variance(2.5, 1.2), gamma.variance(), 1e-12);
    assert_close_scaled!(gamma_skewness(2.5), gamma.skewness(), 1e-12);
    assert_close_scaled!(gamma_excess_kurtosis(2.5), gamma.kurtosis(), 1e-12);

    let beta = Beta::new(2.0, 5.0).unwrap();
    assert_close_scaled!(beta_variance(2.0, 5.0), beta.variance(), 1e-12);
    assert_close_scaled!(beta_skewness(2.0, 5.0), beta.skewness(), 1e-12);
    assert_close_scaled!(beta_excess_kurtosis(2.0, 5.0), beta.kurtosis(), 1e-12);

    let chi = ChiSquared::new(5.0).unwrap();
    assert_close_scaled!(chi_squared_skewness(5.0), chi.skewness(), 1e-12);
    assert_close_scaled!(chi_squared_excess_kurtosis(5.0), chi.kurtosis(), 1e-12);

    let wald = Wald::new(1.0, 2.0).unwrap();
    assert_close_scaled!(wald_skewness(1.0, 2.0), wald.skewness(), 1e-12);
    assert_close_scaled!(wald_excess_kurtosis(1.0, 2.0), wald.kurtosis(), 1e-12);

    let rayleigh = Rayleigh::new(1.4).unwrap();
    assert_close_scaled!(rayleigh_skewness(), rayleigh.skewness(), 1e-12);
    assert_close_scaled!(rayleigh_excess_kurtosis(), rayleigh.kurtosis(), 1e-12);

    let half_normal = HalfNormal::new(2.0).unwrap();
    assert_close_scaled!(half_normal_skewness(), half_normal.skewness(), 1e-12);
    assert_close_scaled!(half_normal_excess_kurtosis(), half_normal.kurtosis(), 1e-12);

    let gumbel = Gumbel::new(0.5, 1.3).unwrap();
    assert_close_scaled!(gumbel_excess_kurtosis(), gumbel.kurtosis(), 1e-12);

    let poisson = Poisson::new(3.5).unwrap();
    assert_close_scaled!(poisson_skewness(3.5), poisson.skewness(), 1e-12);
    assert_close_scaled!(poisson_excess_kurtosis(3.5), poisson.kurtosis(), 1e-12);

    let binomial = Binomial::new(12, 0.3).unwrap();
    assert_close_scaled!(binomial_skewness(12.0, 0.3), binomial.skewness(), 1e-12);

    let geometric = Geometric::new(0.25).unwrap();
    assert_close_scaled!(geometric_skewness(0.25), geometric.skewness(), 1e-12);
    assert_close_scaled!(geometric_excess_kurtosis(0.25), geometric.kurtosis(), 1e-12);

    let negbin = NegativeBinomial::new(5.0, 0.4).unwrap();
    assert_close_scaled!(
        negative_binomial_skewness(5.0, 0.4),
        negbin.skewness(),
        1e-12,
    );
    assert_close_scaled!(
        negative_binomial_excess_kurtosis(5.0, 0.4),
        negbin.kurtosis(),
        1e-12,
    );

    let bernoulli = Bernoulli::new(0.3).unwrap();
    assert_close_scaled!(bernoulli_skewness(0.3), bernoulli.skewness(), 1e-12);
    assert_close_scaled!(bernoulli_excess_kurtosis(0.3), bernoulli.kurtosis(), 1e-12);
}

/// The newly-filled moment formulas, checked against independently-computed
/// textbook reference values (not just self-consistency with the kernels).
#[test]
fn filled_moment_formulas_match_known_values() {
    // LogNormal(σ=1): skew = (e+2)√(e−1) ≈ 6.1849, exkurt = e⁴+2e³+3e²−6 ≈ 110.936.
    let ln1 = LogNormal::new(0.0, 1.0).unwrap();
    assert_close_scaled!(ln1.skewness(), 6.184_877_138_632_554, 1e-9);
    assert_close_scaled!(ln1.kurtosis(), 110.936_392_176_311_53, 1e-9);

    // Beta(1,1) = Uniform(0,1): excess kurtosis = −1.2.
    assert_close_scaled!(Beta::new(1.0, 1.0).unwrap().kurtosis(), -1.2, 1e-12);
    // Beta(2,2): symmetric, excess kurtosis = −6/(2·2+3) = −6/7.
    assert_close_scaled!(Beta::new(2.0, 2.0).unwrap().kurtosis(), -6.0 / 7.0, 1e-12);

    // Bernoulli(0.5): excess kurtosis = (1−6·0.25)/0.25 = −2.
    assert_close_scaled!(Bernoulli::new(0.5).unwrap().kurtosis(), -2.0, 1e-12);

    // Geometric(0.5) on {1,2,…}: excess kurtosis = 6 + 0.25/0.5 = 6.5.
    assert_close_scaled!(Geometric::new(0.5).unwrap().kurtosis(), 6.5, 1e-12);

    // NegativeBinomial(r=1) reduces to the geometric value.
    assert_close_scaled!(
        NegativeBinomial::new(1.0, 0.5).unwrap().kurtosis(),
        6.5,
        1e-12,
    );

    // Rayleigh: scale-free constants ≈ 0.631 (skew), ≈ 0.245 (exkurt).
    let ray = Rayleigh::new(3.0).unwrap();
    assert_close_scaled!(ray.skewness(), 0.631_110_657_818_937_8, 1e-9);
    assert_close_scaled!(ray.kurtosis(), 0.245_089_300_687_638, 1e-9);

    // Half-normal: scale-free constants ≈ 0.995 (skew), ≈ 0.869 (exkurt).
    let hn = HalfNormal::new(5.0).unwrap();
    assert_close_scaled!(hn.skewness(), 0.995_271_746_431_156_5, 1e-9);
    assert_close_scaled!(hn.kurtosis(), 0.869_177_303_605_973_6, 1e-9);

    // Gumbel: excess kurtosis = 12/5 = 2.4, independent of location/scale.
    assert_close_scaled!(Gumbel::new(-1.0, 2.5).unwrap().kurtosis(), 2.4, 1e-12);
}
