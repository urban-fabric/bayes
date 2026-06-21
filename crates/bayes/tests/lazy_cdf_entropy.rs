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
    HalfNormal, HellingerDistance, Hypergeometric, InverseGamma, KlDivergence, Laplace, LogNormal,
    Logistic, Lomax, Moments, NegativeBinomial, Normal, OrderStatistic, Pareto, Poisson, Rayleigh,
    RenyiDivergence, StudentT, TotalVariation, Triangular, Uniform, VonMises, Wald, Weibull,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol + tol * b.abs(),
        "observed {a}, expected {b}"
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
    close(poisson_cdf(6_i64, 3.5), poisson.cdf(6), 1e-12);

    let binomial = Binomial::new(12, 0.3).unwrap();
    close(
        binomial_cdf_on_support(4_i64, 12_i64, 0.3),
        binomial.cdf(4),
        1e-12,
    );

    let negative_binomial = NegativeBinomial::new(5.0, 0.4).unwrap();
    close(
        negative_binomial_cdf(3_i64, 5.0, 0.4),
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
