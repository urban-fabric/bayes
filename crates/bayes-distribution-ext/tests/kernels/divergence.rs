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
use bayes_distribution::{
    Bernoulli, Beta, Binomial, DiscreteUniform, Exponential, Gamma, NegativeBinomial, Normal,
    Poisson, Uniform, Wald,
};
use bayes_distribution_ext::ContinuousCdf;
use bayes_divergence::HellingerDistance;

#[test]
fn lazy_real_stable_log_tails_match_distribution() {
    let exponential = bayes_distribution_ext::Exponential::new(1.7).unwrap();
    assert_close_scaled!(
        exponential_ln_cdf_on_support(0.4, 1.7),
        exponential.ln_cdf(0.4),
        1e-12,
    );

    let logistic = bayes_distribution_ext::Logistic::new(-0.5, 1.3).unwrap();
    assert_close_scaled!(logistic_ln_cdf(0.2, -0.5, 1.3), logistic.ln_cdf(0.2), 1e-12);
    assert_close_scaled!(logistic_ln_sf(0.2, -0.5, 1.3), logistic.ln_sf(0.2), 1e-12);

    // Gumbel has no ln_cdf method; check against ln(cdf).
    let gumbel = bayes_distribution_ext::Gumbel::new(1.0, 2.0).unwrap();
    assert_close_scaled!(gumbel_ln_cdf(1.5, 1.0, 2.0), gumbel.cdf(1.5).ln(), 1e-12);
}

#[test]
fn lazy_real_hellinger_kernels_match_distribution() {
    let normal = (Normal::new(0.0, 1.0), Normal::new(1.0, 2.0));
    assert_close_scaled!(
        hellinger_sq_normal(0.0, 1.0, 1.0, 2.0),
        normal.0.hellinger_sq(&normal.1),
        1e-12,
    );

    let exponential = (Exponential::new(1.5), Exponential::new(0.7));
    assert_close_scaled!(
        hellinger_sq_exponential(1.5, 0.7),
        exponential.0.hellinger_sq(&exponential.1),
        1e-12,
    );

    let poisson = (Poisson::new(2.5), Poisson::new(4.0));
    assert_close_scaled!(
        hellinger_sq_poisson(2.5, 4.0),
        poisson.0.hellinger_sq(&poisson.1),
        1e-12,
    );

    let bernoulli = (Bernoulli::new(0.3), Bernoulli::new(0.7));
    assert_close_scaled!(
        hellinger_sq_bernoulli(0.3, 0.7),
        bernoulli.0.hellinger_sq(&bernoulli.1),
        1e-12,
    );

    let gamma = (Gamma::new(2.5, 1.2), Gamma::new(3.0, 0.8));
    assert_close_scaled!(
        hellinger_sq_gamma(2.5, 1.2, 3.0, 0.8),
        gamma.0.hellinger_sq(&gamma.1),
        1e-12,
    );

    let beta = (Beta::new(2.0, 5.0), Beta::new(3.0, 4.0));
    assert_close_scaled!(
        hellinger_sq_beta(2.0, 5.0, 3.0, 4.0),
        beta.0.hellinger_sq(&beta.1),
        1e-12,
    );

    let wald = (Wald::new(1.0, 2.0), Wald::new(1.5, 3.0));
    assert_close_scaled!(
        hellinger_sq_wald(1.0, 2.0, 1.5, 3.0),
        wald.0.hellinger_sq(&wald.1),
        1e-12,
    );

    // Binomial with a matched trial count n = 12 (BC_Bernoulli^n).
    let binomial = (Binomial::new(12_u64, 0.3), Binomial::new(12_u64, 0.5));
    assert_close_scaled!(
        hellinger_sq_binomial(12.0, 0.3, 0.5),
        binomial.0.hellinger_sq(&binomial.1),
        1e-12,
    );

    // Negative-binomial with matched successes r = 5.
    let negbin = (
        NegativeBinomial::new(5.0, 0.4),
        NegativeBinomial::new(5.0, 0.6),
    );
    assert_close_scaled!(
        hellinger_sq_negative_binomial(5.0, 0.4, 0.6),
        negbin.0.hellinger_sq(&negbin.1),
        1e-12,
    );

    // Uniform / discrete-uniform squared Hellinger (general overlap).
    let uniform = (Uniform::new(-1.0, 2.0), Uniform::new(0.0, 4.0));
    assert_close_scaled!(
        hellinger_sq_uniform(-1.0, 2.0, 0.0, 4.0),
        uniform.0.hellinger_sq(&uniform.1),
        1e-12,
    );
    let du = (
        DiscreteUniform::new(0_i64, 3),
        DiscreteUniform::new(2_i64, 6),
    );
    assert_close_scaled!(
        hellinger_sq_discrete_uniform(0.0, 3.0, 2.0, 6.0),
        du.0.hellinger_sq(&du.1),
        1e-12,
    );
}
