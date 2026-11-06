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
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Moments, NegativeBinomial, Normal,
    OrderStatistic, Pareto, Poisson, Rayleigh, StudentT, Triangular, Uniform, VonMises, Wald,
    Weibull,
};
use bayes_divergence::{
    ChiSquaredDivergence, HellingerDistance, KlDivergence, RenyiDivergence, TotalVariation,
};

fn ln_i0(x: f64) -> f64 {
    x.ln_bessel_i0()
}

fn ln_binomial(n: u64, k: u64) -> f64 {
    n.ln_binomial(&k)
}

#[test]
fn lazy_real_von_mises_kernel_matches_distribution() {
    // The circular family: now reachable because `cos` lives on the lazy tier.
    let von_mises = VonMises::new(0.5, 2.0).unwrap();
    let log_normalizer = core::f64::consts::TAU.ln() + ln_i0(2.0);
    assert_close_scaled!(
        von_mises_log_pdf(0.3, 0.5, 2.0, log_normalizer),
        von_mises.ln_pdf(0.3),
        1e-12,
    );
}

#[test]
fn logit_inverts_sigmoid() {
    for &x in &[-2.3_f64, -0.5, 0.0, 0.7, 3.1] {
        assert_close_scaled!(logit(sigmoid(x)), x, 1e-12);
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
    assert_close_scaled!(got, hyper.ln_pmf(k), 1e-12);
}

#[test]
fn lazy_real_directional_entropy_kernel_matches_distribution() {
    // The S² directional families (von Mises-Fisher / Watson) moved to ito; their lazy
    // kernel cross-checks live in ito's `tests/bayes_kernels_on_eager.rs`. The circular
    // von Mises stays in bayes, so its entropy kernel is exercised here.
    let vm = VonMises::new(0.3, 2.0).unwrap();
    assert_close_scaled!(von_mises_entropy(2.0), vm.entropy(), 1e-12);
}

// Relocated from `bayes-kernels` `primitives` unit tests: the generic normal
// log-probability primitive equals the owning distribution's closed-form `ln_pdf`.
#[test]
fn generic_normal_log_prob_matches_scalar_distribution() {
    for (x, mu, sigma) in [(-2.0, 0.0, 1.0), (0.5, 1.0, 2.0), (3.0, -1.0, 0.7)] {
        let reference = Normal::new(mu, sigma)
            .expect("valid normal parameters")
            .ln_pdf(x);
        let generic = normal_log_prob::<f64>(x, mu, sigma);
        assert!(
            (reference - generic).abs() <= 1e-12 + 1e-12 * reference.abs(),
            "reference {reference} vs generic {generic} for ({x}, {mu}, {sigma})"
        );
    }
}
