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

fn ln_factorial(n: u64) -> f64 {
    n.ln_factorial()
}

fn ln_binomial(n: u64, k: u64) -> f64 {
    n.ln_binomial(&k)
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol + tol * b.abs(),
        "observed {a}, expected {b}"
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

    // The Normal-Inverse-Gamma / Normal-Gamma marginal-likelihood cross-check (against
    // the concrete conjugate models) moved to ito.
}

#[test]
fn lazy_real_reparameterization_transforms_match_constraint() {
    use bayes::Constraint;

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
