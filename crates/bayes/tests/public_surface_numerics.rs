//! Closed-form Bayesian layer and numerical-stability surface, exercised through
//! the public crate API.
#![allow(
    unused_imports,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

// The multivariate-matrix, matrix-conjugate, and S² directional families (and the dense-
// matrix sample-statistic admissibility predicates) moved to the sibling crate `ito`; their
// public-surface tests moved with them (see ito's test suite). What remains here is bayes'
// euclid-only surface: scalar families, scalar conjugate models, divergences, empirical
// distributions, t-digest, and the lazy kernels.
use bayes::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Conjugate, Constraint,
    Continuous, ContinuousCdf, ContinuousCdfExt, ContinuousDensityKernel,
    ContinuousDistributionExt, ContinuousFn, Discrete, DiscreteDistributionExt, DiscreteFn,
    DiscreteMassKernel, DiscreteUniform, Ecdf, Exponential, ExponentialFamily, Gamma,
    GammaExponential, GammaPoisson, IntegerDiscrete, LazyReal, MeanVar, Mixture, Moments, Normal,
    OrderStatistic, Poisson, SufficientStatistic, Uniform, histogram,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn normal_log_cdf_is_accurate_in_the_far_tail_where_naive_underflows() {
    let n = Normal::new(0.0, 1.0).unwrap();
    assert_eq!(n.cdf(-40.0).ln(), f64::NEG_INFINITY);
    let lc: f64 = n.ln_cdf(-40.0);
    assert!(lc.is_finite());
    close(lc, -804.6084, 1e-2);
    close(n.ln_sf(40.0), n.ln_cdf(-40.0), 1e-12);
    close(n.ln_cdf(-1.5).exp(), n.cdf(-1.5), 1e-12);
}

#[test]
fn exponential_log_survival_is_exact() {
    let e = Exponential::new(2.0).unwrap();
    close(e.ln_sf(500.0), -1000.0, 1e-12);
    assert_eq!(e.sf(500.0).ln(), f64::NEG_INFINITY);
}

#[test]
fn lazy_numeric_kernels_are_public() {
    let n = Normal::new(1.0, 2.0).unwrap();
    close(
        bayes::kernels::normal_log_pdf(0.25_f64, 1.0, 2.0),
        n.ln_pdf(0.25),
        1e-12,
    );
    close(
        bayes::kernels::bernoulli_logit_log_pmf(1.0_f64, 2.0),
        2.0 - (1.0 + 2.0_f64.exp()).ln(),
        1e-12,
    );
    close(n.ln_pdf_kernel(0.25_f64), n.ln_pdf(0.25), 1e-12);

    let p = Poisson::new(3.5).unwrap();
    close(p.ln_pmf_kernel(4_u64), p.ln_pmf(4_u64), 1e-12);

    let normal_f32: f32 = n.pdf(0.25_f32);
    close(normal_f32 as f64, n.pdf(0.25), 1e-6);

    let poisson_u32: f32 = p.pmf(4_u32);
    close(poisson_u32 as f64, p.pmf(4_u64), 1e-6);

    let discrete_uniform = DiscreteUniform::new(-2, 5).unwrap();
    close(discrete_uniform.pmf(3_i64), 1.0 / 7.0, 1e-12);
}

#[test]
fn inverse_survival_round_trips() {
    let g = Gamma::new(2.5, 1.3).unwrap();
    for &p in &[0.01, 0.1, 0.5, 0.9] {
        close(g.isf(p), g.inverse_cdf(1.0 - p), 1e-9);
        close(g.sf(g.isf(p)), p, 1e-6);
    }
}

#[test]
fn mixture_log_density_is_stable_far_from_all_components() {
    let m = Mixture::new(
        vec![0.5, 0.5],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(50.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    close(m.ln_pdf(1.0), m.pdf(1.0).ln(), 1e-12);
    let far = m.ln_pdf(-40.0);
    assert!(far.is_finite());
    assert_eq!(m.pdf(-40.0).ln(), f64::NEG_INFINITY);

    let responsibilities = m.pdf_responsibilities(1.0);
    close(responsibilities.iter().sum(), 1.0, 1e-12);
    assert!(responsibilities[0] > responsibilities[1]);
}

#[test]
fn beta_bernoulli_conjugate_flow() {
    let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap());
    let data = [true, true, false, true, true, false, true];
    let post = model.posterior(&data);
    close(post.alpha(), 6.0, 1e-12);
    close(post.beta(), 3.0, 1e-12);
    let pred: Bernoulli = model.posterior_predictive(&data);
    close(pred.p(), 6.0 / 9.0, 1e-12);
    close(model.ln_predictive(true).exp(), 0.5, 1e-12);
    close(
        model.ln_posterior_predictive(&data, true).exp(),
        pred.p(),
        1e-12,
    );
    close(
        model.ln_marginal_likelihood(&data),
        6.0_f64.ln_beta(3.0) - 1.0_f64.ln_beta(1.0),
        1e-12,
    );
    close(
        model.ln_sequential_predictive(&data),
        model.ln_marginal_likelihood(&data),
        1e-12,
    );
    assert!(
        model
            .ln_posterior_predictive_likelihood(&data, &[true, false])
            .is_finite()
    );
}

#[test]
fn beta_binomial_and_gamma_poisson_predictives() {
    let bb_model = BetaBinomialModel::new(Beta::new(2.0, 2.0).unwrap(), 5);
    let pred: BetaBinomial = bb_model.posterior_predictive(&[3, 4, 2]);
    let mass: f64 = (0..=5u64).map(|k| pred.pmf(k)).sum();
    close(mass, 1.0, 1e-12);
    assert_eq!(bb_model.trials(), 5);
    close(
        bb_model.ln_posterior_predictive(&[3, 4, 2], 3).exp(),
        pred.pmf(3_u64),
        1e-12,
    );
    assert!(bb_model.ln_marginal_likelihood(&[3, 4, 2]).is_finite());
    close(
        bb_model.ln_sequential_predictive(&[3, 4, 2]),
        bb_model.ln_marginal_likelihood(&[3, 4, 2]),
        1e-12,
    );
    assert!(
        bb_model
            .ln_posterior_predictive_likelihood(&[3, 4, 2], &[1, 5])
            .is_finite()
    );

    let gp = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
    let nb = gp.posterior_predictive(&[3, 5, 2, 4]);
    assert!(nb.pmf(3_u64) > 0.0);
    close(
        gp.ln_posterior_predictive(&[3, 5, 2, 4], 3).exp(),
        nb.pmf(3_u64),
        1e-12,
    );
    assert!(gp.ln_marginal_likelihood(&[3, 5, 2, 4]).is_finite());
    close(
        gp.ln_sequential_predictive(&[3, 5, 2, 4]),
        gp.ln_marginal_likelihood(&[3, 5, 2, 4]),
        1e-12,
    );
    assert!(
        gp.ln_posterior_predictive_likelihood(&[3, 5, 2, 4], &[1, 0])
            .is_finite()
    );
    let exposed_post = gp.posterior_with_exposure(&[3, 2], &[0.5, 1.5]).unwrap();
    close(exposed_post.shape(), 7.0, 1e-12);
    assert!(
        gp.posterior_predictive_with_exposure(&[3, 2], &[0.5, 1.5], 2.0)
            .unwrap()
            .pmf(4_u64)
            > 0.0
    );
    assert!(
        gp.ln_posterior_predictive_with_exposure(&[3, 2], &[0.5, 1.5], 4, 2.0)
            .is_finite()
    );
    assert!(
        gp.ln_marginal_likelihood_with_exposure(&[3, 2], &[0.5, 1.5])
            .is_finite()
    );
    close(
        gp.ln_sequential_predictive_with_exposure(&[3, 2], &[0.5, 1.5]),
        gp.ln_marginal_likelihood_with_exposure(&[3, 2], &[0.5, 1.5]),
        1e-12,
    );

    let ge = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
    let waiting = [0.5, 1.5, 2.0];
    let waiting_pred = ge.posterior_predictive(&waiting);
    close(
        ge.ln_posterior_predictive(&waiting, 1.0).exp(),
        waiting_pred.pdf(1.0),
        1e-12,
    );
    close(
        ge.ln_sequential_predictive(&waiting),
        ge.ln_marginal_likelihood(&waiting),
        1e-12,
    );
    assert!(
        ge.ln_posterior_predictive_likelihood(&waiting, &[0.25, 0.75])
            .is_finite()
    );
    close(
        ge.posterior_with_exposure(2, 4.0).unwrap().rate(),
        5.0,
        1e-12,
    );
    close(
        ge.posterior_predictive_with_exposure(2, 4.0)
            .unwrap()
            .scale(),
        5.0,
        1e-12,
    );
    assert!(
        ge.ln_posterior_predictive_with_exposure(2, 4.0, 1, 2.0)
            .is_finite()
    );

    let gamma_sum = Gamma::sum_independent_same_rate(&[
        Gamma::new(2.0, 3.0).unwrap(),
        Gamma::new(5.0, 3.0).unwrap(),
    ])
    .unwrap();
    close(gamma_sum.shape(), 7.0, 1e-12);
}

// The Normal-known-variance, Normal-Inverse-Gamma / Normal-Gamma, and Dirichlet-categorical
// conjugate-flow public-surface tests moved to ito (their predictive observations and count
// models are matrix-typed); see ito's test suite. `Normal::linear_combination_independent`
// (scalar) stays exercised below.
