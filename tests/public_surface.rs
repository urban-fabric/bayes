//! Closed-form Bayesian layer and numerical-stability surface, exercised through
//! the public crate API.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The multivariate-matrix, matrix-conjugate, and S² directional families (and the dense-
// matrix sample-statistic admissibility predicates) moved to the sibling crate `ito`; their
// public-surface tests moved with them (see ito's test suite). What remains here is bayes'
// euclid-only surface: scalar families, scalar conjugate models, divergences, empirical
// distributions, t-digest, and the lazy kernels.
use bayes::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Conjugate, Constraint,
    Continuous, ContinuousCdf, ContinuousCdfExt, ContinuousDistributionExt, ContinuousFn, Discrete,
    DiscreteDistributionExt, DiscreteFn, Ecdf, Exponential, ExponentialFamily, Gamma,
    GammaExponential, GammaPoisson, MeanVar, Mixture, Moments, Normal, OrderStatistic, Poisson,
    SufficientStatistic, Uniform, histogram,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn normal_log_cdf_is_accurate_in_the_far_tail_where_naive_underflows() {
    let n = Normal::new(0.0, 1.0).unwrap();
    assert_eq!(n.cdf(-40.0).ln(), f64::NEG_INFINITY);
    let lc = n.ln_cdf(-40.0);
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
        bayes::lazy::normal_log_pdf(0.25_f64, 1.0, 2.0),
        n.ln_pdf(0.25),
        1e-12,
    );
    close(
        bayes::lazy::bernoulli_logit_log_pmf(1.0_f64, 2.0),
        2.0 - (1.0 + 2.0_f64.exp()).ln(),
        1e-12,
    );
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
        bayes::function::ln_beta(6.0, 3.0) - bayes::function::ln_beta(1.0, 1.0),
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
        pred.pmf(3),
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
    assert!(nb.pmf(3) > 0.0);
    close(
        gp.ln_posterior_predictive(&[3, 5, 2, 4], 3).exp(),
        nb.pmf(3),
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
            .pmf(4)
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
#[test]
fn normal_linear_combination_is_public() {
    let combo = Normal::linear_combination_independent(
        &[
            (2.0, Normal::new(1.0, 2.0).unwrap()),
            (-1.0, Normal::new(3.0, 4.0).unwrap()),
        ],
        10.0,
    )
    .unwrap();
    close(combo.mean().unwrap(), 9.0, 1e-12);
    close(combo.variance().unwrap(), 32.0, 1e-12);
}

#[test]
fn trigamma_is_exposed_and_correct() {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    close(bayes::function::trigamma(1.0), pi2 / 6.0, 1e-10);
}

#[test]
fn stable_log_weights_are_public_and_have_valid_ess() {
    use bayes::{
        effective_sample_size, is_effective_sample_size_admissible, normalized_log_weights,
        normalized_weights,
    };

    let raw = normalized_weights(&[1.0, 2.0, 1.0]);
    close(raw.iter().sum(), 1.0, 1e-12);

    let p = normalized_log_weights(&[1000.0, 1001.0, 1000.0]);
    close(p.iter().sum(), 1.0, 1e-12);
    assert!(p[1] > p[0]);

    let ess = effective_sample_size(&p);
    assert!(is_effective_sample_size_admissible(ess, p.len()));

    let lp = bayes::function::log_softmax(&[0.0, 0.0]);
    close(lp[0], -2.0_f64.ln(), 1e-12);
}

#[test]
fn variance_decomposition_is_public_and_exactly_partitions_spread() {
    let values = [1.0, 3.0, 10.0, 12.0];
    let groups = [0usize, 0, 1, 1];
    let split = bayes::variance_decomposition(&values, &groups).unwrap();

    close(split.within_ss, 4.0, 1e-12);
    close(split.between_ss, 81.0, 1e-12);
    close(split.total_ss, 85.0, 1e-12);
    close(
        split.within_variance + split.between_variance,
        split.total_variance,
        1e-12,
    );
    close(split.explained_fraction(), 81.0 / 85.0, 1e-12);
}

#[test]
fn probability_vectors_scores_and_constraints_are_public() {
    use bayes::{
        brier_score, is_probability_vector, is_row_stochastic, log_loss, log_score,
        normalize_probability_rows, normalize_probability_weights, ranked_probability_score,
        spherical_score, validate_probability_vector, validate_row_stochastic,
    };

    let p = normalize_probability_weights(&[2.0, 5.0, 3.0]);
    assert!(is_probability_vector(&p));
    validate_probability_vector(&p).unwrap();
    close(log_score(&p, 1), 0.5_f64.ln(), 1e-12);
    close(log_loss(&p, 1), -0.5_f64.ln(), 1e-12);
    close(brier_score(&p, 1), 0.04 + 0.25 + 0.09, 1e-12);
    close(
        spherical_score(&p, 1),
        0.5 / (0.04_f64 + 0.25 + 0.09).sqrt(),
        1e-12,
    );
    close(ranked_probability_score(&p, 1), 0.04 + 0.09, 1e-12);

    let rows = normalize_probability_rows(&[vec![2.0, 8.0], vec![3.0, 3.0]]);
    assert!(is_row_stochastic(&rows));
    validate_row_stochastic(&rows).unwrap();
    // The dense-matrix row-stochastic validator (`is_row_stochastic_matrix`) moved to ito;
    // its public-surface assertion lives there.

    let c = Constraint::UnitInterval;
    let y = 1000.0;
    close(c.log_abs_det_jacobian(y), -1000.0, 1e-12);

    let x = Constraint::Interval { lo: -2.0, hi: 5.0 }.to_constrained(0.75);
    close(
        Constraint::Interval { lo: -2.0, hi: 5.0 }.log_abs_det_jacobian(0.75)
            + Constraint::Interval { lo: -2.0, hi: 5.0 }.log_abs_det_inverse_jacobian(x),
        0.0,
        1e-9,
    );
}

#[test]
fn empirical_distribution_and_goodness_of_fit_public_surface() {
    use bayes::statistics::hypothesis::{jarque_bera, ks_two_sample};

    let e = Ecdf::new(&[3.0, 1.0, 2.0, 4.0]);
    close(e.eval(2.0), 0.5, 1e-12);
    let h = histogram(&[0.0, 0.5, 1.0, 1.5, 2.0], 2).unwrap();
    assert_eq!(h.counts.iter().sum::<u64>(), 5);

    let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let b: Vec<f64> = a.iter().map(|x| x + 50.0).collect();
    assert!(ks_two_sample(&a, &b).p_value < 0.01);

    let sym: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
    assert!(jarque_bera(&sym).p_value > 0.05);
}

#[test]
fn confidence_intervals_and_association_public_surface() {
    use bayes::statistics::{cramers_v, hypothesis::fishers_exact};
    use bayes::{ConfidenceInterval, mcdiarmid_bound, mean_t_interval, proportion_wilson};

    let data = [4.8, 5.0, 5.2, 4.9, 5.1];
    let ci: ConfidenceInterval = mean_t_interval(&data, 0.95);
    assert!(ci.contains(5.0) && ci.lower < ci.upper);
    let w = proportion_wilson(8, 10, 0.95);
    assert!(w.lower > 0.0 && w.upper < 1.0);
    assert!(mcdiarmid_bound(4.0, 1.0) < 1.0);

    assert!(fishers_exact(10, 0, 0, 10).p_value < 0.01);
    close(cramers_v(&[vec![10.0, 0.0], vec![0.0, 10.0]]), 1.0, 1e-12);
}

#[test]
fn public_distribution_surface_round_trips() {
    use bayes::{Frechet, Logistic, Lomax, Wald};
    let logi = Logistic::new(0.0, 1.0).unwrap();
    let lom = Lomax::new(3.0, 2.0).unwrap();
    let fre = Frechet::new(2.5, 1.0).unwrap();
    let wald = Wald::new(1.0, 2.0).unwrap();
    for &p in &[0.2, 0.5, 0.8] {
        close(logi.cdf(logi.inverse_cdf(p)), p, 1e-9);
        close(lom.cdf(lom.inverse_cdf(p)), p, 1e-9);
        close(fre.cdf(fre.inverse_cdf(p)), p, 1e-9);
        close(wald.cdf(wald.inverse_cdf(p)), p, 1e-6);
    }
    close(wald.mean().unwrap(), 1.0, 1e-12);
    close(wald.variance().unwrap(), 0.5, 1e-12);
}

#[test]
fn von_mises_circular_public_surface() {
    // The S² directional families (von Mises-Fisher / spherical-uniform / Watson) moved to
    // ito; their public-surface assertions live in ito's test suite. The circular von Mises
    // and the `i0` Bessel function stay in bayes.
    use bayes::VonMises;

    let vm = VonMises::new(0.0, 4.0).unwrap();
    assert!(vm.pdf(0.0) > vm.pdf(std::f64::consts::PI));
    let cv = vm.circular_variance();
    assert!((0.0..=1.0).contains(&cv) && cv < 0.5);
    close(bayes::function::i0(0.0), 1.0, 1e-12);
}

#[test]
fn streaming_tdigest_public_surface() {
    // `RunningCovariance` (the paired-observation matrix accumulator) moved to ito; its
    // streaming-covariance surface assertion lives there. `TDigest` stays in bayes.
    use bayes::TDigest;

    let d: TDigest = (0..2000).map(|i| i as f64 / 1999.0).collect();
    assert!((d.quantile(0.5) - 0.5).abs() < 0.02);
    let mut a: TDigest = (0..1000).map(|i| i as f64 / 1999.0).collect();
    let b: TDigest = (1000..2000).map(|i| i as f64 / 1999.0).collect();
    a.merge(&b);
    assert_eq!(a.count(), 2000);
    assert!((a.quantile(0.9) - 0.9).abs() < 0.03);
}

#[test]
fn exponential_family_fit_and_streaming_merge_agree() {
    let data: Vec<f64> = (0..300)
        .map(|i| (i as f64 * 0.13).cos() * 2.0 + 5.0)
        .collect();

    let batch = Normal::fit(&data).unwrap();

    let stat = MeanVar::from_sample(&data[..111]).merge(&MeanVar::from_sample(&data[111..]));
    let streamed = Normal::from_sufficient(&stat).unwrap();

    close(streamed.mean().unwrap(), batch.mean().unwrap(), 1e-10);
    close(
        streamed.variance().unwrap(),
        batch.variance().unwrap(),
        1e-10,
    );

    close(
        Poisson::fit(&[2u64, 4, 6, 0, 3]).unwrap().lambda(),
        3.0,
        1e-12,
    );
}

#[test]
fn functional_distribution_adapters_are_public_and_composable() {
    let unit = ContinuousFn::new(
        0.0,
        1.0,
        |x| if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 },
        |x: f64| x.clamp(0.0, 1.0),
        |p: f64| p.clamp(0.0, 1.0),
    )
    .unwrap();

    close(unit.ln_likelihood(&[0.1, 0.4, 0.9]), 0.0, 1e-12);
    close(unit.probability_between(0.2, 0.7), 0.5, 1e-12);
    close(unit.ln_probability_between(0.2, 0.7).exp(), 0.5, 1e-12);

    let shifted = unit.affine(2.0, -1.0).unwrap();
    close(shifted.pdf(0.0), 0.5, 1e-12);

    let max = unit.maximum_of(4).unwrap();
    close(max.cdf(0.5), 0.5_f64.powi(4), 1e-12);
    let third = OrderStatistic::new(Uniform::unit_interval(), 5, 3).unwrap();
    close(third.pdf(0.5), 1.875, 1e-12);
    let exp = Exponential::new(2.0).unwrap();
    close(exp.hazard(3.0), 2.0, 1e-12);
    close(exp.cumulative_hazard(3.0), 6.0, 1e-12);

    let coin = DiscreteFn::new(
        0u64,
        1u64,
        |k: u64| match k {
            0 => 0.25,
            1 => 0.75,
            _ => 0.0,
        },
        |k: u64| if k == 0 { 0.25 } else { 1.0 },
        |p: f64| if p <= 0.25 { 0 } else { 1 },
    );
    close(
        coin.ln_likelihood(&[1, 1, 0]).exp(),
        0.75 * 0.75 * 0.25,
        1e-12,
    );
}
