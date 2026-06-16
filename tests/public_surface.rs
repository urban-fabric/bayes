//! Closed-form Bayesian layer and numerical-stability surface, exercised through
//! the public crate API.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bayes::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Conjugate, Constraint,
    Continuous, ContinuousCdf, ContinuousCdfExt, ContinuousDistributionExt, ContinuousFn,
    Dirichlet, DirichletCategorical, Discrete, DiscreteDistributionExt, DiscreteFn, Ecdf,
    Exponential, ExponentialFamily, Gamma, GammaPoisson, MarkovChain, MeanVar, Mixture, Moments,
    MultivariateNormal, Normal, NormalInverseGamma, NormalKnownVariance, Poisson, StudentT,
    SufficientStatistic, histogram, is_correlation_matrix_admissible,
    is_covariance_matrix_admissible,
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
    close(
        model.ln_marginal_likelihood(&data),
        bayes::function::ln_beta(6.0, 3.0) - bayes::function::ln_beta(1.0, 1.0),
        1e-12,
    );
}

#[test]
fn beta_binomial_and_gamma_poisson_predictives() {
    let bb_model = BetaBinomialModel::new(Beta::new(2.0, 2.0).unwrap(), 5);
    let pred: BetaBinomial = bb_model.posterior_predictive(&[3, 4, 2]);
    let mass: f64 = (0..=5u64).map(|k| pred.pmf(k)).sum();
    close(mass, 1.0, 1e-12);
    assert!(bb_model.ln_marginal_likelihood(&[3, 4, 2]).is_finite());

    let gp = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
    let nb = gp.posterior_predictive(&[3, 5, 2, 4]);
    assert!(nb.pmf(3) > 0.0);
    assert!(gp.ln_marginal_likelihood(&[3, 5, 2, 4]).is_finite());
}

#[test]
fn normal_normal_known_variance_posterior() {
    let model = NormalKnownVariance::new(Normal::new(0.0, 10.0).unwrap(), 1.0);
    let data = [4.9, 5.1, 5.0, 4.8, 5.2];
    let post = model.posterior(&data);
    close(post.mean().unwrap(), 5.0, 0.02);
    assert!(post.variance().unwrap() < 1.0);
    assert!(model.ln_marginal_likelihood(&data).is_finite());
}

#[test]
fn normal_inverse_gamma_full_conjugate_flow() {
    let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
    let data = [1.0, 2.0, 3.0];
    let post = nig.posterior(&data);
    close(post.mu(), 1.5, 1e-12);
    close(post.alpha(), 3.5, 1e-12);
    let pred: StudentT = nig.posterior_predictive(&data);
    close(pred.freedom(), 7.0, 1e-12);
    close(pred.location(), 1.5, 1e-12);
    assert!(post.mean_marginal().scale() < pred.scale());
    assert!(post.variance_marginal().mean().is_some());
    assert!(nig.ln_marginal_likelihood(&data).is_finite());
}

#[test]
fn dirichlet_categorical_conjugate_flow() {
    let model = DirichletCategorical::new(Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap());
    let post = model.posterior(&[0u64, 0, 1, 2, 0]);
    assert_eq!(post.alpha(), &[4.0, 2.0, 2.0]);
    let pred = model.posterior_predictive(&[0u64, 0, 1, 2, 0]);
    close(pred.probabilities()[0], 0.5, 1e-12);
    assert!(
        model
            .ln_marginal_likelihood(&[0u64, 0, 1, 2, 0])
            .is_finite()
    );
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
        brier_score, is_probability_vector, is_row_stochastic, is_row_stochastic_matrix, log_loss,
        log_score, normalize_probability_rows, normalize_probability_weights,
        ranked_probability_score, spherical_score, validate_probability_vector,
        validate_row_stochastic,
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
    assert!(is_row_stochastic_matrix(
        &ramanujan::DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.5, 0.5],])
    ));

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
fn finite_markov_chain_step_distributions_are_public() {
    let mc = MarkovChain::new(ramanujan::DenseMatrix::from_rows(&[
        vec![0.2, 0.8],
        vec![0.6, 0.4],
    ]))
    .unwrap();

    close(mc.transition_probability(0, 1, 2), 0.48, 1e-12);
    let dist = mc.distribution_after(&[1.0, 0.0], 2);
    close(dist[0], 0.52, 1e-12);
    close(dist[1], 0.48, 1e-12);
    assert_eq!(mc.state_distribution(1, 0), vec![0.0, 1.0]);

    close(mc.first_return_probability(0, 2), 0.48, 1e-12);
    let recurrence = mc.mean_recurrence_times();
    close(recurrence[0], 7.0 / 3.0, 1e-12);
    close(recurrence[1], 7.0 / 4.0, 1e-12);
    assert!(mc.is_reversible());
    let reversed = mc.time_reversed_stationary().unwrap();
    close(reversed.transition().get(0, 1), 0.8, 1e-12);

    let absorbing = MarkovChain::new(ramanujan::DenseMatrix::from_rows(&[
        vec![0.5, 0.5, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.2, 0.3, 0.5],
    ]))
    .unwrap();
    assert_eq!(absorbing.absorbing_states(), vec![1]);
    assert_eq!(absorbing.recurrent_states(), vec![1]);
    assert_eq!(absorbing.transient_states(), vec![0, 2]);
    close(absorbing.hitting_probabilities(&[1])[2], 1.0, 1e-12);
    close(absorbing.expected_hitting_times(&[1])[2], 2.8, 1e-12);
    close(absorbing.absorption_probabilities().get(0, 0), 1.0, 1e-12);
    close(absorbing.hitting_time_variances(&[1])[2], 3.76, 1e-12);
    close(absorbing.absorption_time_variances()[2], 3.76, 1e-12);
    let fundamental = absorbing.fundamental_matrix();
    close(fundamental.get(0, 0), 2.0, 1e-12);
    close(fundamental.get(1, 0), 0.8, 1e-12);
    close(
        absorbing.expected_transient_visit_counts().get(1, 1),
        2.0,
        1e-12,
    );
}

#[test]
fn covariance_and_correlation_admissibility_are_public() {
    let cov = ramanujan::DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
    assert!(is_covariance_matrix_admissible(&cov));
    let bad_cov = ramanujan::DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]]);
    assert!(!is_covariance_matrix_admissible(&bad_cov));

    let corr = ramanujan::DenseMatrix::from_rows(&[vec![1.0, -0.5], vec![-0.5, 1.0]]);
    assert!(is_correlation_matrix_admissible(&corr));
    let bad_corr = ramanujan::DenseMatrix::from_rows(&[vec![1.0, 1.2], vec![1.2, 1.0]]);
    assert!(!is_correlation_matrix_admissible(&bad_corr));
}

#[test]
fn multivariate_normal_closed_form_operations_are_public() {
    let mvn = MultivariateNormal::new(
        vec![0.0, 0.0],
        ramanujan::DenseMatrix::from_rows(&[vec![4.0, 2.0], vec![2.0, 9.0]]),
    )
    .unwrap();

    let marginal = mvn.marginal(&[1]).unwrap();
    close(marginal.mean()[0], 0.0, 1e-12);
    close(marginal.covariance_matrix().get(0, 0), 9.0, 1e-12);

    let conditional = mvn.condition(&[0], &[2.0]).unwrap();
    close(conditional.mean()[0], 1.0, 1e-10);
    close(conditional.covariance_matrix().get(0, 0), 8.0, 1e-10);

    let affine = mvn
        .affine_transform(
            &ramanujan::DenseMatrix::from_rows(&[vec![1.0, 1.0]]),
            &[0.5],
        )
        .unwrap();
    close(affine.mean()[0], 0.5, 1e-12);
    close(affine.covariance_matrix().get(0, 0), 17.0, 1e-12);

    let other =
        MultivariateNormal::new(vec![1.0, 0.0], ramanujan::DenseMatrix::identity(2)).unwrap();
    assert!(mvn.ln_product_integral(&other).is_finite());
    assert!(mvn.precision_fusion(&other).is_ok());
    assert!(
        mvn.ln_likelihood(&[vec![0.0, 0.0], vec![1.0, 1.0]])
            .is_finite()
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
    use bayes::{ConfidenceInterval, mean_t_interval, proportion_wilson};

    let data = [4.8, 5.0, 5.2, 4.9, 5.1];
    let ci: ConfidenceInterval = mean_t_interval(&data, 0.95);
    assert!(ci.contains(5.0) && ci.lower < ci.upper);
    let w = proportion_wilson(8, 10, 0.95);
    assert!(w.lower > 0.0 && w.upper < 1.0);

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
fn von_mises_directional_public_surface() {
    use bayes::VonMises;
    let vm = VonMises::new(0.0, 4.0).unwrap();
    assert!(vm.pdf(0.0) > vm.pdf(std::f64::consts::PI));
    let cv = vm.circular_variance();
    assert!((0.0..=1.0).contains(&cv) && cv < 0.5);
    close(bayes::function::i0(0.0), 1.0, 1e-12);
}

#[test]
fn streaming_covariance_and_tdigest_public_surface() {
    use bayes::{RunningCovariance, TDigest};

    let mut rc = RunningCovariance::new();
    for i in 0..100 {
        let x = i as f64;
        rc.push(x, 3.0 * x + 2.0);
    }
    close(rc.slope(), 3.0, 1e-9);
    close(rc.intercept(), 2.0, 1e-9);
    close(rc.correlation(), 1.0, 1e-9);

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
