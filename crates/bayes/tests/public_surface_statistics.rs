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
    Continuous, ContinuousCdf, ContinuousCdfExt, ContinuousDistributionExt, ContinuousFn, Discrete,
    DiscreteDistributionExt, DiscreteFn, Ecdf, Exponential, ExponentialFamily, Gamma,
    GammaExponential, GammaPoisson, MeanVar, Mixture, Moments, Normal, OrderStatistic, Poisson,
    Real, SufficientStatistic, Uniform, histogram,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

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
    close(combo.mean(), 9.0, 1e-12);
    close(combo.variance(), 32.0, 1e-12);
}

#[test]
fn trigamma_is_exposed_and_correct() {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    close(1.0_f64.trigamma(), pi2 / 6.0, 1e-10);
}

#[test]
fn fisher_information_crlb_and_jeffreys_are_public() {
    use bayes::{Binomial, FisherInformation, Rayleigh};
    // Closed-form information about each family's canonical scalar parameter.
    close(
        Bernoulli::new(0.25).unwrap().fisher_information(),
        1.0 / (0.25 * 0.75),
        1e-12,
    );
    close(Poisson::new(4.0).unwrap().fisher_information(), 0.25, 1e-12);
    close(
        Rayleigh::new(3.0).unwrap().fisher_information(),
        4.0 / 9.0,
        1e-12,
    );
    // The Cramér–Rao bound equals the efficient (MLE) variance for the Bernoulli
    // mean: p(1−p)/n.
    let b = Bernoulli::new(0.3).unwrap();
    close(b.cramer_rao_bound(100), 0.3 * 0.7 / 100.0, 1e-12);
    // Jeffreys priors are the proper Beta(1/2, 1/2).
    let j = Bernoulli::jeffreys_prior();
    close(j.alpha(), 0.5, 1e-12);
    close(j.beta(), 0.5, 1e-12);
    let jb = Binomial::jeffreys_prior();
    close(jb.alpha(), 0.5, 1e-12);
}

#[test]
fn characteristic_pgf_entropy_and_lmoments_are_public() {
    use bayes::{
        Cauchy, CharacteristicFunction, GeneralizedEntropy, LMoments, ProbabilityGenerating,
        Uniform,
    };
    // Characteristic function reaches the Cauchy, which has no MGF: φ(t)=e^{−|t|}.
    let (re, im) = Cauchy::new(0.0, 1.0).unwrap().characteristic_function(2.0);
    close(re, (-2.0_f64).exp(), 1e-12);
    close(im, 0.0, 1e-12);
    // PGF of Poisson(2) at 0 is the zero-count mass e^{−2}.
    close(Poisson::new(2.0).unwrap().pgf(0.0), (-2.0_f64).exp(), 1e-12);
    // Rényi entropy of a Uniform is ln(width) for any order.
    close(
        Uniform::new(0.0, 2.0).unwrap().renyi_entropy(3.0),
        2.0_f64.ln(),
        1e-12,
    );
    // L-moments: exponential L-skewness is the scale-free 1/3.
    close(
        Exponential::new(5.0).unwrap().l_skewness(),
        1.0 / 3.0,
        1e-12,
    );
}

#[test]
fn moment_generating_feeds_chernoff_tail() {
    use bayes::MomentGenerating;
    use bayes::statistics::chernoff_bound;
    // The standard-normal MGF drives the Chernoff method to the exact Gaussian
    // tail: minimized at tilt s = a/σ² = 2, e^{−s·a}·M(s) = e^{−2}.
    let z = Normal::new(0.0, 1.0).unwrap();
    close(
        chernoff_bound(z.mgf(2.0), 2.0, 2.0),
        (-2.0_f64).exp(),
        1e-12,
    );
    // Closed-form MGF: Gamma(3, 2) at t = 1 is (β/(β−t))^k = 8.
    close(Gamma::new(3.0, 2.0).unwrap().mgf(1.0), 8.0, 1e-12);
}

#[test]
fn stable_log_weights_are_public_and_have_valid_ess() {
    use bayes::{
        effective_sample_size, effective_sample_size_real, is_effective_sample_size_admissible,
        normalized_log_weights, normalized_weights, normalized_weights_real, weighted_mean_real,
        weighted_variance, weighted_variance_real,
    };

    let raw = normalized_weights(&[1.0, 2.0, 1.0]);
    close(raw.iter().sum(), 1.0, 1e-12);
    assert_eq!(normalized_weights_real(&[1.0, 2.0, 1.0]), raw);

    let p = normalized_log_weights(&[1000.0, 1001.0, 1000.0]);
    close(p.iter().sum(), 1.0, 1e-12);
    assert!(p[1] > p[0]);

    let ess = effective_sample_size(&p);
    assert!(is_effective_sample_size_admissible(ess, p.len()));
    close(effective_sample_size_real(&p), ess, 1e-12);
    close(
        weighted_mean_real(&[1.0, 2.0, 4.0], &[1.0, 2.0, 1.0]),
        2.25,
        1e-12,
    );
    close(
        weighted_variance_real(&[1.0, 2.0, 4.0], &[1.0, 2.0, 1.0]),
        weighted_variance(&[1.0, 2.0, 4.0], &[1.0, 2.0, 1.0]),
        1e-12,
    );
}

#[test]
fn generic_real_summary_reducers_are_public() {
    use bayes::{
        correlation_real, covariance_real, mean_real, population_variance_real, std_dev_real,
        variance_real, z_scores_real,
    };

    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [2.0, 4.0, 6.0, 8.0];
    close(mean_real(&x), 2.5, 1e-12);
    close(population_variance_real(&x), 1.25, 1e-12);
    close(variance_real(&x), 5.0 / 3.0, 1e-12);
    close(std_dev_real(&x), (5.0_f64 / 3.0).sqrt(), 1e-12);
    close(covariance_real(&x, &y), 10.0 / 3.0, 1e-12);
    close(correlation_real(&x, &y), 1.0, 1e-12);
    close(z_scores_real(&x).iter().sum::<f64>(), 0.0, 1e-12);
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
        normalize_probability_rows, normalize_probability_rows_real, normalize_probability_weights,
        normalize_probability_weights_real, ranked_probability_score, spherical_score,
        validate_probability_vector, validate_row_stochastic,
    };

    let p = normalize_probability_weights(&[2.0, 5.0, 3.0]);
    assert_eq!(normalize_probability_weights_real(&[2.0, 5.0, 3.0]), p);
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
    assert_eq!(
        normalize_probability_rows_real(&[vec![2.0, 8.0], vec![3.0, 3.0]]),
        rows
    );
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
    close(wald.mean(), 1.0, 1e-12);
    close(wald.variance(), 0.5, 1e-12);
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
    close(0.0_f64.ln_bessel_i0().exp(), 1.0, 1e-12);
}

#[test]
fn streaming_tdigest_public_surface() {
    use bayes::{RunningCovariance, TDigest};

    let d: TDigest = (0..2000).map(|i| i as f64 / 1999.0).collect();
    assert!((d.quantile(0.5) - 0.5).abs() < 0.02);
    let mut a: TDigest = (0..1000).map(|i| i as f64 / 1999.0).collect();
    let b: TDigest = (1000..2000).map(|i| i as f64 / 1999.0).collect();
    a.merge(&b);
    assert_eq!(a.count(), 2000);
    assert!((a.quantile(0.9) - 0.9).abs() < 0.03);

    let rc: RunningCovariance = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)].into_iter().collect();
    close(rc.slope(), 2.0, 1e-12);
    close(rc.intercept(), 1.0, 1e-12);
}

#[test]
fn exponential_family_fit_and_streaming_merge_agree() {
    let data: Vec<f64> = (0..300)
        .map(|i| (i as f64 * 0.13).cos() * 2.0 + 5.0)
        .collect();

    let batch = Normal::fit(&data).unwrap();

    let stat = MeanVar::from_sample(&data[..111]).merge(&MeanVar::from_sample(&data[111..]));
    let streamed = Normal::from_sufficient(&stat).unwrap();

    close(streamed.mean(), batch.mean(), 1e-10);
    close(streamed.variance(), batch.variance(), 1e-10);

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
