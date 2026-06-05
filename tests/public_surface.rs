//! Closed-form Bayesian layer and numerical-stability surface, exercised through
//! the public crate API.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bayes::{
    Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Conjugate, Continuous,
    ContinuousCdf, ContinuousCdfExt, ContinuousDistributionExt, ContinuousFn, Dirichlet,
    DirichletCategorical, Discrete, DiscreteDistributionExt, DiscreteFn, Ecdf, Exponential,
    ExponentialFamily, Gamma, GammaPoisson, MeanVar, Mixture, Moments, Normal, NormalInverseGamma,
    NormalKnownVariance, Poisson, StudentT, SufficientStatistic, histogram,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn normal_log_cdf_is_accurate_in_the_far_tail_where_naive_underflows() {
    let n = Normal::new(0.0, 1.0).unwrap();
    // Naive `cdf(x).ln()` collapses to -inf far in the tail...
    assert_eq!(n.cdf(-40.0).ln(), f64::NEG_INFINITY);
    // ...but the stable ln_cdf stays finite and matches the analytic value
    // ln Φ(-40) ≈ -804.6084 (Mills-ratio asymptotic + correction).
    let lc = n.ln_cdf(-40.0);
    assert!(lc.is_finite());
    close(lc, -804.6084, 1e-2);
    // Symmetry: ln_sf(40) == ln_cdf(-40).
    close(n.ln_sf(40.0), n.ln_cdf(-40.0), 1e-12);
    // Consistency with cdf where both are well-conditioned.
    close(n.ln_cdf(-1.5).exp(), n.cdf(-1.5), 1e-12);
}

#[test]
fn exponential_log_survival_is_exact() {
    let e = Exponential::new(2.0).unwrap();
    // ln P(X > x) = -λx exactly — no underflow even far out.
    close(e.ln_sf(500.0), -1000.0, 1e-12);
    assert_eq!(e.sf(500.0).ln(), f64::NEG_INFINITY); // the naive form fails here
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
    // Two far-apart narrow Normals; evaluate far in one tail where each component
    // log-density is hugely negative. ln_pdf must agree with pdf().ln() in the
    // safe regime and stay finite/consistent in the extreme.
    let m = Mixture::new(
        vec![0.5, 0.5],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(50.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    close(m.ln_pdf(1.0), m.pdf(1.0).ln(), 1e-12);
    // Far tail: pdf underflows to 0 (ln -> -inf) but ln_pdf is finite.
    let far = m.ln_pdf(-40.0);
    assert!(far.is_finite());
    assert_eq!(m.pdf(-40.0).ln(), f64::NEG_INFINITY);
}

#[test]
fn beta_bernoulli_conjugate_flow() {
    let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap()); // uniform prior
    let data = [true, true, false, true, true, false, true]; // 5 of 7
    let post = model.posterior(&data);
    close(post.alpha(), 6.0, 1e-12);
    close(post.beta(), 3.0, 1e-12);
    let pred: Bernoulli = model.posterior_predictive(&data);
    close(pred.p(), 6.0 / 9.0, 1e-12);
}

#[test]
fn beta_binomial_and_gamma_poisson_predictives() {
    // Beta-Binomial predictive is a real BetaBinomial distribution.
    let bb_model = BetaBinomialModel::new(Beta::new(2.0, 2.0).unwrap(), 5);
    let pred: BetaBinomial = bb_model.posterior_predictive(&[3, 4, 2]);
    let mass: f64 = (0..=5u64).map(|k| pred.pmf(k)).sum();
    close(mass, 1.0, 1e-12);

    // Gamma-Poisson predictive is a NegativeBinomial.
    let gp = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
    let nb = gp.posterior_predictive(&[3, 5, 2, 4]);
    assert!(nb.pmf(3) > 0.0);
}

#[test]
fn normal_normal_known_variance_posterior() {
    let model = NormalKnownVariance::new(Normal::new(0.0, 10.0).unwrap(), 1.0);
    let data = [4.9, 5.1, 5.0, 4.8, 5.2];
    let post = model.posterior(&data);
    close(post.mean().unwrap(), 5.0, 0.02);
    assert!(post.variance().unwrap() < 1.0);
}

#[test]
fn normal_inverse_gamma_full_conjugate_flow() {
    let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
    let data = [1.0, 2.0, 3.0];
    let post = nig.posterior(&data);
    close(post.mu(), 1.5, 1e-12);
    close(post.alpha(), 3.5, 1e-12);
    // Posterior-predictive is a location-scale Student-t.
    let pred: StudentT = nig.posterior_predictive(&data);
    close(pred.freedom(), 7.0, 1e-12); // 2·α′
    close(pred.location(), 1.5, 1e-12);
    // Marginal of the mean is tighter than the predictive (no likelihood noise).
    assert!(post.mean_marginal().scale() < pred.scale());
    // Marginal of the variance is an Inverse-Gamma with a defined mean here.
    assert!(post.variance_marginal().mean().is_some());
}

#[test]
fn dirichlet_categorical_conjugate_flow() {
    let model = DirichletCategorical::new(Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap());
    let post = model.posterior(&[0u64, 0, 1, 2, 0]);
    assert_eq!(post.alpha(), &[4.0, 2.0, 2.0]);
    let pred = model.posterior_predictive(&[0u64, 0, 1, 2, 0]);
    close(pred.probabilities()[0], 0.5, 1e-12); // 4/8
}

#[test]
fn trigamma_is_exposed_and_correct() {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    close(bayes::function::trigamma(1.0), pi2 / 6.0, 1e-10);
}

#[test]
fn empirical_distribution_and_goodness_of_fit_public_surface() {
    use bayes::statistics::hypothesis::{jarque_bera, ks_two_sample};

    let e = Ecdf::new(&[3.0, 1.0, 2.0, 4.0]);
    close(e.eval(2.0), 0.5, 1e-12); // two of four ≤ 2
    let h = histogram(&[0.0, 0.5, 1.0, 1.5, 2.0], 2).unwrap();
    assert_eq!(h.counts.iter().sum::<u64>(), 5);

    // KS two-sample clearly separates a large shift.
    let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let b: Vec<f64> = a.iter().map(|x| x + 50.0).collect();
    assert!(ks_two_sample(&a, &b).p_value < 0.01);

    // Jarque–Bera does not reject symmetric data.
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

    // Fisher's exact (2×2) and Cramér's V.
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
    // Wald (inverse-Gaussian) analytic moments.
    close(wald.mean().unwrap(), 1.0, 1e-12);
    close(wald.variance().unwrap(), 0.5, 1e-12); // μ³/λ = 1/2
}

#[test]
fn von_mises_directional_public_surface() {
    use bayes::VonMises;
    let vm = VonMises::new(0.0, 4.0).unwrap();
    // Density peaks at the mean direction.
    assert!(vm.pdf(0.0) > vm.pdf(std::f64::consts::PI));
    // Concentration ⇒ small circular variance, in [0, 1].
    let cv = vm.circular_variance();
    assert!((0.0..=1.0).contains(&cv) && cv < 0.5);
    // Bessel I₀ is exposed and correct: I₀(0) = 1.
    close(bayes::function::i0(0.0), 1.0, 1e-12);
}

#[test]
fn streaming_covariance_and_tdigest_public_surface() {
    use bayes::{RunningCovariance, TDigest};

    // Bivariate accumulator recovers an exact linear relationship.
    let mut rc = RunningCovariance::new();
    for i in 0..100 {
        let x = i as f64;
        rc.push(x, 3.0 * x + 2.0);
    }
    close(rc.slope(), 3.0, 1e-9);
    close(rc.intercept(), 2.0, 1e-9);
    close(rc.correlation(), 1.0, 1e-9);

    // t-digest median of uniform data ≈ 0.5; merge stays consistent.
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

    // Batch fit through the public ExponentialFamily API.
    let batch = Normal::fit(&data).unwrap();

    // The same fit assembled from two independently-summarised shards.
    let stat = MeanVar::from_sample(&data[..111]).merge(&MeanVar::from_sample(&data[111..]));
    let streamed = Normal::from_sufficient(&stat).unwrap();

    close(streamed.mean().unwrap(), batch.mean().unwrap(), 1e-10);
    close(
        streamed.variance().unwrap(),
        batch.variance().unwrap(),
        1e-10,
    );

    // Discrete families fit too.
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
