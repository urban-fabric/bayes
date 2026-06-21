//! Cross-distribution checks of the trait spine: CDF monotonicity + bounds,
//! `cdf(inverse_cdf(p)) ≈ p`, PDF/PMF sanity, and analytic-vs-sampled moments.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bayes::{
    Bernoulli, Continuous, ContinuousCdf, Discrete, DiscreteCdf, DiscreteUniform, Distribution,
    Estimate, IntegerDiscrete, IntegerDiscreteCdf, LogNormal, Moments, Normal, Poisson, Support,
    Uniform,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn normal_cdf_pdf_and_quantile() {
    let n = Normal::new(2.0, 3.0).unwrap();
    close(n.cdf(2.0), 0.5, 1e-12);
    close(n.cdf(2.0 + 3.0), 0.841_344_746_068_543, 1e-9);
    for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
        close(n.cdf(n.inverse_cdf(p)), p, 1e-9);
    }
    close(
        n.pdf(2.0),
        1.0 / (3.0 * std::f64::consts::TAU.sqrt()),
        1e-12,
    );
    close(n.cdf(1.0) + n.sf(1.0), 1.0, 1e-12);
    assert_eq!(n.mean(), 2.0);
    assert_eq!(n.variance(), 9.0);
}

#[test]
fn uniform_cdf_and_moments() {
    let u = Uniform::new(-2.0, 4.0).unwrap();
    close(u.cdf(1.0), 0.5, 1e-12);
    close(u.inverse_cdf(0.5), 1.0, 1e-12);
    let lo: f64 = u.min();
    let hi: f64 = u.max();
    assert_eq!(lo, -2.0);
    assert_eq!(hi, 4.0);
    close(u.mean(), 1.0, 1e-12);
    close(u.variance(), 36.0 / 12.0, 1e-12);
    close(u.pdf(0.0), 1.0 / 6.0, 1e-12);
    assert_eq!(u.pdf(5.0), 0.0);
}

#[test]
fn lognormal_cdf_quantile_and_mean() {
    let ln = LogNormal::new(0.0, 1.0).unwrap();
    close(ln.cdf(1.0), 0.5, 1e-9);
    close(ln.median(), 1.0, 1e-12);
    close(ln.mean(), (0.5f64).exp(), 1e-12);
    for &p in &[0.1, 0.5, 0.9] {
        close(ln.cdf(ln.inverse_cdf(p)), p, 1e-9);
    }
    assert_eq!(ln.pdf(-1.0), 0.0);
}

#[test]
fn poisson_pmf_cdf_and_moments() {
    let p = Poisson::new(4.0).unwrap();
    let mut cum = 0.0;
    for k in 0..20u64 {
        cum += p.pmf(k);
        close(p.cdf(k), cum, 1e-9);
    }
    close(cum, 1.0, 1e-6);
    assert_eq!(p.mean(), 4.0);
    assert_eq!(p.variance(), 4.0);
    close(p.pmf(0_u64), (-4.0f64).exp(), 1e-12);
}

#[test]
fn bernoulli_and_discrete_uniform_spine() {
    let b = Bernoulli::new(0.3).unwrap();
    close(b.pmf(0_u64), 0.7, 1e-12);
    close(b.pmf(1_u64), 0.3, 1e-12);
    close(b.cdf(0_u64), 0.7, 1e-12);
    assert_eq!(b.cdf(1_u64), 1.0);
    assert_eq!(b.mean(), 0.3);

    let d = DiscreteUniform::new(1, 7).unwrap();
    close(d.pmf(3_i64), 1.0 / 6.0, 1e-12);
    assert_eq!(d.pmf(7_i64), 0.0);
    assert_eq!(d.min(), 1);
    assert_eq!(d.max(), 6);
    close(d.cdf(3_i64), 0.5, 1e-12);
    close(d.mean(), 3.5, 1e-12);
}

#[test]
fn estimate_recovers_parameters_from_samples() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let truth = Normal::new(5.0, 2.0).unwrap();
    let data: Vec<f64> = (0..20_000).map(|_| truth.sample(&mut rng)).collect();
    let fit = Normal::from_sample(&data).unwrap();
    close(fit.mean(), 5.0, 0.05);
    close(fit.std_dev(), 2.0, 0.05);
}

#[test]
fn sampled_mean_matches_analytic_moments() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let p = Poisson::new(7.0).unwrap();
    let n = 50_000;
    let sum: u64 = (0..n).map(|_| p.sample(&mut rng)).sum();
    close(sum as f64 / n as f64, p.mean(), 0.05);
}
