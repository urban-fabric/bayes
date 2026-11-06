//! Against samples drawn from the defining construction.
//!
//! Each family's doc comment says what it *is* — a sum of squared shifted
//! normals, a shifted normal over a root chi-squared, a ratio of two
//! chi-squareds. Those constructions are built here out of `Normal` and
//! `ChiSquared` and touch none of the code under test, so agreement says the
//! closed-form moments are the moments of the documented thing rather than of
//! the series that happens to be implemented.
//!
//! This replaced a check that integrated `Q(u)` over the unit interval. That
//! route cannot work here — `integrate_unit` places abscissae within `4e-18` of
//! both endpoints, which is well inside the plateau where a CDF is 1 to within
//! an ulp, so the quantile came back as `2^64` and the `(Q(u) − mean)²`
//! integrand turned it into `9e23`. It was also the slowest thing in the
//! workspace by two orders of magnitude.

use bayes_distribution_ext::{
    ChiSquared, Moments, NoncentralChiSquared, NoncentralF, NoncentralT, Normal,
};
use bayes_utils::Failures;
use bayes_utils::gof::{mean_tolerance, variance_tolerance};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

/// The mean and variance of `n` draws.
fn sample_moments(n: usize, mut draw: impl FnMut() -> f64) -> (f64, f64) {
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..n {
        let x = draw();
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / n as f64;
    (mean, sum_sq / n as f64 - mean * mean)
}

/// Every bound here is `5σ` of the relevant sampling error, derived from the
/// closed-form variance and kurtosis rather than tuned — a bound tightened
/// until the suite passed would be asserting the sampler's bias instead of the
/// family's moments.
#[test]
fn the_closed_form_moments_match_the_constructive_definition() {
    const N: usize = 200_000;
    let mut f = Failures::default();
    let mut rng = ChaCha8Rng::seed_from_u64(20_261_105);
    let z = Normal::new(0.0, 1.0).unwrap();

    // X = (Z + √λ)² + χ²_{k−1}: the sum Σ(Z_i + μ_i)² with the whole of the
    // non-centrality placed in one coordinate, which is legitimate because only
    // Σμ_i² enters the distribution.
    for (k, lambda) in [(3.0, 2.0), (5.0, 10.0), (2.0, 30.0)] {
        let d = NoncentralChiSquared::new(k, lambda).unwrap();
        let rest = ChiSquared::new(k - 1.0).unwrap();
        let (mean, variance) = sample_moments(N, || {
            let shifted = z.sample(&mut rng) + lambda.sqrt();
            shifted * shifted + rest.sample(&mut rng)
        });
        f.close(
            || format!("NC-chi2({k},{lambda}) mean"),
            mean,
            d.mean(),
            mean_tolerance(d.variance(), N, 5.0),
        );
        let bound = variance_tolerance(d.variance(), d.kurtosis(), N, 5.0)
            .expect("a non-central chi-squared has a fourth moment");
        f.close(
            || format!("NC-chi2({k},{lambda}) variance"),
            variance,
            d.variance(),
            bound,
        );
    }

    // T = (Z + δ) / √(V/ν).
    for (nu, delta) in [(20.0, 1.5), (50.0, -2.0)] {
        let d = NoncentralT::new(nu, delta).unwrap();
        let v = ChiSquared::new(nu).unwrap();
        let (mean, _) = sample_moments(N, || {
            (z.sample(&mut rng) + delta) / (v.sample(&mut rng) / nu).sqrt()
        });
        f.close(
            || format!("NC-t({nu},{delta}) mean"),
            mean,
            d.mean(),
            mean_tolerance(d.variance(), N, 5.0),
        );
    }

    // F = (W/d1) / (V/d2) with W non-central and V central.
    for (d1, d2, lambda) in [(4.0, 20.0, 5.0), (2.0, 30.0, 3.0)] {
        let d = NoncentralF::new(d1, d2, lambda).unwrap();
        let numerator_rest = ChiSquared::new(d1 - 1.0).unwrap();
        let denominator = ChiSquared::new(d2).unwrap();
        let (mean, _) = sample_moments(N, || {
            let shifted = z.sample(&mut rng) + lambda.sqrt();
            let w = shifted * shifted + numerator_rest.sample(&mut rng);
            (w / d1) / (denominator.sample(&mut rng) / d2)
        });
        f.close(
            || format!("NC-F({d1},{d2},{lambda}) mean"),
            mean,
            d.mean(),
            mean_tolerance(d.variance(), N, 5.0),
        );
    }

    f.finish("closed-form moments vs the definition");
}
