//! Against a simulated t-test — the reason this family exists.
//!
//! Everything else in this suite checks that a series converges to the right
//! number. This checks that the right number is the one a user wants: the
//! probability that a t-test rejects, when the effect is real.

use bayes_distribution_ext::{ContinuousCdf, NoncentralT, Normal, StudentT};
use bayes_statistics::hypothesis::one_sample_t;
use bayes_utils::Failures;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

/// The **exact** power of a one-sample t-test is a non-central t tail, and it
/// must agree with brute force.
///
/// This is also the check that retires an approximation. The `sample_size`
/// example routes power through the normal because no non-central t existed
/// when it was written; the exact figure differs from that approximation by a
/// point or two at small `n`, and here both are compared against the truth.
#[test]
fn exact_t_test_power_matches_simulation() {
    let mut f = Failures::default();
    let alpha = 0.05;
    let z = Normal::new(0.0, 1.0).unwrap();

    for (n, effect) in [(16_usize, 0.5), (34, 0.5), (64, 0.5), (25, 0.8), (50, 0.3)] {
        let nu = n as f64 - 1.0;
        let crit: f64 = StudentT::new(nu).unwrap().inverse_cdf(1.0 - alpha / 2.0);
        let delta = effect * (n as f64).sqrt();
        let nct = NoncentralT::new(nu, delta).unwrap();
        // Two-sided power: reject when |T| > crit.
        let exact = (1.0 - nct.cdf(crit)) + nct.cdf(-crit);

        // Simulate the same experiment.
        let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
        let source = Normal::new(effect, 1.0).unwrap();
        let trials = 20_000;
        let rejections = (0..trials)
            .filter(|_| {
                let xs: Vec<f64> = (0..n).map(|_| source.sample(&mut rng)).collect();
                one_sample_t(&xs, 0.0).p_value < alpha
            })
            .count();
        let simulated = rejections as f64 / trials as f64;
        // The standard error at 20000 trials is at most 0.0035, so six of those
        // is 0.021.
        f.check((exact - simulated).abs() < 0.021, || {
            format!("n={n}, d={effect}: exact power {exact:.4} vs simulated {simulated:.4}")
        });

        // And the normal approximation the example had to use is measurably
        // worse, which is the reason this family was added.
        let approximate: f64 = z.sf(crit - delta);
        f.check(
            (exact - simulated).abs() <= (approximate - simulated).abs() + 0.01,
            || {
                format!(
                    "n={n}, d={effect}: the exact non-central t power {exact:.4} should be at \
                     least as close to {simulated:.4} as the normal approximation \
                     {approximate:.4}"
                )
            },
        );
    }
    f.finish("exact t-test power");
}
