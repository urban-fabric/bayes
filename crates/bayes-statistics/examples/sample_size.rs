//! **How many observations do I need?** — power analysis, with the analytic
//! answer checked against simulation.
//!
//! ```sh
//! cargo run -p bayes-statistics --example sample_size
//! ```
//!
//! This is the question people most often skip and most often regret. A study
//! powered at 40% is a coin flip dressed up as evidence: the effect is real, the
//! experiment just cannot see it, and the null result gets read as "no effect".
//!
//! The example is also a worked cross-check of the kind this repository's test
//! suite is built on. The analytic power formula and a brute-force simulation
//! reach the same number by completely different routes — one through the normal
//! quantile, the other by running 20,000 experiments and counting rejections —
//! so when they agree to within Monte Carlo error, both are probably right.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here: every
// `expect` below is on a literal parameter that is valid by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, Normal, StudentT};
use bayes_statistics::hypothesis::{one_sample_t, two_sample_t};
use bayes_statistics::{mean, proportion_wilson};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("Sample size and power for a two-sample comparison\n");

    // ---------------------------------------------------------------------
    // The planning question: we expect the new process to raise yield by
    // 0.4 standard deviations. How many units per arm?
    // ---------------------------------------------------------------------
    let effect = 0.4_f64; // Cohen's d
    let alpha = 0.05;
    let target_power = 0.80;

    // The textbook formula for a two-sided two-sample z-test:
    //     n per arm = 2*(z_{1-a/2} + z_{power})^2 / d^2
    let z = Normal::new(0.0, 1.0).expect("standard normal");
    let z_alpha: f64 = z.inverse_cdf(1.0 - alpha / 2.0);
    let z_power: f64 = z.inverse_cdf(target_power);
    let n_analytic = 2.0 * (z_alpha + z_power).powi(2) / (effect * effect);

    println!("-- planning --");
    println!("  effect size d      = {effect}");
    println!("  alpha              = {alpha}");
    println!("  target power       = {target_power}");
    println!("  z_(1-a/2)          = {z_alpha:.6}");
    println!("  z_power            = {z_power:.6}");
    println!(
        "  n per arm (formula) = {:.1}  -> use {}",
        n_analytic,
        n_analytic.ceil()
    );
    assert!(
        (z_alpha - 1.959_963_985).abs() < 1e-6,
        "the 97.5% normal quantile"
    );
    assert!(
        (z_power - 0.841_621_234).abs() < 1e-6,
        "the 80% normal quantile"
    );

    let n_per_arm = n_analytic.ceil() as usize;

    // ---------------------------------------------------------------------
    // Now the cross-check: run the experiment 20,000 times and count how
    // often it rejects. That fraction *is* the power.
    // ---------------------------------------------------------------------
    println!("\n-- simulated power at n = {n_per_arm} per arm --");
    let empirical = simulate_power(n_per_arm, effect, alpha, 20_000, 1);
    let ci = proportion_wilson((empirical * 20_000.0).round() as u64, 20_000, 0.95);
    println!("  rejections / trials = {empirical:.4}");
    println!("  95% Wilson interval = [{:.4}, {:.4}]", ci.lower, ci.upper);
    println!("  the planning target was {target_power}");

    // The t-test has slightly less power than the z-test the formula assumes,
    // because the variance is estimated rather than known. At this n that costs
    // about a point of power, so the interval should sit just under the target.
    assert!(
        ci.lower < target_power + 0.03 && ci.upper > target_power - 0.05,
        "simulated power {empirical} should bracket the planned {target_power}"
    );

    // ---------------------------------------------------------------------
    // Power is a curve, not a number. This is the part worth internalizing.
    // ---------------------------------------------------------------------
    println!("\n-- power vs sample size (d = {effect}) --");
    println!("    n/arm   analytic   simulated");
    let mut last_power = 0.0;
    for n in [10_usize, 25, 50, n_per_arm, 200] {
        let analytic = analytic_power(n, effect, alpha);
        let sim = simulate_power(n, effect, alpha, 8_000, 7);
        println!("    {n:>5}    {analytic:.4}     {sim:.4}");
        // Power must rise with n, and the two routes must agree to within the
        // Monte Carlo standard error (~0.006 at 8000 trials) plus the z-vs-t gap.
        assert!(analytic > last_power, "power must increase with n");
        assert!(
            (analytic - sim).abs() < 0.04,
            "analytic {analytic} and simulated {sim} disagree at n = {n}"
        );
        last_power = analytic;
    }

    println!("\n-- power vs effect size (n = 50 per arm) --");
    println!("    d       analytic   simulated");
    for d in [0.1_f64, 0.25, 0.5, 0.8, 1.2] {
        let analytic = analytic_power(50, d, alpha);
        let sim = simulate_power(50, d, alpha, 8_000, 11);
        println!("    {d:<6}  {analytic:.4}     {sim:.4}");
        assert!(
            (analytic - sim).abs() < 0.04,
            "analytic {analytic} and simulated {sim} disagree at d = {d}"
        );
    }

    // ---------------------------------------------------------------------
    // The other half: a test must hold its false-positive rate. If the null
    // is true, it should reject exactly `alpha` of the time.
    // ---------------------------------------------------------------------
    println!("\n-- false-positive rate under the null (d = 0) --");
    for n in [10_usize, 50, 200] {
        let rate = simulate_power(n, 0.0, alpha, 20_000, 13);
        println!("    n = {n:>3}: rejected {rate:.4} of the time (alpha = {alpha})");
        // Standard error at 20000 trials is sqrt(.05*.95/20000) = 0.0015, so
        // six of those is 0.009.
        assert!(
            (rate - alpha).abs() < 0.01,
            "at n = {n} the test rejected {rate} of true nulls, not {alpha}"
        );
    }
    println!("  a test that rejects more than alpha of true nulls is broken;");
    println!("  one that rejects less is wasting power.");

    // ---------------------------------------------------------------------
    // One-sample planning, for completeness, and the exact t critical value.
    // ---------------------------------------------------------------------
    println!("\n-- one-sample: detecting a shift of 0.5 sd --");
    for n in [16_usize, 34, 64] {
        let t = StudentT::new(n as f64 - 1.0).expect("positive freedom");
        let crit: f64 = t.inverse_cdf(1.0 - alpha / 2.0);
        // Non-centrality is d*sqrt(n); approximate the power through the normal.
        let ncp = 0.5 * (n as f64).sqrt();
        let power: f64 = z.sf(crit - ncp);
        println!("    n = {n:>3}: t critical {crit:.4}, approximate power {power:.4}");
        let sim = simulate_one_sample_power(n, 0.5, alpha, 8_000, 17);
        println!("             simulated power {sim:.4}");
        assert!(
            (power - sim).abs() < 0.05,
            "one-sample power {power} vs simulated {sim} at n = {n}"
        );
    }

    println!("\nall assertions held");
}

/// The analytic power of a two-sided two-sample test, via the normal
/// approximation: `P(reject | effect) = Phi(-z_a/2 + d*sqrt(n/2))` plus the
/// negligible far tail.
fn analytic_power(n_per_arm: usize, effect: f64, alpha: f64) -> f64 {
    let z = Normal::new(0.0, 1.0).expect("standard normal");
    let z_alpha: f64 = z.inverse_cdf(1.0 - alpha / 2.0);
    let shift = effect * (n_per_arm as f64 / 2.0).sqrt();
    let upper: f64 = z.sf(z_alpha - shift);
    let lower: f64 = z.cdf(-z_alpha - shift);
    upper + lower
}

/// The same quantity by brute force: run the experiment `trials` times and count
/// how often the t-test rejects.
fn simulate_power(n_per_arm: usize, effect: f64, alpha: f64, trials: usize, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let control = Normal::new(0.0, 1.0).expect("standard normal");
    let treatment = Normal::new(effect, 1.0).expect("standard normal");
    let mut rejections = 0;
    for _ in 0..trials {
        let a: Vec<f64> = (0..n_per_arm).map(|_| control.sample(&mut rng)).collect();
        let b: Vec<f64> = (0..n_per_arm).map(|_| treatment.sample(&mut rng)).collect();
        if two_sample_t(&a, &b).p_value < alpha {
            rejections += 1;
        }
    }
    rejections as f64 / trials as f64
}

/// The one-sample analogue, testing a sample mean against zero.
fn simulate_one_sample_power(n: usize, effect: f64, alpha: f64, trials: usize, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let source = Normal::new(effect, 1.0).expect("standard normal");
    let mut rejections = 0;
    for _ in 0..trials {
        let xs: Vec<f64> = (0..n).map(|_| source.sample(&mut rng)).collect();
        // Guard against the degenerate case; `mean` is only used for the report.
        let _ = mean(&xs);
        if one_sample_t(&xs, 0.0).p_value < alpha {
            rejections += 1;
        }
    }
    rejections as f64 / trials as f64
}
