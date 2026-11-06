//! Structural properties of conjugacy that hold whatever the family.
//!
//! The natural parameters form a **monoid**, so updating on all the data at once
//! must equal folding it in one observation at a time, and the empty data set
//! must leave the prior untouched. Those are algebraic, so they are asserted at
//! machine precision.
//!
//! The concentration check is the one statistical test here: a posterior must
//! tighten as data accumulates, and its spread must approach the
//! Bernstein–von Mises rate `√(p(1−p)/n)`.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::Failures;
use bayes_distribution_ext::{Continuous, Moments, *};

/// Conjugacy must be **sequential**: updating on all the data at once has to
/// equal folding the data in one observation at a time, because the natural
/// parameters form a monoid.
#[test]
fn conjugate_updates_are_associative_over_the_data() {
    let mut f = Failures::default();

    let data = [true, false, true, true, false, true, true, true];
    let model = BetaBernoulli::new(Beta::new(2.0, 3.0).unwrap());
    let all_at_once = model.posterior(&data);
    // Fold one datum at a time, re-seating the model on each intermediate posterior.
    let mut rolling = BetaBernoulli::new(Beta::new(2.0, 3.0).unwrap());
    for &d in &data {
        rolling = BetaBernoulli::new(rolling.posterior(&[d]));
    }
    let sequential = rolling.prior();
    for p in [0.1, 0.4, 0.6, 0.9] {
        f.close(
            || format!("BetaBernoulli: batch vs sequential posterior at {p}"),
            all_at_once.pdf(p),
            sequential.pdf(p),
            1e-10,
        );
    }

    let counts: [u64; 5] = [1, 3, 0, 2, 5];
    let gp = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
    let batch = gp.posterior(&counts);
    let mut roll = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
    for &c in &counts {
        roll = GammaPoisson::new(roll.posterior(&[c]));
    }
    let seq = roll.prior();
    for lambda in [0.5, 1.5, 3.0] {
        f.close(
            || format!("GammaPoisson: batch vs sequential posterior at {lambda}"),
            batch.pdf(lambda),
            seq.pdf(lambda),
            1e-10,
        );
    }

    // The empty data set must leave the prior untouched.
    let untouched = BetaBernoulli::new(Beta::new(2.0, 3.0).unwrap()).posterior(&[]);
    for p in [0.2, 0.5, 0.8] {
        f.close(
            || format!("BetaBernoulli: empty-data posterior = prior at {p}"),
            untouched.pdf(p),
            Beta::new(2.0, 3.0).unwrap().pdf(p),
            1e-12,
        );
    }

    f.finish("conjugate associativity");
}

/// A conjugate posterior should contract toward the truth as data accumulates —
/// the one check here that is statistical rather than algebraic.
#[test]
fn posteriors_concentrate_on_the_generating_parameter() {
    use rand::SeedableRng;
    use rand::distr::Distribution as _;
    use rand_chacha::ChaCha8Rng;

    let mut f = Failures::default();
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);

    let true_p = 0.35;
    let source = Bernoulli::new(true_p).unwrap();
    let draws: Vec<bool> = (0..20_000).map(|_| source.sample(&mut rng)).collect();

    let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap());
    let mut last_sd = f64::INFINITY;
    for n in [100usize, 1_000, 20_000] {
        let post = model.posterior(&draws[..n]);
        let sd = post.variance().sqrt();
        // Each tenfold increase in data must shrink the posterior spread.
        f.check(sd < last_sd, || {
            format!("posterior sd did not shrink at n={n}: {sd} vs {last_sd}")
        });
        last_sd = sd;
        // And the posterior mean must sit within a few posterior sds of the truth.
        f.check((post.mean() - true_p).abs() < 5.0 * sd, || {
            format!(
                "at n={n} posterior mean {} is >5sd from {true_p}",
                post.mean()
            )
        });
    }
    // Bernstein-von Mises: the posterior sd approaches √(p(1-p)/n).
    let post = model.posterior(&draws);
    let asymptotic = (true_p * (1.0 - true_p) / 20_000.0).sqrt();
    f.close(
        || "posterior sd vs √(p(1-p)/n)".into(),
        post.variance().sqrt(),
        asymptotic,
        0.05,
    );

    f.finish("posterior concentration");
}
