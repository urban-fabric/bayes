//! **Did the new checkout page convert better?** — the same A/B test answered
//! two ways, frequentist and Bayesian, so you can see what each one gives you.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p bayes-statistics --example ab_test
//! ```
//!
//! Every number printed is also asserted, so this file doubles as a test: if the
//! library's answers drift, the example fails rather than quietly printing
//! something wrong.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here: every
// `expect` below is on a literal parameter that is valid by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_distribution_ext::{Beta, BetaBernoulli, Conjugate, ContinuousCdf, Moments};
use bayes_statistics::hypothesis::chi_squared_independence;
use bayes_statistics::{proportion_difference_wald, proportion_wilson};

fn main() {
    // ---------------------------------------------------------------------
    // The data: two variants, a few thousand sessions each.
    // ---------------------------------------------------------------------
    let (a_conversions, a_sessions) = (182_u64, 2_410_u64);
    let (b_conversions, b_sessions) = (231_u64, 2_385_u64);

    let a_rate = a_conversions as f64 / a_sessions as f64;
    let b_rate = b_conversions as f64 / b_sessions as f64;

    println!("A/B test: checkout conversion");
    println!(
        "  A: {a_conversions:>4} / {a_sessions} = {:.3}%",
        a_rate * 100.0
    );
    println!(
        "  B: {b_conversions:>4} / {b_sessions} = {:.3}%",
        b_rate * 100.0
    );
    println!(
        "  observed lift: {:+.3} percentage points",
        (b_rate - a_rate) * 100.0
    );

    // ---------------------------------------------------------------------
    // Frequentist: is the difference bigger than sampling noise?
    // ---------------------------------------------------------------------
    println!("\n-- frequentist --");

    // A 2x2 contingency table of converted / did-not-convert by variant.
    let table = vec![
        vec![a_conversions as f64, (a_sessions - a_conversions) as f64],
        vec![b_conversions as f64, (b_sessions - b_conversions) as f64],
    ];
    let test = chi_squared_independence(&table);
    println!(
        "  chi-squared independence: X² = {:.4}, p = {:.6}",
        test.statistic, test.p_value
    );

    // A confidence interval on the *difference* is more useful than a p-value,
    // because it carries the effect size as well as the significance.
    let diff =
        proportion_difference_wald(b_conversions, b_sessions, a_conversions, a_sessions, 0.95);
    println!(
        "  95% CI on the difference: [{:+.4}, {:+.4}] ({:+.2} to {:+.2} points)",
        diff.lower,
        diff.upper,
        diff.lower * 100.0,
        diff.upper * 100.0
    );

    // Per-variant intervals. Wilson rather than Wald: it behaves at small counts
    // and never leaves [0, 1].
    let a_ci = proportion_wilson(a_conversions, a_sessions, 0.95);
    let b_ci = proportion_wilson(b_conversions, b_sessions, 0.95);
    println!("  A 95% Wilson: [{:.4}, {:.4}]", a_ci.lower, a_ci.upper);
    println!("  B 95% Wilson: [{:.4}, {:.4}]", b_ci.lower, b_ci.upper);

    assert!(
        test.p_value < 0.01,
        "the effect in this data is significant at 1%"
    );
    assert!(diff.lower > 0.0, "the whole interval favours B");
    assert!(a_ci.lower <= a_rate && a_rate <= a_ci.upper);
    assert!(b_ci.lower <= b_rate && b_rate <= b_ci.upper);

    // ---------------------------------------------------------------------
    // Bayesian: what do we now believe each rate is?
    // ---------------------------------------------------------------------
    println!("\n-- bayesian --");

    // A Beta(1,1) prior is uniform on [0,1]: no opinion before the data. The
    // conjugate update is exact, so there is no sampler and no convergence to
    // worry about.
    let prior = Beta::new(1.0, 1.0).expect("valid prior");
    let model = BetaBernoulli::new(prior);

    let a_trials = trials(a_conversions, a_sessions);
    let b_trials = trials(b_conversions, b_sessions);
    let a_post = model.posterior(&a_trials);
    let b_post = model.posterior(&b_trials);

    println!(
        "  A posterior: mean {:.4}, 95% credible [{:.4}, {:.4}]",
        a_post.mean(),
        a_post.inverse_cdf(0.025),
        a_post.inverse_cdf(0.975)
    );
    println!(
        "  B posterior: mean {:.4}, 95% credible [{:.4}, {:.4}]",
        b_post.mean(),
        b_post.inverse_cdf(0.025),
        b_post.inverse_cdf(0.975)
    );

    // The posterior mean is (successes + 1)/(sessions + 2) under this prior —
    // essentially the observed rate, nudged toward ½ by one pseudo-observation.
    assert!((a_post.mean() - (a_conversions + 1) as f64 / (a_sessions + 2) as f64).abs() < 1e-12);

    // P(rate_B > rate_A). The posteriors are independent, so this is a double
    // integral; one dimension is closed form, so only one needs quadrature.
    let p_b_better = probability_b_beats_a(&a_post, &b_post);
    println!("  P(B > A) = {:.5}", p_b_better);

    // Expected lift, and the risk of being wrong.
    println!(
        "  posterior mean lift: {:+.3} points",
        (b_post.mean() - a_post.mean()) * 100.0
    );
    println!("  P(B worse than A)  = {:.5}", 1.0 - p_b_better);

    assert!(
        p_b_better > 0.99,
        "B is very likely better: P = {p_b_better}"
    );

    // The two approaches should agree on the direction and roughly on strength:
    // a one-sided frequentist p-value and P(A >= B) answer different questions
    // but land close under a flat prior.
    let one_sided_p = test.p_value / 2.0;
    let bayes_tail = 1.0 - p_b_better;
    println!(
        "\n  one-sided p = {:.5} vs P(B worse) = {:.5} — close, as a flat prior implies",
        one_sided_p, bayes_tail
    );
    assert!(
        (one_sided_p - bayes_tail).abs() < 0.01,
        "flat-prior Bayes and the frequentist tail should roughly agree"
    );

    println!("\nall assertions held");
}

/// Expand a success count into the Bernoulli trials the conjugate model consumes.
fn trials(successes: u64, total: u64) -> Vec<bool> {
    let mut v = vec![true; successes as usize];
    v.resize(total as usize, false);
    v
}

/// `P(B > A)` for independent Beta posteriors, by integrating `A`'s density
/// against `B`'s survival function.
///
/// `∫₀¹ pdf_A(x)·P(B > x) dx`. Integration runs in `A`'s quantile space —
/// substituting `x = Q_A(u)` turns it into `∫₀¹ P(B > Q_A(u)) du` — which removes
/// the density from the integrand entirely and so is accurate even when the
/// posterior is sharply peaked, as it is here with a few thousand observations.
fn probability_b_beats_a(a_post: &Beta, b_post: &Beta) -> f64 {
    let n = 20_000;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = i as f64 / n as f64;
        let u = u.clamp(1e-12, 1.0 - 1e-12);
        let x = a_post.inverse_cdf(u);
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += w * b_post.sf(x);
    }
    acc / n as f64
}
