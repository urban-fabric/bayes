//! Proportion intervals, held to coverage rather than to shape.
//!
//! A binomial proportion is the one place where coverage — the defining
//! property of a confidence interval — is computable exactly rather than by
//! simulation, because the sample space is finite. That makes it the right
//! thing to assert here, and it is a far stronger statement than the shape
//! checks that any plausible-looking interval would pass.

use bayes_statistics::{proportion_clopper_pearson, proportion_wilson};
use bayes_utils::Failures;

/// Clopper–Pearson delivers the coverage it promises; Wald does not.
///
/// Coverage is the *defining* property of a confidence interval, and for a
/// proportion it is computable exactly rather than by simulation: for a true
/// `p`, sum the binomial mass of every count whose interval contains `p`. The
/// Clopper–Pearson construction guarantees that sum is at least the nominal
/// level for **every** `p`, which is what "exact" means for it.
///
/// The Wald interval is included as the counter-example that motivates the
/// others: its coverage collapses near the boundary, and at `p = 0.02` with
/// `n = 20` it is nowhere near 95%. A suite that only checked "the interval is
/// ordered and inside [0,1]" would pass Wald happily.
///
/// A first attempt asserted that Clopper–Pearson *contains* Wilson at every
/// count. That is false, and interestingly so: at `0/50` and level `0.95` the
/// exact upper bound is `0.0711` while Wilson's is `0.0713`. Clopper–Pearson is
/// usually the wider of the two but not uniformly, so the comparison worth
/// making is about coverage, not about nesting.
#[test]
fn clopper_pearson_covers_at_least_its_nominal_level_and_wald_does_not() {
    use bayes_distribution_ext::{Binomial, Discrete};

    let mut f = Failures::default();
    let coverage = |n: u64, truth: f64, build: &dyn Fn(u64, u64, f64) -> _, level: f64| -> f64 {
        let binomial = Binomial::new(n, truth).expect("a valid binomial");
        (0..=n)
            .filter(|&k| {
                let interval: bayes_statistics::ConfidenceInterval = build(k, n, level);
                interval.contains(truth)
            })
            .map(|k| binomial.pmf(k))
            .sum()
    };

    for n in [10_u64, 20, 50] {
        for level in [0.90, 0.95] {
            for truth in [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] {
                let exact = coverage(n, truth, &proportion_clopper_pearson, level);
                f.check(exact >= level - 1e-9, || {
                    format!(
                        "Clopper-Pearson at n={n}, p={truth}, level={level} covered only \
                         {exact:.6} — an exact interval must never fall below nominal"
                    )
                });
            }
        }
    }

    // And the Wald interval, which is why the others exist: at a small `p` its
    // coverage is far below nominal.
    let wald_coverage = coverage(20, 0.02, &bayes_statistics::proportion_wald, 0.95);
    assert!(
        wald_coverage < 0.90,
        "the Wald interval is supposed to under-cover badly near the boundary; at n=20, \
         p=0.02 it covered {wald_coverage:.6}, which would mean this comparison has lost its \
         point"
    );
    let exact_coverage = coverage(20, 0.02, &proportion_clopper_pearson, 0.95);
    assert!(
        exact_coverage > wald_coverage,
        "Clopper-Pearson ({exact_coverage:.6}) should cover better than Wald \
         ({wald_coverage:.6}) here"
    );
    f.finish("proportion coverage");
}

/// Every proportion interval is ordered, inside `[0, 1]`, and — for the two
/// that guarantee it — contains its own point estimate.
///
/// Agresti–Coull is deliberately excluded from the last check: it centres on a
/// shrunken estimate, so containing `p̂` is not one of its properties.
#[test]
fn the_proportion_intervals_are_well_formed() {
    let mut f = Failures::default();
    for n in [1_u64, 2, 5, 10, 50, 200] {
        for successes in 0..=n {
            for level in [0.90, 0.95, 0.99] {
                let estimate = successes as f64 / n as f64;
                for (name, interval) in [
                    (
                        "Clopper-Pearson",
                        proportion_clopper_pearson(successes, n, level),
                    ),
                    ("Wilson", proportion_wilson(successes, n, level)),
                ] {
                    f.check((0.0..=1.0).contains(&interval.lower), || {
                        format!(
                            "{name}({successes}/{n}, {level}) lower = {} escapes [0,1]",
                            interval.lower
                        )
                    });
                    f.check((0.0..=1.0).contains(&interval.upper), || {
                        format!(
                            "{name}({successes}/{n}, {level}) upper = {} escapes [0,1]",
                            interval.upper
                        )
                    });
                    f.check(interval.lower <= interval.upper, || {
                        format!("{name}({successes}/{n}, {level}) is inverted")
                    });
                    f.check(interval.contains(estimate), || {
                        format!(
                            "{name}({successes}/{n}, {level}) = [{}, {}] excludes its own \
                             estimate {estimate}",
                            interval.lower, interval.upper
                        )
                    });
                }
                // A higher level is never narrower.
                let narrow = proportion_wilson(successes, n, 0.90);
                let wide = proportion_wilson(successes, n, 0.99);
                f.check(wide.width() >= narrow.width() - 1e-12, || {
                    format!(
                        "Wilson at {successes}/{n}: the 99% interval ({}) is narrower than the \
                         90% one ({})",
                        wide.width(),
                        narrow.width()
                    )
                });
            }
        }
    }
    f.finish("proportion interval shape");
}
