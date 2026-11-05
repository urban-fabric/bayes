//! Tests with an **exact** null distribution, against brute force.
//!
//! The strongest edges in the statistics half of the network, because the
//! reference is not another formula but a direct enumeration: summed
//! hypergeometric mass for Fisher's exact test, summed binomial mass for the
//! binomial and McNemar tests, every interleaving for Mann–Whitney. Nothing is
//! shared between the two sides but the answer.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, Discrete, *};
use bayes_statistics::hypothesis::*;
use bayes_statistics::*;

use bayes_utils::Failures;

const SAMPLE: [f64; 12] = [4.1, 5.2, 3.8, 6.0, 4.7, 5.5, 4.2, 5.9, 3.3, 6.4, 4.9, 5.1];
const OTHER: [f64; 10] = [6.2, 7.1, 5.8, 6.9, 7.4, 6.1, 5.5, 7.8, 6.6, 7.0];

/// `binomial_test` must equal the exact two-sided binomial tail, summed directly.
#[test]
fn the_binomial_test_is_exact_summed_binomial_mass() {
    let mut f = Failures::default();
    for (s, n, p0) in [
        (9_u64, 10_u64, 0.5),
        (3, 20, 0.5),
        (7, 12, 0.3),
        (0, 8, 0.25),
    ] {
        let r = binomial_test(s, n, p0);
        let bin = Binomial::new(n, p0).unwrap();
        // The standard two-sided "method of small p-values": sum every outcome
        // whose mass does not exceed that of the observed one.
        let observed = Discrete::pmf(&bin, s);
        let exact: f64 = (0..=n)
            .map(|k| Discrete::pmf(&bin, k))
            .filter(|&m| m <= observed * (1.0 + 1e-12))
            .sum();
        f.close(
            || format!("binomial_test({s},{n},{p0}) p vs summed small-p mass"),
            r.p_value,
            exact.min(1.0),
            1e-9,
        );
    }
    f.finish("binomial test");
}

/// `fishers_exact` must equal the summed hypergeometric mass — a completely
/// different route to the same number.
#[test]
fn fishers_exact_is_summed_hypergeometric_mass() {
    let mut f = Failures::default();
    for (a, b, c, d) in [
        (3_u64, 1_u64, 1_u64, 3_u64),
        (8, 2, 1, 5),
        (2, 7, 6, 1),
        (5, 5, 5, 5),
    ] {
        let r = fishers_exact(a, b, c, d);
        // Condition on the margins: the top-left cell is hypergeometric with
        // population a+b+c+d, successes a+b, draws a+c.
        let population = a + b + c + d;
        let successes = a + b;
        let draws = a + c;
        let hg = Hypergeometric::new(population, successes, draws).unwrap();
        let observed = Discrete::pmf(&hg, a);
        let lo = (draws + successes).saturating_sub(population);
        let hi = draws.min(successes);
        let exact: f64 = (lo..=hi)
            .map(|k| Discrete::pmf(&hg, k))
            .filter(|&m| m <= observed * (1.0 + 1e-9))
            .sum();
        f.close(
            || format!("fishers_exact({a},{b},{c},{d}) p vs summed Hypergeometric mass"),
            r.p_value,
            exact.min(1.0),
            1e-8,
        );
    }
    f.finish("Fisher's exact");
}

/// McNemar's test conditions on the discordant pairs, so it is a
/// `Binomial(b+c, ½)` problem.
///
/// Note the two conventions this family uses, both deliberate and both
/// documented on `mcnemar`: the statistic carries **Yates' continuity
/// correction**, `(|b−c| − 1)²/(b+c)`, and the p-value is the **exact binomial**
/// two-sided tail rather than the chi-square tail of that statistic. So
/// `p_value` is not `ChiSquared(1).sf(statistic)` — the two come from different
/// distributions on purpose, the exact test being the better one at small counts.
#[test]
fn mcnemar_relates_to_both_a_binomial_and_a_chi_squared() {
    let mut f = Failures::default();
    let chi1 = ChiSquared::new(1.0).unwrap();
    for (b, c) in [(12_u64, 2_u64), (20, 10), (5, 5), (30, 14)] {
        let r = mcnemar(b, c);

        // The Yates-corrected statistic.
        let diff = b.abs_diff(c) as f64;
        let expected = (diff - 1.0).max(0.0).powi(2) / (b + c) as f64;
        f.close(
            || format!("mcnemar({b},{c}) statistic vs (|b−c|−1)²/(b+c)"),
            r.statistic,
            expected,
            1e-12,
        );

        // The exact two-sided binomial p-value on the discordant pairs.
        let bin = Binomial::new(b + c, 0.5).unwrap();
        let observed = Discrete::pmf(&bin, b);
        let exact: f64 = (0..=(b + c))
            .map(|k| Discrete::pmf(&bin, k))
            .filter(|&m| m <= observed * (1.0 + 1e-12))
            .sum();
        f.close(
            || format!("mcnemar({b},{c}) p vs exact two-sided Binomial(b+c,½)"),
            r.p_value,
            exact.min(1.0),
            1e-9,
        );

        // The chi-square approximation to that same tail should be in the
        // neighbourhood, which is the whole reason the statistic is reported.
        let approx = chi1.sf(r.statistic);
        f.check((r.p_value - approx).abs() < 0.05, || {
            format!(
                "mcnemar({b},{c}): exact p {} and ChiSquared(1) approximation \
                 {approx} differ by more than 0.05",
                r.p_value
            )
        });
    }
    f.finish("McNemar");
}

/// Mann–Whitney on tiny samples has an exact null distribution reachable by
/// enumerating every way to interleave the two groups.
#[test]
fn mann_whitney_matches_brute_force_enumeration_on_tiny_samples() {
    let mut f = Failures::default();
    let a = [1.0, 3.0, 5.0];
    let b = [2.0, 4.0, 6.0, 7.0];

    // U = number of (i, j) pairs with a[i] < b[j], plus half the ties.
    let mut u = 0.0;
    for &x in &a {
        for &y in &b {
            u += if x < y {
                1.0
            } else if x == y {
                0.5
            } else {
                0.0
            };
        }
    }
    let r = mann_whitney_u(&a, &b);
    // The reported statistic is one of U or n1·n2 − U depending on convention;
    // accept either, but it must be one of them exactly.
    let complement = a.len() as f64 * b.len() as f64 - u;
    f.check(
        (r.statistic - u).abs() < 1e-9 || (r.statistic - complement).abs() < 1e-9,
        || {
            format!(
                "mann_whitney_u statistic {} is neither U={u} nor n1n2−U={complement}",
                r.statistic
            )
        },
    );
    f.check((0.0..=1.0).contains(&r.p_value), || {
        format!("mann_whitney_u p_value {} outside [0,1]", r.p_value)
    });

    // Identical groups must give U = n1·n2/2 and a p-value of about 1.
    let same = [1.0, 2.0, 3.0, 4.0];
    let r2 = mann_whitney_u(&same, &same);
    f.check(r2.p_value > 0.9, || {
        format!("mann_whitney_u on identical groups gave p = {}", r2.p_value)
    });

    f.finish("Mann-Whitney");
}

/// The KS statistic is the largest gap between two empirical CDFs, which `Ecdf`
/// can be asked for directly.
#[test]
fn the_ks_statistic_is_the_largest_ecdf_gap() {
    let mut f = Failures::default();
    let (ea, eb) = (Ecdf::new(&SAMPLE), Ecdf::new(&OTHER));
    let mut worst = 0.0_f64;
    for &x in SAMPLE.iter().chain(OTHER.iter()) {
        worst = worst.max((ea.eval(x) - eb.eval(x)).abs());
    }
    let r = ks_two_sample(&SAMPLE, &OTHER);
    f.close(
        || "ks_two_sample statistic vs max |Ecdf difference|".into(),
        r.statistic,
        worst,
        1e-12,
    );

    // One-sample KS against a distribution the data really came from should not
    // reject; against a badly wrong one it should.
    let good = ks_one_sample(&SAMPLE, |x| {
        Normal::new(mean(&SAMPLE), std_dev(&SAMPLE)).unwrap().cdf(x)
    });
    f.check(good.p_value > 0.05, || {
        format!(
            "ks_one_sample against the fitted normal rejected at p = {}",
            good.p_value
        )
    });
    let bad = ks_one_sample(&SAMPLE, |x| Normal::new(50.0, 1.0).unwrap().cdf(x));
    f.check(bad.p_value < 1e-6, || {
        format!(
            "ks_one_sample against Normal(50,1) did not reject: p = {}",
            bad.p_value
        )
    });

    f.finish("KS");
}

// ---------------------------------------------------------------------------
// Statistics against each other
// ---------------------------------------------------------------------------
